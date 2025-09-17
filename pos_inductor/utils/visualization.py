"""Visualization utilities for training curves and confusion matrices."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import os


def plot_training_curves(experiment_name: str,
                         loss_values: List[float],
                         val_values: List[float],
                         vocab_loss_values: Optional[List[float]] = None,
                         div_loss_values: Optional[List[float]] = None,
                         char_loss_values: Optional[List[float]] = None,
                         save_dir: str = "plots") -> None:
    """
    Plot training and validation curves.
    
    Args:
        experiment_name: Name of the experiment
        loss_values: Training loss values
        val_values: Validation loss values
        vocab_loss_values: Vocabulary reconstruction loss values
        div_loss_values: Diversity loss values
        char_loss_values: Character reconstruction loss values
        save_dir: Directory to save plots
    """
    # Set style
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })

    plt.figure(figsize=(12, 8))

    # Plot main losses
    plt.plot(np.array(loss_values), 'r-', linewidth=2, label='Training Loss')
    plt.plot(np.array(val_values), 'b-', linewidth=2, label='Validation Loss')

    # Plot component losses if available
    if vocab_loss_values is not None:
        plt.plot(np.array(vocab_loss_values), 'm--', linewidth=1.5, alpha=0.8,
                 label='Vocab Reconstruction Loss')

    if char_loss_values is not None:
        plt.plot(np.array(char_loss_values), 'y--', linewidth=1.5, alpha=0.8,
                 label='Char Reconstruction Loss')

    if div_loss_values is not None:
        # Scale diversity loss for better visualization
        div_loss_scaled = [loss * 10 for loss in div_loss_values]
        plt.plot(np.array(div_loss_scaled), 'c--', linewidth=1.5, alpha=0.8,
                 label='Diversity Loss (×10)')

    # Formatting
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title(f"{experiment_name} Training & Validation Loss Curves")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{experiment_name}_loss_curves.png", dpi=300, bbox_inches='tight')

    # Save data
    data = {
        "epoch": np.arange(len(loss_values)),
        "training_loss": loss_values,
        "validation_loss": val_values
    }

    if vocab_loss_values is not None:
        data["vocab_loss"] = vocab_loss_values
    if div_loss_values is not None:
        data["diversity_loss"] = div_loss_values
    if char_loss_values is not None:
        data["char_loss"] = char_loss_values

    df = pd.DataFrame(data)
    df.to_csv(f"{save_dir}/{experiment_name}_loss_data.csv", index=False)

    plt.show()


def plot_confusion_matrix(counts: Dict,
                          experiment_name: str,
                          index2label: Optional[Dict[int, str]] = None,
                          is_m1: bool = False,
                          save_dir: str = "plots") -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        counts: Count dictionary (predicted -> gold counts)
        experiment_name: Name of the experiment
        index2label: Index to label mapping
        is_m1: Whether this is M1 mapping or raw predictions
        save_dir: Directory to save plots
    """
    # Standard tag order for POS tags
    tag_order = [
        'NOUN', 'PROPN', 'PRON', 'NUM',  # Nominal Elements
        'ADJ', 'DET',  # Nominal Modifiers
        'VERB', 'AUX',  # Verbal Elements
        'ADV', 'PART',  # Adverbials
        'ADP', 'CCONJ', 'SCONJ'  # Connectives
    ]

    # Convert counts to percentages
    if is_m1:
        counts_eng_labels = {pred: {gold: count for gold, count in gold_counts.items()}
                             for pred, gold_counts in counts.items()}
        pred_tags = tag_order
    else:
        counts_eng_labels = {pred: {index2label[gold]: count for gold, count in gold_counts.items()}
                             for pred, gold_counts in counts.items()}
        pred_tags = sorted(counts_eng_labels.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))

    # Convert to percentages
    percents_eng_labels = {}
    for k, v in counts_eng_labels.items():
        total = sum(v.values())
        if total > 0:
            percents_eng_labels[k] = {label: round(count / total * 100, 1) for label, count in v.items()}

    # Create DataFrame
    df = pd.DataFrame.from_dict(percents_eng_labels, orient='index')
    df = df.reindex(columns=tag_order, fill_value=0)
    df = df.reindex(index=pred_tags, fill_value=0)
    df = df.T  # Transpose so gold labels are on y-axis

    # Set style
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
    })

    # Create heatmap
    plt.figure(figsize=(14, 10))

    # Custom color palette
    custom_palette = sns.color_palette("YlOrRd", as_cmap=True)

    if is_m1:
        # Calculate M1 accuracy
        m1_accuracy = calculate_m1_accuracy(counts)

        sns.heatmap(df, cmap=custom_palette, annot=True, fmt='.1f',
                    cbar_kws={'label': 'Percentage', 'shrink': 0.8})
        title = f"{experiment_name} Many-to-One Mapping\nM1 Accuracy: {m1_accuracy:.2f}%"
        filename = f"{experiment_name}_m1_confusion_matrix.png"
    else:
        sns.heatmap(df, cmap=custom_palette,
                    cbar_kws={'label': 'Percentage', 'shrink': 0.8})
        title = f"{experiment_name} Predicted vs Gold Labels"
        filename = f"{experiment_name}_confusion_matrix.png"

    plt.title(title, pad=20)
    plt.xlabel('Predicted Labels', labelpad=10)
    plt.ylabel('Gold Labels', labelpad=10)
    plt.tight_layout()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{filename}", dpi=300, bbox_inches='tight')
    plt.show()


def calculate_m1_accuracy(m1_totals: Dict) -> float:
    """Calculate M1 accuracy from totals dictionary."""
    correct = 0
    total = 0

    for label, count_dict in m1_totals.items():
        if label in count_dict:
            correct += count_dict[label]
        total += sum(count_dict.values())

    return (correct / total) * 100 if total > 0 else 0


def plot_codebook_usage(usage_stats: Dict,
                        experiment_name: str,
                        save_dir: str = "plots") -> None:
    """
    Plot codebook usage statistics.
    
    Args:
        usage_stats: Dictionary with usage statistics
        experiment_name: Name of the experiment
        save_dir: Directory to save plots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Usage distribution
    ax1.bar(range(len(usage_stats['usage_per_tag'])),
            sorted(usage_stats['usage_per_tag'], reverse=True))
    ax1.set_title('Tag Usage Distribution (Sorted)')
    ax1.set_xlabel('Tag Rank')
    ax1.set_ylabel('Usage Frequency')
    ax1.grid(True, alpha=0.3)

    # Most used tags
    most_used = usage_stats['most_used_tags']
    most_used_values = [usage_stats['usage_per_tag'][i] for i in most_used]
    ax2.bar(range(len(most_used)), most_used_values)
    ax2.set_title('Top 10 Most Used Tags')
    ax2.set_xlabel('Tag Index')
    ax2.set_ylabel('Usage Frequency')
    ax2.set_xticks(range(len(most_used)))
    ax2.set_xticklabels([f'T{i}' for i in most_used], rotation=45)
    ax2.grid(True, alpha=0.3)

    # Usage histogram
    ax3.hist(usage_stats['usage_per_tag'], bins=20, alpha=0.7, edgecolor='black')
    ax3.set_title('Usage Frequency Histogram')
    ax3.set_xlabel('Usage Frequency')
    ax3.set_ylabel('Number of Tags')
    ax3.grid(True, alpha=0.3)

    # Summary statistics
    stats_text = f"""Summary Statistics:
Total Tags: {len(usage_stats['usage_per_tag'])}
Active Tags (>0.1%): {usage_stats['num_active_tags']}
Entropy: {usage_stats['entropy']:.3f}
Max Usage: {max(usage_stats['usage_per_tag']):.3f}
Min Usage: {min(usage_stats['usage_per_tag']):.3f}
Mean Usage: {np.mean(usage_stats['usage_per_tag']):.3f}"""

    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')

    plt.suptitle(f'{experiment_name} Codebook Usage Analysis', fontsize=16)
    plt.tight_layout()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{experiment_name}_codebook_usage.png", dpi=300, bbox_inches='tight')
    plt.show()


def create_top_words_table(word_tag_counts: Dict,
                           index2word: Dict[int, str],
                           index2label: Optional[Dict[int, str]] = None,
                           pred_to_m1: Optional[Dict[int, int]] = None,
                           experiment_name: str = "experiment",
                           save_dir: str = "plots",
                           top_k: int = 10) -> pd.DataFrame:
    """
    Create table of top words for each predicted tag.
    
    Args:
        word_tag_counts: Dictionary mapping tag -> word -> count
        index2word: Index to word mapping
        index2label: Index to label mapping (optional)
        pred_to_m1: Predicted tag to M1 mapping (optional)
        experiment_name: Name of the experiment
        save_dir: Directory to save table
        top_k: Number of top words to show per tag
        
    Returns:
        DataFrame with top words per tag
    """
    data = []

    for tag, word_dict in word_tag_counts.items():
        # Get M1 label if available
        m1_tag = None
        if pred_to_m1 is not None and index2label is not None:
            m1_tag = index2label.get(pred_to_m1.get(tag, 0), f"UNKNOWN_{tag}")

        # Sort words by count and get top k
        word_dict_sorted = dict(sorted(word_dict.items(), key=lambda item: item[1], reverse=True))
        top_word_indices = list(word_dict_sorted.keys())[:top_k]
        top_words = [index2word.get(k, f"UNK_{k}") for k in top_word_indices]
        top_counts = [word_dict_sorted[k] for k in top_word_indices]

        # Create row
        row = [f"TAG_{tag}"]
        if m1_tag is not None:
            row.append(m1_tag)

        # Add top words with counts
        for i in range(top_k):
            if i < len(top_words):
                row.append(f"{top_words[i]} ({top_counts[i]})")
            else:
                row.append("")

        data.append(row)

    # Create DataFrame
    columns = ['Predicted_Tag']
    if pred_to_m1 is not None:
        columns.append('M1_Label')
    columns.extend([f'Word_{i + 1}' for i in range(top_k)])

    df = pd.DataFrame(data, columns=columns)

    # Save to CSV
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/{experiment_name}_top_words.csv", index=False)

    return df


def plot_loss_comparison(experiments: Dict[str, Dict[str, List[float]]],
                         save_dir: str = "plots") -> None:
    """
    Compare training curves across multiple experiments.
    
    Args:
        experiments: Dictionary mapping experiment_name -> loss_type -> values
        save_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    loss_types = ['training_loss', 'validation_loss', 'diversity_loss', 'vocab_loss']
    titles = ['Training Loss', 'Validation Loss', 'Diversity Loss', 'Vocabulary Loss']

    colors = plt.cm.Set1(np.linspace(0, 1, len(experiments)))

    for i, (loss_type, title) in enumerate(zip(loss_types, titles)):
        ax = axes[i]

        for j, (exp_name, losses) in enumerate(experiments.items()):
            if loss_type in losses and losses[loss_type] is not None:
                epochs = range(len(losses[loss_type]))
                ax.plot(epochs, losses[loss_type],
                        color=colors[j], label=exp_name, linewidth=2)

        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Curves Comparison', fontsize=16)
    plt.tight_layout()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/loss_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
