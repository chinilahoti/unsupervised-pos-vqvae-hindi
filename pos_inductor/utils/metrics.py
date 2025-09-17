"""Metrics and evaluation utilities."""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import Counter


def calculate_m1_accuracy(m1_totals: Dict[str, Dict[str, int]]) -> float:
    """
    Calculate Many-to-One (M1) accuracy.
    
    Args:
        m1_totals: Dictionary mapping M1 labels to gold label counts
        
    Returns:
        M1 accuracy percentage
    """
    correct = 0
    total = 0

    for m1_label, count_dict in m1_totals.items():
        # Correct predictions are when predicted M1 label matches gold label
        if m1_label in count_dict:
            correct += count_dict[m1_label]
        total += sum(count_dict.values())

    return (correct / total) * 100 if total > 0 else 0.0


def calculate_cluster_purity(gold_labels: List[int], pred_labels: List[int]) -> float:
    """
    Calculate cluster purity score.
    
    Args:
        gold_labels: Ground truth labels
        pred_labels: Predicted cluster labels
        
    Returns:
        Purity score between 0 and 1
    """
    # Create confusion matrix
    clusters = set(pred_labels)
    total_correct = 0
    total_samples = len(gold_labels)

    for cluster in clusters:
        # Get indices of samples in this cluster
        cluster_indices = [i for i, pred in enumerate(pred_labels) if pred == cluster]

        if not cluster_indices:
            continue

        # Get gold labels for this cluster
        cluster_gold_labels = [gold_labels[i] for i in cluster_indices]

        # Find most common gold label in this cluster
        most_common_label = Counter(cluster_gold_labels).most_common(1)[0][1]
        total_correct += most_common_label

    return total_correct / total_samples if total_samples > 0 else 0.0


def calculate_cluster_metrics(gold_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
    """
    Calculate various clustering metrics.
    
    Args:
        gold_labels: Ground truth labels
        pred_labels: Predicted cluster labels
        
    Returns:
        Dictionary containing various metrics
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure

    # Basic clustering metrics
    ari = adjusted_rand_score(gold_labels, pred_labels)
    nmi = normalized_mutual_info_score(gold_labels, pred_labels)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(gold_labels, pred_labels)
    purity = calculate_cluster_purity(gold_labels, pred_labels)

    return {
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'purity': purity
    }


def calculate_tag_level_metrics(gold_labels: List[int],
                                pred_labels: List[int],
                                label_names: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate precision, recall, and F1 for each tag.
    
    Args:
        gold_labels: Ground truth labels
        pred_labels: Predicted labels
        label_names: Names of labels (optional)
        
    Returns:
        Dictionary mapping label names to metrics
    """
    # Remove padding tokens (-100)
    valid_indices = [i for i, gold in enumerate(gold_labels) if gold != -100]
    filtered_gold = [gold_labels[i] for i in valid_indices]
    filtered_pred = [pred_labels[i] for i in valid_indices]

    if not filtered_gold:
        return {}

    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        filtered_gold, filtered_pred, average=None, zero_division=0
    )

    # Get unique labels
    unique_labels = sorted(set(filtered_gold + filtered_pred))

    # Create results dictionary
    results = {}
    for i, label in enumerate(unique_labels):
        label_name = label_names[label] if label_names and label < len(label_names) else f"Label_{label}"

        if i < len(precision):
            results[label_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i] if i < len(support) else 0
            }

    # Add macro and micro averages
    macro_precision = np.mean(precision) if len(precision) > 0 else 0
    macro_recall = np.mean(recall) if len(recall) > 0 else 0
    macro_f1 = np.mean(f1) if len(f1) > 0 else 0

    results['macro_avg'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1_score': macro_f1,
        'support': sum(support) if len(support) > 0 else 0
    }

    # Micro averages (same as accuracy for multi-class)
    micro_precision = micro_recall = micro_f1 = accuracy_score(filtered_gold, filtered_pred)

    results['micro_avg'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1_score': micro_f1,
        'support': len(filtered_gold)
    }

    return results


def calculate_perplexity(log_probs: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Calculate perplexity from log probabilities.
    
    Args:
        log_probs: Log probabilities, shape [batch, seq_len, vocab_size]
        mask: Mask for valid positions, shape [batch, seq_len]
        
    Returns:
        Perplexity score
    """
    if mask is not None:
        # Only calculate perplexity for valid positions
        valid_log_probs = log_probs[mask]
    else:
        valid_log_probs = log_probs.flatten()

    if len(valid_log_probs) == 0:
        return float('inf')

    # Calculate cross-entropy
    cross_entropy = -np.mean(valid_log_probs)

    # Calculate perplexity
    perplexity = np.exp(cross_entropy)

    return perplexity


def create_confusion_matrix_dict(gold_labels: List[int],
                                 pred_labels: List[int],
                                 num_classes: int) -> Dict[int, Dict[int, int]]:
    """
    Create confusion matrix as nested dictionary.
    
    Args:
        gold_labels: Ground truth labels
        pred_labels: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Dictionary mapping pred_class -> gold_class -> count
    """
    # Initialize confusion matrix dictionary
    conf_dict = {pred: {gold: 0 for gold in range(1, num_classes + 1)}
                 for pred in range(num_classes)}

    # Fill confusion matrix
    for gold, pred in zip(gold_labels, pred_labels):
        if gold != -100:  # Skip padding tokens
            conf_dict[pred][gold] = conf_dict[pred].get(gold, 0) + 1

    return conf_dict


def calculate_entropy(probabilities: np.ndarray) -> float:
    """
    Calculate entropy of probability distribution.
    
    Args:
        probabilities: Probability distribution
        
    Returns:
        Entropy value
    """
    # Add small epsilon to avoid log(0)
    probabilities = probabilities + 1e-8

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def calculate_codebook_statistics(usage_counts: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics about codebook usage.
    
    Args:
        usage_counts: Usage counts for each code
        
    Returns:
        Dictionary with statistics
    """
    total_usage = np.sum(usage_counts)

    if total_usage == 0:
        return {
            'active_codes': 0,
            'usage_entropy': 0.0,
            'gini_coefficient': 0.0,
            'max_usage': 0.0,
            'min_usage': 0.0,
            'mean_usage': 0.0,
            'std_usage': 0.0
        }

    # Normalize to probabilities
    usage_probs = usage_counts / total_usage

    # Calculate statistics
    active_codes = np.sum(usage_counts > 0)
    usage_entropy = calculate_entropy(usage_probs)

    # Calculate Gini coefficient (measure of inequality)
    sorted_usage = np.sort(usage_probs)
    n = len(sorted_usage)
    index = np.arange(1, n + 1)
    gini_coefficient = (2 * np.sum(index * sorted_usage)) / (n * np.sum(sorted_usage)) - (n + 1) / n

    return {
        'active_codes': int(active_codes),
        'usage_entropy': float(usage_entropy),
        'gini_coefficient': float(gini_coefficient),
        'max_usage': float(np.max(usage_probs)),
        'min_usage': float(np.min(usage_probs)),
        'mean_usage': float(np.mean(usage_probs)),
        'std_usage': float(np.std(usage_probs))
    }


def compare_experiment_results(results: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
    """
    Compare results across multiple experiments.
    
    Args:
        results: Dictionary mapping experiment_name -> metrics_dict
        
    Returns:
        Comparison dictionary with relative improvements
    """
    if not results:
        return {}

    # Get baseline (first experiment)
    baseline_name = list(results.keys())[0]
    baseline_results = results[baseline_name]

    comparison = {}

    for exp_name, exp_results in results.items():
        if exp_name == baseline_name:
            comparison[exp_name] = {metric: 0.0 for metric in exp_results.keys() if
                                    isinstance(exp_results[metric], (int, float))}
            continue

        exp_comparison = {}
        for metric, value in exp_results.items():
            if isinstance(value, (int, float)) and metric in baseline_results:
                baseline_value = baseline_results[metric]
                if baseline_value != 0:
                    improvement = ((value - baseline_value) / baseline_value) * 100
                    exp_comparison[metric] = improvement
                else:
                    exp_comparison[metric] = 0.0

        comparison[exp_name] = exp_comparison

    return comparison
