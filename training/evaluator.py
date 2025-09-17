"""Evaluation utilities for the POS tagging model."""

import torch
import numpy as np
import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class Evaluator:
    """Evaluator class for the POS tagging model."""
    
    def __init__(self, model, config, device: torch.device, char2index: Optional[Dict[str, int]] = None):
        """
        Initialize evaluator.
        
        Args:
            model: The tagging model to evaluate
            config: Configuration object  
            device: Device for evaluation
            char2index: Character to index mapping
        """
        self.model = model
        self.config = config
        self.device = device
        self.char2index = char2index

    def evaluate(self, 
                dataloader: DataLoader,
                label2index: Optional[Dict[str, int]] = None,
                index2label: Optional[Dict[int, str]] = None,
                mode: str = 'test') -> Dict:
        """
        Evaluate model performance.
        
        Args:
            dataloader: Data loader for evaluation
            label2index: Label to index mapping (for supervised evaluation)
            index2label: Index to label mapping (for supervised evaluation)
            mode: Evaluation mode ('test', 'validation', etc.)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        # Storage for predictions and analysis
        word_tag_counts_dict = {i: {} for i in range(self.config.num_tag)}
        count_dict = None
        if label2index is not None:
            count_dict = {i: {j+1: 0 for j in range(len(label2index))} for i in range(self.config.num_tag)}

        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc=f"Evaluating ({mode})"):
                # Move batch to device
                token_ids = batch['token_ids'].to(self.device)
                token_word_ids = batch['token_word_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                special_tokens_mask = batch['spec_tok_mask'].to(self.device)
                vocab_ids = batch['vocab_ids'].to(self.device)
                
                char_ids = None
                char_word_ids = None
                if self.config.use_char_architecture:
                    char_ids = batch['char_ids'].to(self.device) if batch['char_ids'] is not None else None
                    char_word_ids = batch['char_word_ids'].to(self.device) if batch['char_word_ids'] is not None else None

                # Compute loss and predictions
                loss, vocab_reconstr, char_reconstr, div_loss, tag_logits = self.model.compute_loss(
                    token_ids, token_word_ids, attention_mask, special_tokens_mask,
                    vocab_ids, char_ids, char_word_ids, self.char2index, self.device
                )
                
                predicted = torch.argmax(tag_logits, -1)
                total_loss += loss.item()
                num_batches += 1

                # Analyze predictions for test mode
                if mode == 'test':
                    self._collect_word_tag_counts(vocab_ids, predicted, word_tag_counts_dict)
                    
                    # Supervised evaluation if labels available
                    if label2index is not None and 'labels' in batch:
                        labels = batch['labels'].to(self.device)
                        self._collect_label_tag_counts(labels, predicted, count_dict)

        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        # Prepare results
        results = {
            'avg_loss': avg_loss,
            'word_tag_counts': word_tag_counts_dict if mode == 'test' else None
        }

        # Add supervised evaluation results
        if mode == 'test' and label2index is not None and count_dict is not None:
            m1_dict, pred_to_m1 = self._many_to_one_mapping(count_dict, label2index, index2label)
            m1_accuracy = self._calculate_m1_accuracy(m1_dict)
            
            results.update({
                'count_dict': count_dict,
                'm1_dict': m1_dict,
                'pred_to_m1': pred_to_m1,
                'm1_accuracy': m1_accuracy
            })

        return results

    def _collect_word_tag_counts(self, vocab_ids: torch.Tensor, predicted: torch.Tensor, word_tag_counts_dict: Dict):
        """Collect word-tag co-occurrence counts."""
        vocab_ids_np = vocab_ids.cpu().detach().numpy()
        predicted_np = predicted.cpu().detach().numpy()
        vocab_predicted_joined = np.rec.fromarrays([vocab_ids_np, predicted_np])

        for seq in vocab_predicted_joined:
            for gold_word_tag in seq:
                gold_word = int(gold_word_tag[0])
                pred_tag = int(gold_word_tag[1])

                if gold_word != -100:
                    word_tag_counts_dict[pred_tag][gold_word] = word_tag_counts_dict[pred_tag].get(gold_word, 0) + 1

    def _collect_label_tag_counts(self, labels: torch.Tensor, predicted: torch.Tensor, count_dict: Dict):
        """Collect gold label to predicted tag counts."""
        gold_np = labels.cpu().detach().numpy()
        predicted_np = predicted.cpu().detach().numpy()
        gold_predicted_joined = np.rec.fromarrays([gold_np, predicted_np])

        for seq in gold_predicted_joined:
            for tup_tag in seq:
                gold_tag = int(tup_tag[0])
                pred_tag = int(tup_tag[1])

                if gold_tag != -100:
                    count_dict[pred_tag][gold_tag] = count_dict[pred_tag].get(gold_tag, 0) + 1

    def _many_to_one_mapping(self, 
                           count_dict: Dict, 
                           label2index: Dict[str, int], 
                           index2label: Dict[int, str]) -> Tuple[Dict, Dict]:
        """
        Create many-to-one mapping from predicted tags to gold labels.
        
        Returns:
            Tuple of (m1_dict, pred_to_m1_mapping)
        """
        m1_dict = {i+1: [] for i in range(len(label2index))}
        pred_to_m1 = {i: 0 for i in range(self.config.num_tag)}

        for pred, gold_counts in count_dict.items():
            # Get the gold tag most associated with each predicted tag
            top_gold_tag = max(gold_counts, key=gold_counts.get)
            m1_dict[top_gold_tag].append(gold_counts)
            pred_to_m1[pred] = top_gold_tag

        # Aggregate counts for M1 mapping
        m1_totals = {}
        for m1_tag, dict_list in m1_dict.items():
            totals = {}
            for d in dict_list:
                for k, v in d.items():
                    label_name = index2label[k]
                    if label_name not in totals:
                        totals[label_name] = v
                    else:
                        totals[label_name] += v
            
            m1_totals[index2label[m1_tag]] = totals

        return m1_totals, pred_to_m1

    def _calculate_m1_accuracy(self, m1_totals: Dict) -> float:
        """Calculate many-to-one accuracy."""
        correct = 0
        total = 0
        
        for k, count_dict in m1_totals.items():
            if k in count_dict:
                correct += count_dict[k]
            total += sum(count_dict.values())

        accuracy = (correct / total) * 100 if total > 0 else 0
        return accuracy

    def predict_sequences(self, 
                         dataloader: DataLoader,
                         index2label: Optional[Dict[int, str]] = None) -> List[List[str]]:
        """
        Predict tag sequences for input data.
        
        Args:
            dataloader: Data loader for prediction
            index2label: Optional mapping from indices to labels
            
        Returns:
            List of predicted tag sequences
        """
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="Predicting"):
                token_ids = batch['token_ids'].to(self.device)
                token_word_ids = batch['token_word_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                special_tokens_mask = batch['spec_tok_mask'].to(self.device)
                
                char_ids = None
                char_word_ids = None
                if self.config.use_char_architecture:
                    char_ids = batch['char_ids'].to(self.device) if batch['char_ids'] is not None else None
                    char_word_ids = batch['char_word_ids'].to(self.device) if batch['char_word_ids'] is not None else None

                predictions = self.model.predict_tags(
                    token_ids, token_word_ids, attention_mask, special_tokens_mask,
                    char_ids, char_word_ids, self.device
                )

                # Convert to numpy and process
                predictions_np = predictions.cpu().numpy()
                
                for seq_preds in predictions_np:
                    if index2label is not None:
                        # Convert to label names (for M1 mapping)
                        seq_labels = [index2label.get(pred, f'TAG_{pred}') for pred in seq_preds]
                    else:
                        # Keep as tag indices
                        seq_labels = [f'TAG_{pred}' for pred in seq_preds]
                    
                    all_predictions.append(seq_labels)

        return all_predictions

    def analyze_codebook_usage(self, dataloader: DataLoader) -> Dict:
        """
        Analyze how the codebook is being used.
        
        Returns:
            Dictionary with usage statistics
        """
        self.model.eval()
        all_weights = []

        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc="Analyzing codebook usage"):
                token_ids = batch['token_ids'].to(self.device)
                token_word_ids = batch['token_word_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                special_tokens_mask = batch['spec_tok_mask'].to(self.device)
                
                char_ids = None
                char_word_ids = None
                if self.config.use_char_architecture:
                    char_ids = batch['char_ids'].to(self.device) if batch['char_ids'] is not None else None
                    char_word_ids = batch['char_word_ids'].to(self.device) if batch['char_word_ids'] is not None else None

                _, _, _, _, weights = self.model.forward(
                    char_ids, char_word_ids, token_ids, token_word_ids,
                    attention_mask, special_tokens_mask, self.device
                )
                
                all_weights.append(weights.cpu())

        # Combine all weights
        all_weights = torch.cat(all_weights, dim=0)  # [total_samples, W, num_tag]
        
        # Calculate usage statistics
        usage_per_tag = all_weights.mean(dim=(0, 1))  # [num_tag]
        entropy = -(all_weights * torch.log(all_weights + 1e-8)).sum(dim=-1).mean()
        
        # Find most/least used tags
        sorted_usage = torch.argsort(usage_per_tag, descending=True)
        
        return {
            'usage_per_tag': usage_per_tag.numpy(),
            'entropy': entropy.item(),
            'most_used_tags': sorted_usage[:10].numpy(),
            'least_used_tags': sorted_usage[-10:].numpy(),
            'num_active_tags': (usage_per_tag > 0.001).sum().item()  # Tags used more than 0.1%
        }
