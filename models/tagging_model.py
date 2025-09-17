"""Main tagging model combining all components."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Tuple, Optional, Dict


class TaggingModel(nn.Module):
    """Main POS tagging model with discrete representation learning."""
    
    def __init__(self, 
                 config,
                 embeddings,
                 encoder,
                 codebook,
                 vocab_decoder,
                 char_decoder,
                 vocab_size: int):
        """
        Initialize tagging model.
        
        Args:
            config: Configuration object
            embeddings: Embedding module
            encoder: Encoder module
            codebook: Codebook module
            vocab_decoder: Vocabulary decoder module
            char_decoder: Character decoder module (optional)
            vocab_size: Size of vocabulary
        """
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.embeddings = embeddings
        self.encoder = encoder
        self.codebook = codebook
        self.vocab_decoder = vocab_decoder
        self.char_decoder = char_decoder
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def _calculate_diversity_loss(self, label_probs: torch.Tensor, word_mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Calculate diversity loss to prevent codebook collapse.
        
        Args:
            label_probs: Label probabilities, shape: [B, W, L]
            word_mask: Mask for valid words, shape: [B, W]
            device: Device for computations
            
        Returns:
            Diversity loss scalar
        """
        # Calculate average label probabilities across all valid positions
        avg_label_probs = torch.logsumexp(
            torch.where(word_mask[:, :, None], label_probs, -torch.tensor(float('inf')).to(device)),
            dim=(0, 1)
        ) - torch.log(torch.sum(word_mask))  # shape: [L]

        # Calculate entropy
        actual_entropy = F.cross_entropy(
            avg_label_probs.unsqueeze(0),  # Add batch dim: [1, num_labels]
            torch.exp(avg_label_probs).unsqueeze(0),  # Convert to probs and add batch dim
            reduction='sum'
        )

        # Maximum possible entropy
        max_entropy = torch.log(torch.tensor(label_probs.shape[-1], dtype=torch.float32).to(device))
        diversity_loss = max_entropy - actual_entropy

        return diversity_loss

    def _create_char_targets(self, 
                           char_ids: torch.Tensor, 
                           char_word_ids: torch.Tensor, 
                           char2index: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert flat character sequences to word-aligned character targets.
        
        Args:
            char_ids: Character indices, shape: [B, W*C]
            char_word_ids: Word alignment for characters, shape: [B, W*C]
            char2index: Character to index mapping
            
        Returns:
            Tuple of (target_chars, char_mask)
        """
        B = char_ids.size(0)
        W = self.config.max_seq_len
        C = self.config.max_word_len

        target_chars = torch.full(
            (B, W, C), char2index.get("PAD", 0),
            dtype=torch.long, device=char_ids.device
        )
        char_mask = torch.zeros((B, W, C), dtype=torch.bool, device=char_ids.device)

        # Fill in actual characters for each word
        for b in range(B):
            char_pos_in_word = {}  # Track position within each word

            for c_idx, (char_id, word_id) in enumerate(zip(char_ids[b], char_word_ids[b])):
                word_id_val = word_id.item()
                char_id_val = char_id.item()

                # Skip padding tokens and invalid word indices
                if (word_id_val >= W or char_id_val == char2index.get("PAD", -100)):
                    continue

                # Initialize position counter for this word
                if word_id_val not in char_pos_in_word:
                    char_pos_in_word[word_id_val] = 0

                char_pos = char_pos_in_word[word_id_val]

                # Add character if within bounds
                if char_pos < C:
                    target_chars[b, word_id_val, char_pos] = char_id_val
                    char_mask[b, word_id_val, char_pos] = True
                    char_pos_in_word[word_id_val] += 1

        return target_chars, char_mask

    def forward(self, 
               char_ids: Optional[torch.Tensor],
               char_word_ids: Optional[torch.Tensor],
               token_ids: torch.Tensor,
               token_word_ids: torch.Tensor,
               attention_mask: torch.Tensor,
               special_tokens_mask: torch.Tensor,
               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass of the tagging model.
        
        Returns:
            Tuple of (label_logprobs, word_logprobs, char_seq_logprobs, word_level_mask, weights)
        """
        # Get word embeddings
        word_embeddings, word_level_mask = self.embeddings(
            char_ids, char_word_ids, token_ids, token_word_ids, 
            attention_mask, special_tokens_mask, device
        )
        
        # Encode to tag logits
        enc_logits = self.encoder(word_embeddings)  # shape: [B, W, num_tag]
        
        # Get discrete representations via codebook
        quantized, weights = self.codebook(enc_logits, self.config.gumbel_temperature)
        
        # Decode to vocabulary
        word_logits = self.vocab_decoder(quantized)  # shape: [B, W, vocab_size]
        word_logprobs = F.log_softmax(word_logits, dim=-1)
        
        # Character decoding (optional)
        char_seq_logprobs = None
        if self.char_decoder is not None:
            char_seq_logits = self.char_decoder(quantized, device=device)
            char_seq_logprobs = F.log_softmax(char_seq_logits, dim=-1)

        label_logprobs = F.log_softmax(enc_logits, dim=-1)  # shape: [B, W, num_tag]

        return label_logprobs, word_logprobs, char_seq_logprobs, word_level_mask, weights

    def compute_loss(self,
                    token_ids: torch.Tensor,
                    token_word_ids: torch.Tensor,
                    attention_mask: torch.Tensor,
                    special_tokens_mask: torch.Tensor,
                    vocab_ids: torch.Tensor,
                    char_ids: Optional[torch.Tensor] = None,
                    char_word_ids: Optional[torch.Tensor] = None,
                    char2index: Optional[Dict[str, int]] = None,
                    device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute unsupervised loss for training.
        
        Returns:
            Tuple of (total_loss, vocab_reconstr_loss, char_reconstr_loss, diversity_loss, predicted_tags)
        """
        if device is None:
            device = vocab_ids.device

        # Forward pass
        label_probs, word_probs, char_probs, word_mask, pred_tags = self.forward(
            char_ids, char_word_ids, token_ids, token_word_ids, 
            attention_mask, special_tokens_mask, device
        )

        # 1. Diversity loss
        div_loss = self._calculate_diversity_loss(label_probs, word_mask, device)

        # 2. Vocabulary reconstruction loss
        mask_vocab_ids = torch.where(word_mask, vocab_ids, -100)
        vocab_flat = einops.rearrange(mask_vocab_ids, 'B W -> (B W)')
        word_probs_flat = einops.rearrange(word_probs, 'B W V -> (B W) V')
        vocab_reconstr = self.loss_fn(word_probs_flat, vocab_flat)

        # 3. Character reconstruction loss (optional)
        char_reconstr = torch.tensor(0.0, device=device)
        if self.char_decoder is not None and char_ids is not None and char2index is not None:
            target_chars, char_mask = self._create_char_targets(char_ids, char_word_ids, char2index)
            
            # Flatten for loss calculation
            char_probs_flat = einops.rearrange(char_probs, 'B W C A -> (B W C) A')
            target_chars_flat = einops.rearrange(target_chars, 'B W C -> (B W C)')
            char_mask_flat = einops.rearrange(char_mask, 'B W C -> (B W C)')
            
            # Apply mask: set padded positions to ignore_index
            masked_char_targets = torch.where(char_mask_flat, target_chars_flat, -100)
            char_reconstr = self.loss_fn(char_probs_flat, masked_char_targets)

        # 4. Combined total loss
        total_loss = (
            (self.config.vocab_loss_weight * vocab_reconstr) +
            (self.config.diversity_weight * div_loss) +
            (self.config.char_loss_weight * char_reconstr)
        )

        return total_loss, vocab_reconstr, char_reconstr, div_loss, pred_tags

    def predict_tags(self,
                    token_ids: torch.Tensor,
                    token_word_ids: torch.Tensor,
                    attention_mask: torch.Tensor,
                    special_tokens_mask: torch.Tensor,
                    char_ids: Optional[torch.Tensor] = None,
                    char_word_ids: Optional[torch.Tensor] = None,
                    device: torch.device = None) -> torch.Tensor:
        """
        Predict tags for input sequences.
        
        Returns:
            Predicted tag indices, shape: [B, W]
        """
        if device is None:
            device = token_ids.device
            
        with torch.no_grad():
            label_logprobs, _, _, _, weights = self.forward(
                char_ids, char_word_ids, token_ids, token_word_ids,
                attention_mask, special_tokens_mask, device
            )
            return torch.argmax(weights, dim=-1)
