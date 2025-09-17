"""Embedding modules for character and BERT representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from transformers import AutoModel


class Embeddings(nn.Module):
    """Combined character and BERT embeddings."""
    
    def __init__(self, num_chars: int, config, bert_model: AutoModel, num_layers: int = 1):
        """
        Initialize embedding layers.
        
        Args:
            num_chars: Size of character vocabulary
            config: Configuration object
            bert_model: Pre-trained BERT model
            num_layers: Number of LSTM layers for character embeddings
        """
        super().__init__()
        self.config = config
        self.E = self.config.char_dim
        self.R = self.config.word_dim
        self.H = self.config.emb_hidden
        self.W = self.config.max_seq_len
        self.C = self.config.max_word_len
        self.bert_model = bert_model

        # Character embedding components
        self.cemb = nn.Embedding(num_chars, self.E)
        self.wemb = nn.LSTM(self.E, self.R, num_layers, batch_first=True)
        self.linear = nn.Linear(self.R, self.H)

    def _char_embeddings(self, char_ids: torch.Tensor, char_word_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Generate character-based word embeddings.
        
        Args:
            char_ids: Character indices, shape: [B, W*C]
            char_word_ids: Word alignment for characters
            device: Device to run computations on
            
        Returns:
            Word representations, shape: [B, W, H]
        """
        char_embeds = self.cemb(char_ids)  # shape: [B, W*C, E]

        pad_mask = einops.rearrange(char_ids == -100, 'B WC -> B WC 1')
        char_embeds = char_embeds.masked_fill(pad_mask, 0.0)

        char_embeds = einops.rearrange(
            char_embeds, 'B (W C) E -> (B W) C E',
            W=self.W, C=self.C
        )  # shape: [B*W, C, E]

        _, (hidden, _) = self.wemb(char_embeds)
        word_repr = einops.rearrange(
            hidden[-1], '(B W) R -> B W R', W=self.W
        )  # shape: [B, W, R]

        return self.linear(word_repr)  # shape: [B, W, H]

    def _bert_embeddings(self, 
                        token_ids: torch.Tensor, 
                        token_word_ids: torch.Tensor, 
                        attention_mask: torch.Tensor, 
                        special_tokens_mask: torch.Tensor,
                        device: torch.device) -> torch.Tensor:
        """
        Generate BERT-based word embeddings.
        
        Args:
            token_ids: BERT token indices
            token_word_ids: Token to word alignment
            attention_mask: Attention mask for tokens
            special_tokens_mask: Mask for special tokens
            device: Device to run computations on
            
        Returns:
            Word representations, shape: [B, W, H]
        """
        token2word_mapping = F.one_hot(
            token_word_ids, num_classes=self.W
        ).to(dtype=torch.float32, device=device)  # shape: [B, T, W]

        embeddings = self.bert_model(
            input_ids=token_ids, 
            attention_mask=attention_mask
        )
        last_hidden_state = embeddings.last_hidden_state.to(device)  # shape: [B, T, H]

        content_mask = attention_mask & ~special_tokens_mask
        content_mask = content_mask.to(device)

        # Filter to content-only embeddings
        content_embeddings = []
        content_lengths = []

        for batch_idx in range(last_hidden_state.size(0)):
            seq_content_mask = content_mask[batch_idx].bool()
            seq_content_embeddings = last_hidden_state[batch_idx][seq_content_mask]
            
            content_embeddings.append(seq_content_embeddings)
            content_lengths.append(seq_content_embeddings.size(0))

        # Pad content embeddings for batching
        padded_content_embeddings = torch.zeros(
            len(content_embeddings), 
            self.config.max_tok_len, 
            last_hidden_state.size(-1)
        ).to(device)
        
        for i, (emb, length) in enumerate(zip(content_embeddings, content_lengths)):
            padded_content_embeddings[i, :length] = emb

        return torch.einsum('BTH,BTW->BWH', padded_content_embeddings, token2word_mapping)

    def forward(self, 
               char_ids: torch.Tensor, 
               char_word_ids: torch.Tensor, 
               token_ids: torch.Tensor, 
               token_word_ids: torch.Tensor, 
               attention_mask: torch.Tensor, 
               special_tokens_mask: torch.Tensor,
               device: torch.device) -> tuple:
        """
        Forward pass combining BERT and character embeddings.
        
        Returns:
            Tuple of (word_embeddings, word_level_mask)
        """
        bert_embeds = self._bert_embeddings(
            token_ids, token_word_ids, attention_mask, special_tokens_mask, device
        )
        
        if self.config.use_char_architecture and char_ids is not None:
            char_embeds = self._char_embeddings(char_ids, char_word_ids, device)
        else:
            char_embeds = torch.zeros_like(bert_embeds)

        word_embeds = bert_embeds + char_embeds  # shape: [B, W, H]

        mask_padded_words = torch.where(word_embeds == 0, 0, 1)
        word_level_mask = torch.sum(mask_padded_words, dim=-1) > 0  # shape: [B, W]

        return word_embeds, word_level_mask
