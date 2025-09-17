"""Decoder modules for vocabulary and character reconstruction."""

import torch
import torch.nn as nn
import einops
from typing import Optional


class DecoderFFN(nn.Module):
    """Simple feedforward decoder for vocabulary reconstruction."""

    def __init__(self, tag_dim: int, vocab_size: int):
        """
        Initialize FFN decoder.
        
        Args:
            tag_dim: Dimension of input tag representations
            vocab_size: Size of vocabulary
        """
        super().__init__()
        self.ffn = nn.Linear(tag_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tag representations, shape: [B, W, tag_dim]
            
        Returns:
            Vocabulary logits, shape: [B, W, vocab_size]
        """
        return self.ffn(x)


class DecoderBiLSTM(nn.Module):
    """Bidirectional LSTM decoder for vocabulary reconstruction."""

    def __init__(self, tag_dim: int, dec_hidden: int, vocab_size: int, num_layers: int, dropout: float = 0.1):
        """
        Initialize BiLSTM decoder.
        
        Args:
            tag_dim: Dimension of input tag representations
            dec_hidden: LSTM hidden size
            vocab_size: Size of vocabulary
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        self.lstm = nn.LSTM(
            tag_dim,
            dec_hidden,
            num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(dec_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tag representations, shape: [B, W, tag_dim]
            
        Returns:
            Vocabulary logits, shape: [B, W, vocab_size]
        """
        lstm_out, _ = self.lstm(x)  # shape: [B, W, dec_hidden*2]
        lstm_out = self.dropout(lstm_out)
        return self.output_projection(lstm_out)  # shape: [B, W, vocab_size]


class CharDecoder(nn.Module):
    """Character-level decoder for fine-grained reconstruction."""

    def __init__(self, tag_dim: int, hidden_size: int, char_vocab_size: int, max_char_len: int):
        """
        Initialize character decoder.
        
        Args:
            tag_dim: Dimension of input tag representations
            hidden_size: LSTM hidden size
            char_vocab_size: Size of character vocabulary
            max_char_len: Maximum number of characters per word
        """
        super().__init__()
        self.input_projection = nn.Linear(tag_dim, hidden_size)
        self.char_lstm = nn.LSTM(char_vocab_size, hidden_size, batch_first=True)
        self.output_projection = nn.Linear(hidden_size, char_vocab_size)
        self.max_char_len = max_char_len
        self.char_vocab_size = char_vocab_size
        self.hidden_size = hidden_size

    def forward(self,
                quantized_repr: torch.Tensor,
                target_chars: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5,
                device: torch.device = None) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing.
        
        Args:
            quantized_repr: Tag representations, shape: [B, W, tag_dim]
            target_chars: Target character sequences for teacher forcing, shape: [B, W, C]
            teacher_forcing_ratio: Probability of using teacher forcing
            device: Device for tensor operations
            
        Returns:
            Character logits, shape: [B, W, C, char_vocab_size]
        """
        if device is None:
            device = quantized_repr.device

        B, W, Q = quantized_repr.shape
        C = self.max_char_len
        A = self.char_vocab_size

        # Project tag representations to LSTM hidden size
        initial_hidden = self.input_projection(quantized_repr)  # [B, W, hidden_size]
        initial_hidden = einops.rearrange(initial_hidden, 'B W D -> 1 (B W) D')
        initial_cell = torch.zeros_like(initial_hidden)

        # Initialize with START token (assume index 0 is START)
        decoder_input = torch.zeros(B * W, 1, A, device=device)
        decoder_input[:, 0, 0] = 1.0  # START token

        outputs = []
        hidden_state = (initial_hidden, initial_cell)

        # Autoregressive character generation
        for t in range(C):
            # LSTM forward pass
            lstm_out, hidden_state = self.char_lstm(decoder_input, hidden_state)

            # Project to character probabilities
            char_logits = self.output_projection(lstm_out)  # [B*W, 1, char_vocab_size]
            outputs.append(char_logits)

            # Prepare next input (teacher forcing vs autoregressive)
            if target_chars is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                if t < C - 1:
                    target_chars_flat = einops.rearrange(target_chars, 'B W C -> (B W) C')
                    next_char = target_chars_flat[:, t]  # [B*W]

                    # Create one-hot encoding
                    decoder_input = torch.zeros(B * W, 1, A, device=device)
                    # Handle potential invalid indices
                    valid_mask = (next_char >= 0) & (next_char < A)
                    valid_chars = torch.where(valid_mask, next_char, torch.tensor(0, device=device))
                    char_indices = einops.rearrange(valid_chars, 'BW -> BW 1 1')
                    decoder_input.scatter_(2, char_indices, 1.0)
            else:
                # Use model prediction
                predicted_char = torch.argmax(char_logits, dim=-1)  # [B*W, 1]
                decoder_input = torch.zeros(B * W, 1, A, device=device)
                pred_indices = einops.rearrange(predicted_char, 'BW 1 -> BW 1 1')
                decoder_input.scatter_(2, pred_indices, 1.0)

        # Concatenate all outputs and reshape
        char_logits = torch.cat(outputs, dim=1)  # [B*W, C, A]
        char_logits = einops.rearrange(char_logits, '(B W) C A -> B W C A', B=B, W=W)

        return char_logits


class AttentionDecoder(nn.Module):
    """Attention-based decoder for vocabulary reconstruction."""

    def __init__(self, tag_dim: int, vocab_size: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize attention decoder.
        
        Args:
            tag_dim: Dimension of input tag representations
            vocab_size: Size of vocabulary
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            tag_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(tag_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(tag_dim, tag_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tag_dim * 4, tag_dim)
        )
        self.norm2 = nn.LayerNorm(tag_dim)
        self.output_projection = nn.Linear(tag_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tag representations, shape: [B, W, tag_dim]
            
        Returns:
            Vocabulary logits, shape: [B, W, vocab_size]
        """
        # Self-attention
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feedforward
        ff_out = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_out))

        # Output projection
        return self.output_projection(x)
