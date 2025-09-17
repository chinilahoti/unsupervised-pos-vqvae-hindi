"""Encoder modules for the tagging model."""

import torch
import torch.nn as nn


class EncoderFFN(nn.Module):
    """Simple feedforward encoder for generating tag logits."""
    
    def __init__(self, emb_hidden: int, num_tag: int):
        """
        Initialize encoder.
        
        Args:
            emb_hidden: Hidden dimension of input embeddings
            num_tag: Number of output tags
        """
        super().__init__()
        self.ffn = nn.Linear(emb_hidden, num_tag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings, shape: [B, W, emb_hidden]
            
        Returns:
            Tag logits, shape: [B, W, num_tag]
        """
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply transformer
        transformer_out = self.transformer(x)  # shape: [B, W, emb_hidden]
        return self.output_projection(transformer_out)  # shape: [B, W, num_tag].Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings, shape: [B, W, emb_hidden]
            
        Returns:
            Tag logits, shape: [B, W, num_tag]
        """
        return self.ffn(x)


class EncoderBiLSTM(nn.Module):
    """Bidirectional LSTM encoder for generating tag logits."""
    
    def __init__(self, emb_hidden: int, hidden_size: int, num_tag: int, num_layers: int = 1, dropout: float = 0.1):
        """
        Initialize BiLSTM encoder.
        
        Args:
            emb_hidden: Hidden dimension of input embeddings
            hidden_size: LSTM hidden size
            num_tag: Number of output tags
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        self.lstm = nn.LSTM(
            emb_hidden, 
            hidden_size, 
            num_layers, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_size * 2, num_tag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings, shape: [B, W, emb_hidden]
            
        Returns:
            Tag logits, shape: [B, W, num_tag]
        """
        lstm_out, _ = self.lstm(x)  # shape: [B, W, hidden_size*2]
        lstm_out = self.dropout(lstm_out)
        return self.output_projection(lstm_out)  # shape: [B, W, num_tag]


class EncoderTransformer(nn.Module):
    """Transformer encoder for generating tag logits."""
    
    def __init__(self, emb_hidden: int, num_tag: int, num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize Transformer encoder.
        
        Args:
            emb_hidden: Hidden dimension of input embeddings
            num_tag: Number of output tags
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, emb_hidden))  # Max seq length 512
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_hidden,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(emb_hidden, num_tag)

    def forward(self, x: torch