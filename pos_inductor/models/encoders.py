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
        return self.ffn(x)
