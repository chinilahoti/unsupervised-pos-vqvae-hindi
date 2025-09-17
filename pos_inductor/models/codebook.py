"""Gumbel softmax codebook for discrete representation learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelCodebook(nn.Module):
    """Gumbel softmax codebook for learning discrete representations."""
    
    def __init__(self, num_tag: int, tag_dim: int):
        """
        Initialize codebook.
        
        Args:
            num_tag: Number of discrete codes in the codebook
            tag_dim: Dimension of each code vector
        """
        super().__init__()
        self.num_tag = num_tag
        self.tag_dim = tag_dim
        self.codebook = nn.Parameter(torch.randn(num_tag, tag_dim))
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.codebook)

    def forward(self, logits: torch.Tensor, temperature: float = 1.0) -> tuple:
        """
        Forward pass using Gumbel softmax.
        
        Args:
            logits: Input logits, shape: [B, W, num_tag]
            temperature: Temperature for Gumbel softmax
            
        Returns:
            Tuple of (quantized_representations, weights)
            quantized: shape [B, W, tag_dim]
            weights: shape [B, W, num_tag]
        """
        # Apply Gumbel softmax to get soft discrete distribution
        weights = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
        
        # Get quantized representations by weighted sum of codebook vectors
        quantized = weights @ self.codebook  # [B, W, num_tag] @ [num_tag, tag_dim] = [B, W, tag_dim]
        
        return quantized, weights

    def get_codebook_usage(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate codebook usage statistics.
        
        Args:
            weights: Gumbel softmax weights, shape: [B, W, num_tag]
            
        Returns:
            Usage per code, shape: [num_tag]
        """
        return weights.mean(dim=(0, 1))  # Average across batch and sequence dimensions

    def get_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy of the discrete distribution.
        
        Args:
            weights: Gumbel softmax weights, shape: [B, W, num_tag]
            
        Returns:
            Entropy value
        """
        # Calculate entropy over the tag dimension
        log_weights = torch.log(weights + 1e-8)  # Add small epsilon for numerical stability
        entropy = -(weights * log_weights).sum(dim=-1).mean()  # Average across batch and sequence
        return entropy


class VQCodebook(nn.Module):
    """Vector Quantization codebook as an alternative to Gumbel softmax."""
    
    def __init__(self, num_tag: int, tag_dim: int, commitment_cost: float = 0.25):
        """
        Initialize VQ codebook.
        
        Args:
            num_tag: Number of discrete codes in the codebook
            tag_dim: Dimension of each code vector
            commitment_cost: Weight for commitment loss
        """
        super().__init__()
        self.num_tag = num_tag
        self.tag_dim = tag_dim
        self.commitment_cost = commitment_cost
        
        self.codebook = nn.Embedding(num_tag, tag_dim)
        # Initialize with uniform distribution
        self.codebook.weight.data.uniform_(-1/num_tag, 1/num_tag)

    def forward(self, inputs: torch.Tensor) -> tuple:
        """
        Forward pass for vector quantization.
        
        Args:
            inputs: Input representations, shape: [B, W, tag_dim]
            
        Returns:
            Tuple of (quantized, vq_loss, encoding_indices)
        """
        # Flatten input for easier computation
        flat_input = inputs.view(-1, self.tag_dim)  # [B*W, tag_dim]
        
        # Calculate distances to all codebook vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.codebook.weight.t()))
        
        # Get closest codebook indices
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [B*W, 1]
        
        # Get quantized representations
        quantized = self.codebook(encoding_indices.squeeze(1))  # [B*W, tag_dim]
        quantized = quantized.view(inputs.shape)  # [B, W, tag_dim]
        
        # Calculate VQ loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, vq_loss, encoding_indices.view(inputs.shape[:-1])
