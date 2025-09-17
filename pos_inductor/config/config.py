"""Configuration settings for the POS tagging model."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for POS tagging model."""
    
    # Architecture settings
    use_char_architecture: bool = False
    bert_model_name: str = "google/muril-base-cased"
    
    # Sequence and token limits
    max_seq_len: int = 32
    max_tok_len: int = 54
    max_word_len: int = 10
    
    # Embedding dimensions
    char_dim: int = 64
    word_dim: int = 128
    emb_hidden: int = 768  # Must match BERT hidden size
    
    # Tag settings
    num_tag: int = 100
    tag_dim: int = 50
    
    # Decoder settings
    dec_hidden: int = 256
    dec_layers: int = 2
    
    # Training hyperparameters
    learning_rate: float = 6e-5
    epochs: int = 100
    batch_size: int = 256
    
    # Loss weights and temperature
    gumbel_temperature: float = 1.0
    diversity_weight: float = 0.6
    vocab_loss_weight: float = 1.0
    
    @property
    def char_loss_weight(self) -> float:
        """Character loss weight is complement of vocab loss weight."""
        return 1.0 - self.vocab_loss_weight


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""
    
    data_dir: str = "../../../../Downloads/data"
    output_dir: str = "content"
    save_every_n_epochs: int = 10
    use_cuda: bool = True
    seed: int = 42
    
    # Data file paths
    train_file: str = "hi_hdtb-ud-train.conllu"
    dev_file: str = "hi_hdtb-ud-dev.conllu" 
    test_file: str = "hi_hdtb-ud-test.conllu"


def get_device():
    """Get the appropriate device for training."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
