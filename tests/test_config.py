"""Test configuration classes."""

import pytest
from pos_tagger.config.config import Config, ExperimentConfig, get_device


def test_config_creation():
    """Test that Config can be created with default values."""
    config = Config()
    
    assert config.use_char_architecture == False
    assert config.bert_model_name == "google/muril-base-cased"
    assert config.max_seq_len == 32
    assert config.num_tag == 100
    assert config.learning_rate == 6e-5
    assert config.char_loss_weight == 1.0 - config.vocab_loss_weight


def test_config_modification():
    """Test that Config values can be modified."""
    config = Config()
    config.epochs = 50
    config.batch_size = 128
    config.use_char_architecture = True
    
    assert config.epochs == 50
    assert config.batch_size == 128
    assert config.use_char_architecture == True


def test_experiment_config():
    """Test ExperimentConfig creation."""
    exp_config = ExperimentConfig()
    
    assert exp_config.data_dir == "data"
    assert exp_config.output_dir == "content"
    assert exp_config.seed == 42


def test_get_device():
    """Test device selection function."""
    device = get_device()
    assert str(device) in ["cpu", "cuda:0", "cuda"]  # Depends on system
