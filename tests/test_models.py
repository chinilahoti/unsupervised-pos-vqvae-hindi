"""Test model components with small tensors."""

import torch
import pytest
from pos_tagger.config.config import Config
from pos_tagger.models.encoders import EncoderFFN
from pos_tagger.models.codebook import GumbelCodebook
from pos_tagger.models.decoders import DecoderFFN, DecoderBiLSTM


def test_encoder_ffn():
    """Test FFN encoder with small tensors."""
    emb_hidden = 8
    num_tag = 5
    batch_size = 2
    seq_len = 4
    
    encoder = EncoderFFN(emb_hidden, num_tag)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, emb_hidden)
    
    # Forward pass
    output = encoder(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, num_tag)
    assert not torch.isnan(output).any(), "Output contains NaN values"


def test_gumbel_codebook():
    """Test Gumbel codebook with small tensors."""
    num_tag = 5
    tag_dim = 6
    batch_size = 2
    seq_len = 4
    
    codebook = GumbelCodebook(num_tag, tag_dim)
    
    # Create dummy logits
    logits = torch.randn(batch_size, seq_len, num_tag)
    
    # Forward pass
    quantized, weights = codebook(logits, temperature=1.0)
    
    # Check shapes
    assert quantized.shape == (batch_size, seq_len, tag_dim)
    assert weights.shape == (batch_size, seq_len, num_tag)
    
    # Check that weights sum to 1 (softmax property)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
    
    # Check no NaN values
    assert not torch.isnan(quantized).any()
    assert not torch.isnan(weights).any()


def test_decoder_ffn():
    """Test FFN decoder with small tensors."""
    tag_dim = 6
    vocab_size = 10
    batch_size = 2
    seq_len = 4
    
    decoder = DecoderFFN(tag_dim, vocab_size)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, tag_dim)
    
    # Forward pass
    output = decoder(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, vocab_size)
    assert not torch.isnan(output).any()


def test_decoder_bilstm():
    """Test BiLSTM decoder with small tensors."""
    tag_dim = 6
    dec_hidden = 8
    vocab_size = 10
    num_layers = 1
    batch_size = 2
    seq_len = 4
    
    decoder = DecoderBiLSTM(tag_dim, dec_hidden, vocab_size, num_layers)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, tag_dim)
    
    # Forward pass
    output = decoder(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, vocab_size)
    assert not torch.isnan(output).any()


def test_model_integration():
    """Test that encoder -> codebook -> decoder chain works."""
    # Small dimensions for fast testing
    emb_hidden = 8
    num_tag = 5
    tag_dim = 6
    vocab_size = 10
    batch_size = 2
    seq_len = 4
    
    # Create components
    encoder = EncoderFFN(emb_hidden, num_tag)
    codebook = GumbelCodebook(num_tag, tag_dim)
    decoder = DecoderFFN(tag_dim, vocab_size)
    
    # Create dummy input (like word embeddings)
    word_embeddings = torch.randn(batch_size, seq_len, emb_hidden)
    
    # Forward pass through the chain
    enc_logits = encoder(word_embeddings)
    quantized, weights = codebook(enc_logits)
    vocab_logits = decoder(quantized)
    
    # Check final output
    assert vocab_logits.shape == (batch_size, seq_len, vocab_size)
    assert not torch.isnan(vocab_logits).any()
    
    # Check that we can compute loss
    dummy_targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    loss_fn = torch.nn.CrossEntropyLoss()
    
    vocab_logits_flat = vocab_logits.view(-1, vocab_size)
    targets_flat = dummy_targets.view(-1)
    
    loss = loss_fn(vocab_logits_flat, targets_flat)
    assert not torch.isnan(loss)
    assert loss.item() > 0  # Should be positive


def test_gradients_flow():
    """Test that gradients flow through the model components."""
    emb_hidden = 8
    num_tag = 5
    tag_dim = 6
    vocab_size = 10
    
    # Create components
    encoder = EncoderFFN(emb_hidden, num_tag)
    codebook = GumbelCodebook(num_tag, tag_dim)
    decoder = DecoderFFN(tag_dim, vocab_size)
    
    # Create dummy data
    x = torch.randn(1, 3, emb_hidden, requires_grad=True)
    targets = torch.randint(0, vocab_size, (1, 3))
    
    # Forward pass
    enc_logits = encoder(x)
    quantized, weights = codebook(enc_logits)
    vocab_logits = decoder(quantized)
    
    # Compute loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(vocab_logits.view(-1, vocab_size), targets.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Check that gradients exist
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    
    # Check that model parameters have gradients
    for param in encoder.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


if __name__ == "__main__":
    # Run tests manually
    test_encoder_ffn()
    test_gumbel_codebook()
    test_decoder_ffn() 
    test_decoder_bilstm()
    test_model_integration()
    test_gradients_flow()
    print("All model tests passed!")
