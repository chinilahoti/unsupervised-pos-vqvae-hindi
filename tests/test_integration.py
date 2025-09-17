"""Integration test with minimal data to verify the full pipeline works."""

import torch
import tempfile
import os
from pos_tagger.config.config import Config
from pos_tagger.utils.data_utils import read_conllu_file, prepare_data_indices
from pos_tagger.data.tokenizer import Tokenizer
from pos_tagger.data.data_loader import POSDataset, create_data_loaders
from pos_tagger.models.embeddings import Embeddings
from pos_tagger.models.encoders import EncoderFFN
from pos_tagger.models.codebook import GumbelCodebook
from pos_tagger.models.decoders import DecoderFFN
from pos_tagger.models.tagging_model import TaggingModel
from transformers import AutoTokenizer, AutoModel


def create_minimal_conllu():
    """Create minimal test data."""
    return """# sent_id = 1
1	राम	राम	PROPN	_	_	_	_	_	_
2	घर	घर	NOUN	_	_	_	_	_	_

# sent_id = 2  
1	वह	वह	PRON	_	_	_	_	_	_
2	पानी	पानी	NOUN	_	_	_	_	_	_
"""


def test_full_pipeline_minimal():
    """Test the full pipeline with minimal data and tiny model."""
    # Create temporary data file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conllu', delete=False) as f:
        f.write(create_minimal_conllu())
        temp_file = f.name
    
    try:
        # Create minimal config
        config = Config()
        config.max_seq_len = 4  # Very small
        config.max_tok_len = 8
        config.max_word_len = 3
        config.num_tag = 3  # Very few tags
        config.tag_dim = 4
        config.emb_hidden = 16  # Small BERT-like hidden size
        config.char_dim = 8
        config.word_dim = 12
        config.batch_size = 2
        config.use_char_architecture = False  # Disable for speed
        
        # Load minimal data
        sentences, labels = read_conllu_file(temp_file)
        l2i, i2l, v2i, i2v, c2i, i2c = prepare_data_indices(sentences, labels)
        
        print(f"Loaded {len(sentences)} sentences")
        print(f"Vocabulary size: {len(v2i)}")
        print(f"Labels: {labels}")
        
        # Use a very small model for testing
        try:
            bert_tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
            bert_model = AutoModel.from_pretrained('prajjwal1/bert-tiny')
            config.emb_hidden = 128  # bert-tiny hidden size
        except:
            # Fallback if bert-tiny not available
            print("Using distilbert as fallback...")
            bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
            config.emb_hidden = 768
        
        # Create tokenizer
        tokenizer = Tokenizer(v2i, l2i, c2i, config, bert_tokenizer)
        
        # Create dataset (duplicate data to have enough for batching)
        extended_sentences = sentences * 2  # Duplicate to have 4 sentences
        dataset = POSDataset(extended_sentences, tokenizer, config)
        
        print(f"Dataset size: {len(dataset)}")
        
        # Create a single batch manually to test
        batch_data = []
        for i in range(min(2, len(dataset))):
            batch_data.append(dataset[i])
        
        # Test tokenization worked
        sample = batch_data[0]
        print(f"Sample vocab_ids shape: {sample['vocab_ids'].shape}")
        print(f"Sample token_ids shape: {sample['token_ids'].shape}")
        
        # Create minimal model components
        embeddings = Embeddings(
            num_chars=len(c2i),
            config=config,
            bert_model=bert_model,
            num_layers=1
        )
        
        encoder = EncoderFFN(config.emb_hidden, config.num_tag)
        codebook = GumbelCodebook(config.num_tag, config.tag_dim)
        decoder = DecoderFFN(config.tag_dim, len(v2i))
        
        # Create model
        model = TaggingModel(
            config=config,
            embeddings=embeddings,
            encoder=encoder,
            codebook=codebook,
            vocab_decoder=decoder,
            char_decoder=None,
            vocab_size=len(v2i)
        )
        
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass with a single sample
        device = torch.device('cpu')  # Force CPU for testing
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            # Prepare single batch
            token_ids = sample['token_ids'].unsqueeze(0).to(device)
            token_word_ids = sample['token_word_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            spec_tok_mask = sample['spec_tok_mask'].unsqueeze(0).to(device)
            vocab_ids = sample['vocab_ids'].unsqueeze(0).to(device)
            
            print(f"Input shapes - tokens: {token_ids.shape}, vocab: {vocab_ids.shape}")
            
            # Forward pass
            label_probs, word_probs, char_probs, word_mask, weights = model.forward(
                None, None,  # No char data
                token_ids, token_word_ids, attention_mask, spec_tok_mask, device
            )
            
            print(f"Output shapes:")
            print(f"  Label probs: {label_probs.shape}")
            print(f"  Word probs: {word_probs.shape}")
            print(f"  Word mask: {word_mask.shape}")
            print(f"  Weights: {weights.shape}")
            
            # Test loss computation
            loss, vocab_loss, char_loss, div_loss, pred_tags = model.compute_loss(
                token_ids, token_word_ids, attention_mask, spec_tok_mask,
                vocab_ids, None, None, None, device
            )
            
            print(f"Losses:")
            print(f"  Total: {loss.item():.4f}")
            print(f"  Vocab: {vocab_loss.item():.4f}")
            print(f"  Diversity: {div_loss.item():.4f}")
            
            # Check for NaN values
            assert not torch.isnan(loss), "Loss contains NaN"
            assert not torch.isnan(label_probs).any(), "Label probs contain NaN"
            assert not torch.isnan(word_probs).any(), "Word probs contain NaN"
            
            print("✓ Forward pass successful - no NaN values")
            print("✓ Loss computation successful")
            print("✓ Full pipeline integration test PASSED")
            
    finally:
        os.unlink(temp_file)


if __name__ == "__main__":
    test_full_pipeline_minimal()
