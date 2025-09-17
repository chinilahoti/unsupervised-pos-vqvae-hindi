"""Test data utilities."""

import pytest
import tempfile
import os
from pos_tagger.utils.data_utils import read_conllu_file, item_indexer, prepare_data_indices


def create_sample_conllu_file():
    """Create a small sample CoNLL-U file for testing."""
    content = """# sent_id = 1
# text = राम घर गया।
1	राम	राम	PROPN	_	_	_	_	_	_
2	घर	घर	NOUN	_	_	_	_	_	_
3	गया	जा	VERB	_	_	_	_	_	_
4	।	।	PUNCT	_	_	_	_	_	_

# sent_id = 2
# text = वह पानी पीता है।
1	वह	वह	PRON	_	_	_	_	_	_
2	पानी	पानी	NOUN	_	_	_	_	_	_
3	पीता	पी	VERB	_	_	_	_	_	_
4	है	है	AUX	_	_	_	_	_	_
5	।	।	PUNCT	_	_	_	_	_	_

"""
    return content


def test_read_conllu_file():
    """Test reading CoNLL-U format files."""
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.conllu', delete=False) as f:
        f.write(create_sample_conllu_file())
        temp_file = f.name
    
    try:
        sentences, labels = read_conllu_file(temp_file)
        
        # Check basic structure
        assert len(sentences) == 2  # Two sentences
        assert len(sentences[0]) == 4  # First sentence has 4 tokens
        assert len(sentences[1]) == 5  # Second sentence has 5 tokens
        
        # Check first sentence content
        first_sent = sentences[0]
        assert first_sent[0] == ('राम', 'PROPN')
        assert first_sent[1] == ('घर', 'NOUN')
        assert first_sent[2] == ('गया', 'VERB')
        assert first_sent[3] == ('।', 'PUNCT')
        
        # Check labels
        expected_labels = {'PROPN', 'NOUN', 'VERB', 'PUNCT', 'PRON', 'AUX'}
        assert labels == expected_labels
        
    finally:
        os.unlink(temp_file)


def test_item_indexer():
    """Test item indexer function."""
    items = ['apple', 'banana', 'apple', 'cherry', 'banana']
    
    # Test with labels=False (adds UNK and PAD)
    item2idx, idx2item = item_indexer(items, labels=False)
    
    assert 'PAD' in item2idx
    assert 'UNK' in item2idx
    assert item2idx['PAD'] == 0
    assert len(item2idx) == 5  # apple, banana, cherry, UNK, PAD
    
    # Test bidirectional mapping
    for item, idx in item2idx.items():
        assert idx2item[idx] == item
    
    # Test with labels=True (no UNK/PAD)
    label2idx, idx2label = item_indexer(['NOUN', 'VERB', 'NOUN'], labels=True)
    
    assert 'PAD' not in label2idx
    assert 'UNK' not in label2idx
    assert len(label2idx) == 2  # NOUN, VERB


def test_prepare_data_indices():
    """Test preparing all data indices."""
    # Create sample data
    sentences = [
        [('राम', 'PROPN'), ('घर', 'NOUN')],
        [('वह', 'PRON'), ('पानी', 'NOUN')]
    ]
    labels = {'PROPN', 'NOUN', 'PRON'}
    
    l2i, i2l, v2i, i2v, c2i, i2c = prepare_data_indices(sentences, labels)
    
    # Check that all mappings exist
    assert len(l2i) == 3  # 3 labels
    assert len(v2i) == 6  # 4 words + UNK + PAD
    assert 'PAD' in v2i and 'UNK' in v2i
    assert 'PAD' in c2i and 'UNK' in c2i
    
    # Check bidirectional mappings work
    for label, idx in l2i.items():
        assert i2l[idx] == label
    
    for word, idx in v2i.items():
        assert i2v[idx] == word


if __name__ == "__main__":
    # Run tests manually
    test_read_conllu_file()
    test_item_indexer()
    test_prepare_data_indices()
    print("All data utils tests passed!")
