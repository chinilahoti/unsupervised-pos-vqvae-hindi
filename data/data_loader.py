"""Dataset and DataLoader classes for POS tagging."""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict


class POSDataset(Dataset):
    """Dataset class for POS tagging data."""
    
    def __init__(self, sentences: List[List[Tuple[str, str]]], tokenizer, config):
        """
        Initialize dataset.
        
        Args:
            sentences: List of sentences, each containing (word, tag) tuples
            tokenizer: Tokenizer instance
            config: Configuration object
        """
        super().__init__()
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Dict[str, torch.LongTensor]:
        """
        Get item by index.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary containing tokenized tensors
        """
        input_words, tags = zip(*self.sentences[idx])
        input_words = list(input_words)
        tags = list(tags)

        return self.tokenizer.tokenize(input_words, tags)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.LongTensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of dictionaries from dataset
        
    Returns:
        Batched dictionary of tensors
    """
    vocab_ids = torch.stack([item['vocab_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    char_ids = None
    char_word_ids = None
    if batch[0]['char_ids'] is not None:
        char_ids = torch.stack([item['char_ids'] for item in batch])
    if batch[0]['char_word_ids'] is not None:
        char_word_ids = torch.stack([item['char_word_ids'] for item in batch])

    # Pad sequences to maximum length
    max_len = 54

    def pad_sequence(seq_list, pad_value):
        """Pad sequences to max_len."""
        return torch.stack([
            torch.cat([s, torch.full((max_len - s.size(0),), pad_value, dtype=torch.long)])
            if s.size(0) < max_len else s[:max_len]
            for s in seq_list
        ])

    token_ids = pad_sequence([item['token_ids'] for item in batch], pad_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], pad_value=0)
    token_word_ids = pad_sequence([item['token_word_ids'] for item in batch], pad_value=0)
    spec_tok_mask = pad_sequence([item['spec_tok_mask'] for item in batch], pad_value=1)

    return {
        "vocab_ids": vocab_ids,
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "token_word_ids": token_word_ids,
        "spec_tok_mask": spec_tok_mask,
        "char_ids": char_ids,
        "char_word_ids": char_word_ids,
        "labels": labels
    }


def create_data_loaders(train_dataset: POSDataset, 
                       dev_dataset: POSDataset, 
                       test_dataset: POSDataset, 
                       batch_size: int, 
                       shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, dev, and test datasets.
    
    Args:
        train_dataset: Training dataset
        dev_dataset: Development dataset
        test_dataset: Test dataset
        batch_size: Batch size for data loaders
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, dev_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train, 
        collate_fn=collate_fn
    )
    
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_loader, dev_loader, test_loader
