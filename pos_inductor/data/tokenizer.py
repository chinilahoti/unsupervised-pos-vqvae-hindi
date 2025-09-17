"""Tokenizer for handling vocabulary, character, and BERT tokenization."""

import torch
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer


class Tokenizer:
    """Handles tokenization for vocabulary, characters, and BERT."""
    
    def __init__(self, 
                 vocab2index: Dict[str, int], 
                 label2index: Dict[str, int], 
                 char2index: Dict[str, int], 
                 config,
                 bert_tokenizer: AutoTokenizer):
        """
        Initialize tokenizer with index mappings.
        
        Args:
            vocab2index: Vocabulary to index mapping
            label2index: Label to index mapping  
            char2index: Character to index mapping
            config: Configuration object
            bert_tokenizer: Pre-trained BERT tokenizer
        """
        self.vocab2index = vocab2index
        self.label2index = label2index
        self.char2index = char2index
        self.config = config
        self.bert_tokenizer = bert_tokenizer
        self.pad_label_id = -100
        self.max_seq_len = self.config.max_seq_len
        self.max_word_len = self.config.max_word_len

    def _pad_or_truncate(self, sequence: List, max_length: int, pad_token: str = "PAD") -> List:
        """Pad or truncate sequence to specified length."""
        return sequence[:max_length] + [pad_token] * (max_length - len(sequence))

    def _vocab_tokenizer(self, input_words: List[str]) -> torch.LongTensor:
        """Convert words to vocabulary indices."""
        vocab_ids = [
            self.vocab2index[word] if word in self.vocab2index else self.vocab2index["UNK"] 
            for word in input_words
        ]
        return torch.LongTensor(vocab_ids)

    def _char_tokenizer(self, input_words: List[str]) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """Convert words to character indices with word alignment."""
        char_ids = []
        char_word_ids = []
        filtered_word_idx = 0
        
        for word in input_words:
            if word != "PAD":
                processed_word = self._pad_or_truncate(list(word), self.max_word_len)
                for char in processed_word:
                    if char in self.char2index:
                        char_ids.append(self.char2index[char])
                    else:
                        char_ids.append(self.char2index["UNK"])
                    char_word_ids.append(filtered_word_idx)
            else:
                for i in range(self.max_word_len):
                    char_ids.append(self.char2index["PAD"])
                    char_word_ids.append(filtered_word_idx)
            filtered_word_idx += 1

        return torch.LongTensor(char_ids), torch.LongTensor(char_word_ids)

    def _bert_tokenizer(self, input_words: List[str]) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """Tokenize using BERT tokenizer."""
        bert_tokenized_input = self.bert_tokenizer(
            input_words,
            is_split_into_words=True,
            padding='max_length',
            max_length=self.config.max_tok_len,
            truncation=True,
            return_tensors='pt',
            return_special_tokens_mask=True
        )

        token_ids = bert_tokenized_input['input_ids'].squeeze(0)
        attention_mask = bert_tokenized_input['attention_mask'].squeeze(0)
        special_tokens_mask = bert_tokenized_input['special_tokens_mask'].squeeze(0)

        token_word_ids = bert_tokenized_input.word_ids()
        token_word_ids = [id for id in token_word_ids if id is not None]

        return token_ids, attention_mask, special_tokens_mask, torch.LongTensor(token_word_ids)

    def _label_tokenizer(self, tags: List[str]) -> torch.LongTensor:
        """Convert tags to label indices."""
        label_ids = [self.label2index[tag] if tag != "PAD" else -100 for tag in tags]
        return torch.LongTensor(label_ids)

    def tokenize(self, input_words: List[str], tags: List[str]) -> Dict[str, Optional[torch.LongTensor]]:
        """
        Tokenize input words and tags for model consumption.
        
        Args:
            input_words: List of input words
            tags: List of corresponding tags
            
        Returns:
            Dictionary containing tokenized tensors
        """
        processed_input = self._pad_or_truncate(input_words, self.max_seq_len)
        processed_tags = self._pad_or_truncate(tags, self.max_seq_len)
        
        vocab_ids = self._vocab_tokenizer(processed_input)
        token_ids, attention_mask, special_tokens_mask, token_word_ids = self._bert_tokenizer(processed_input)
        labels = self._label_tokenizer(processed_tags)

        tensor_dictionary = {
            "vocab_ids": vocab_ids,
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "spec_tok_mask": special_tokens_mask,
            "token_word_ids": token_word_ids,
            "labels": labels,
            "char_ids": None,
            "char_word_ids": None
        }

        if self.config.use_char_architecture:
            char_ids, char_word_ids = self._char_tokenizer(processed_input)
            tensor_dictionary["char_ids"] = char_ids
            tensor_dictionary["char_word_ids"] = char_word_ids

        return tensor_dictionary
