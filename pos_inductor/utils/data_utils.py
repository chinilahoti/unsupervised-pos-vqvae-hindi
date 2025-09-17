"""Utilities for data loading and preprocessing."""

import regex
from typing import List, Tuple, Dict, Set, Union


def read_conllu_file(file_path: str) -> Tuple[List[List[Tuple[str, str]]], Set[str]]:
    """
    Read CoNLL-U format files for labeled data.
    
    Args:
        file_path: Path to the CoNLL-U file
        
    Returns:
        Tuple of (sentences, unique_labels) where sentences is a list of 
        (word, tag) tuples and unique_labels is a set of all tags
    """
    sentences = []
    unique_labels = set()

    with open(file_path, "r", encoding="UTF-8") as in_f:
        current_sentence = []
        for line in in_f:
            line = line.strip()
            if line.startswith("#") or line == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            parts = line.split("\t")
            idx = parts[0]

            if "." in idx or "-" in idx or len(parts) < 4:
                continue

            word, tag = parts[1], parts[3]
            # Keep only Devanagari characters
            word = regex.sub(r"[^\p{Devanagari}+]", "", word)
            if word != "":
                unique_labels.add(tag)
                current_sentence.append((word, tag))

    return sentences, unique_labels


def item_indexer(list_of_items: List[str], labels: bool = False) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create bidirectional mapping between items and indices.
    
    Args:
        list_of_items: List of items to index
        labels: If True, don't add UNK and PAD tokens
        
    Returns:
        Tuple of (item2index, index2item) dictionaries
    """
    item2index = {item: idx + 1 for idx, item in enumerate(dict.fromkeys(sorted(list_of_items)))}

    if not labels:
        item2index["UNK"] = len(item2index)
        item2index["PAD"] = 0

    index2item = {idx: item for item, idx in item2index.items()}

    return item2index, index2item


def extract_vocabulary_and_chars(sentences: List[List[Tuple[str, str]]]) -> Tuple[List[str], List[str]]:
    """
    Extract vocabulary and character sets from sentences.
    
    Args:
        sentences: List of sentences, each containing (word, tag) tuples
        
    Returns:
        Tuple of (vocabulary_list, unique_characters_list)
    """
    vocabulary = [word for sentence in sentences for word, _ in sentence]
    unique_chars = list(set(' '.join(vocabulary)))

    return vocabulary, unique_chars


def prepare_data_indices(train_sents: List[List[Tuple[str, str]]],
                         train_labels: Set[str]) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    """
    Prepare all indexing dictionaries for the dataset.
    
    Args:
        train_sents: Training sentences
        train_labels: Set of training labels
        
    Returns:
        Tuple of (l2i, i2l, v2i, i2v, c2i, i2c) dictionaries
    """
    vocabulary, unique_chars = extract_vocabulary_and_chars(train_sents)

    l2i, i2l = item_indexer(train_labels, labels=True)
    v2i, i2v = item_indexer(vocabulary, labels=False)
    c2i, i2c = item_indexer(unique_chars, labels=False)

    return l2i, i2l, v2i, i2v, c2i, i2c
