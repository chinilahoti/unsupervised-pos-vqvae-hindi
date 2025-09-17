# Unsupervised Part-of-Speech Induction for a Morphologically Rich Language using Gumbel Softmax

A PyTorch implementation of a neural part-of-speech induction system that combines BERT embeddings with character-level representations and uses discrete representation learning through Gumbel softmax for unsupervised morphosyntactic analysis.

## Features

- **Multi-modal embeddings**: Combines BERT token embeddings with character-level LSTM embeddings
- **Discrete representation learning**: Uses Gumbel softmax codebook for learning discrete morphosyntactic categories
- **Flexible decoders**: Support for both vocabulary and character-level reconstruction
- **Comprehensive evaluation**: Many-to-one accuracy, clustering metrics, and visualization tools
- **Modular architecture**: Easy to extend and experiment with different components
