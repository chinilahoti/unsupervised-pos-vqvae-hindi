"""Training utilities for the POS tagging model."""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
from typing import List, Tuple, Optional, Dict
import os
import datetime


class Trainer:
    """Trainer class for the POS tagging model."""
    
    def __init__(self, 
                 model,
                 config,
                 device: torch.device,
                 char2index: Optional[Dict[str, int]] = None):
        """
        Initialize trainer.
        
        Args:
            model: The tagging model to train
            config: Configuration object
            device: Device for training
            char2index: Character to index mapping (needed for character decoding)
        """
        self.model = model
        self.config = config
        self.device = device
        self.char2index = char2index
        
        # Initialize optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.diversity_losses = []
        self.vocab_losses = []
        self.char_losses = []

    def train_epoch(self, 
                   train_loader: DataLoader, 
                   epoch: int,
                   batch_counter: Optional[int] = None) -> Tuple[float, float, float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            batch_counter: Optional limit on number of batches
            
        Returns:
            Tuple of (avg_loss, avg_div_loss, avg_vocab_loss, avg_char_loss)
        """
        self.model.train()
        
        epoch_loss = 0
        epoch_div_loss = 0
        epoch_vocab_loss = 0
        epoch_char_loss = 0
        num_batches = 0

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
        
        for i, batch in enumerate(pbar):
            if batch_counter is not None and i >= batch_counter:
                break

            # Move batch to device
            token_ids = batch['token_ids'].to(self.device)
            token_word_ids = batch['token_word_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            vocab_ids = batch['vocab_ids'].to(self.device)
            special_tokens_mask = batch['spec_tok_mask'].to(self.device)
            
            char_ids = None
            char_word_ids = None
            if self.config.use_char_architecture:
                char_ids = batch['char_ids'].to(self.device) if batch['char_ids'] is not None else None
                char_word_ids = batch['char_word_ids'].to(self.device) if batch['char_word_ids'] is not None else None

            # Zero gradients
            self.optimizer.zero_grad()

            # Compute loss
            loss, vocab_reconstr, char_reconstr, div_loss, _ = self.model.compute_loss(
                token_ids, token_word_ids, attention_mask, special_tokens_mask, 
                vocab_ids, char_ids, char_word_ids, self.char2index, self.device
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            epoch_loss += loss.item()
            epoch_div_loss += div_loss.item()
            epoch_vocab_loss += vocab_reconstr.item()
            epoch_char_loss += char_reconstr.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Div': f'{div_loss.item():.4f}',
                'Vocab': f'{vocab_reconstr.item():.4f}',
                'Char': f'{char_reconstr.item():.4f}'
            })

        # Calculate averages
        avg_loss = epoch_loss / num_batches
        avg_div_loss = epoch_div_loss / num_batches
        avg_vocab_loss = epoch_vocab_loss / num_batches
        avg_char_loss = epoch_char_loss / num_batches

        return avg_loss, avg_div_loss, avg_vocab_loss, avg_char_loss

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc="Validating"):
                token_ids = batch['token_ids'].to(self.device)
                token_word_ids = batch['token_word_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                vocab_ids = batch['vocab_ids'].to(self.device)
                special_tokens_mask = batch['spec_tok_mask'].to(self.device)
                
                char_ids = None
                char_word_ids = None
                if self.config.use_char_architecture:
                    char_ids = batch['char_ids'].to(self.device) if batch['char_ids'] is not None else None
                    char_word_ids = batch['char_word_ids'].to(self.device) if batch['char_word_ids'] is not None else None

                loss, _, _, _, _ = self.model.compute_loss(
                    token_ids, token_word_ids, attention_mask, special_tokens_mask,
                    vocab_ids, char_ids, char_word_ids, self.char2index, self.device
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, 
             train_loader: DataLoader, 
             val_loader: DataLoader,
             save_dir: str,
             experiment_name: str,
             batch_counter: Optional[int] = None) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory to save model checkpoints
            experiment_name: Name of the experiment
            batch_counter: Optional limit on batches per epoch
            
        Returns:
            Tuple of training metrics lists
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {self.config.epochs} epochs...")
        
        for epoch in range(self.config.epochs):
            # Train epoch
            train_loss, div_loss, vocab_loss, char_loss = self.train_epoch(
                train_loader, epoch, batch_counter
            )
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.diversity_losses.append(div_loss)
            self.vocab_losses.append(vocab_loss)
            self.char_losses.append(char_loss)
            
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Diversity Loss: {div_loss:.4f}")
            print(f"  Vocab Loss: {vocab_loss:.4f}")
            print(f"  Char Loss: {char_loss:.4f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
                self.save_checkpoint(checkpoint_path, epoch)
                
        # Save final model
        final_path = os.path.join(save_dir, "final_model.pt")
        self.save_checkpoint(final_path, self.config.epochs - 1)
        
        print("Training completed!")
        
        return (self.train_losses, self.val_losses, self.diversity_losses, 
                self.vocab_losses, self.char_losses)

    def save_checkpoint(self, path: str, epoch: int):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> int:
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
            
        Returns:
            Epoch number from checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {path}, epoch {epoch}")
        return epoch