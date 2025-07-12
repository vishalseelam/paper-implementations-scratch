import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, List, Tuple
import math
from tqdm import tqdm

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss as used in the paper.
    """
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch_size, seq_len, vocab_size)
            target: (batch_size, seq_len)
        """
        batch_size, seq_len, vocab_size = pred.size()
        
        # Reshape predictions and targets
        pred = pred.view(-1, vocab_size)
        target = target.view(-1)
        
        # Create smoothed target distribution
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (vocab_size - 2))  # -2 for padding and true class
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        
        # Mask out padding tokens
        mask = (target != self.padding_idx)
        
        # Calculate KL divergence
        kl_div = nn.functional.kl_div(
            nn.functional.log_softmax(pred, dim=1),
            true_dist,
            reduction='none'
        ).sum(dim=1)
        
        # Apply mask and return mean
        return (kl_div * mask).sum() / mask.sum()


class NoamOptimizer:
    """
    Optimizer with learning rate scheduling as described in the paper.
    """
    def __init__(self, model_size: int, factor: float = 2.0, warmup: int = 4000,
                 optimizer: Optional[optim.Optimizer] = None):
        self.model_size = model_size
        self.factor = factor
        self.warmup = warmup
        self.optimizer = optimizer
        self._step = 0
        self._rate = 0
        
    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step: Optional[int] = None) -> float:
        """Implement learning rate scheduling"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * 
                             min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()


class TransformerTrainer:
    """
    Training utilities for the Transformer model.
    """
    def __init__(self, model: nn.Module, train_loader, val_loader, 
                 src_vocab_size: int, tgt_vocab_size: int,
                 device: torch.device, pad_idx: int = 0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.pad_idx = pad_idx
        
        # Initialize loss function
        self.criterion = LabelSmoothingLoss(
            vocab_size=tgt_vocab_size,
            padding_idx=pad_idx,
            smoothing=0.1
        )
        
        # Initialize optimizer
        self.optimizer = NoamOptimizer(
            model_size=model.d_model,
            factor=2.0,
            warmup=4000,
            optimizer=optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        )
        
        # Move model to device
        self.model.to(device)
        
    def create_masks(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create masks for source and target sequences.
        """
        batch_size = src.size(0)
        src_len = src.size(1)
        tgt_len = tgt.size(1)
        
        # Source padding mask
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        src_mask = src_mask.expand(batch_size, 1, src_len, src_len)
        
        # Target padding mask
        tgt_padding_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_padding_mask = tgt_padding_mask.expand(batch_size, 1, tgt_len, tgt_len)
        
        # Target look-ahead mask
        tgt_look_ahead_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1)
        tgt_look_ahead_mask = tgt_look_ahead_mask.bool().unsqueeze(0).unsqueeze(0)
        tgt_look_ahead_mask = tgt_look_ahead_mask.expand(batch_size, 1, tgt_len, tgt_len)
        
        # Combine target masks
        tgt_mask = tgt_padding_mask & ~tgt_look_ahead_mask
        
        return src_mask.to(self.device), tgt_mask.to(self.device)
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            src, tgt = batch
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # Create input and target sequences
            tgt_input = tgt[:, :-1]  # Remove last token for input
            tgt_output = tgt[:, 1:]  # Remove first token for target
            
            # Create masks
            src_mask, tgt_mask = self.create_masks(src, tgt_input)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            
            # Calculate loss
            loss = self.criterion(output, tgt_output)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item(), 'lr': self.optimizer._rate})
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """
        Validate the model.
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                src, tgt = batch
                src, tgt = src.to(self.device), tgt.to(self.device)
                
                # Create input and target sequences
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                # Create masks
                src_mask, tgt_mask = self.create_masks(src, tgt_input)
                
                # Forward pass
                output = self.model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
                
                # Calculate loss
                loss = self.criterion(output, tgt_output)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, num_epochs: int, save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        """
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            val_losses.append(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save model if path provided
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                }, f"{save_path}_epoch_{epoch + 1}.pt")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }


def create_data_loaders(src_data: List[List[int]], tgt_data: List[List[int]],
                       batch_size: int = 32, shuffle: bool = True,
                       val_split: float = 0.2) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create data loaders for training and validation.
    """
    # Convert to tensors
    src_tensors = [torch.tensor(seq, dtype=torch.long) for seq in src_data]
    tgt_tensors = [torch.tensor(seq, dtype=torch.long) for seq in tgt_data]
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        torch.nn.utils.rnn.pad_sequence(src_tensors, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(tgt_tensors, batch_first=True)
    )
    
    # Split dataset
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, val_loader


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history.
    """
    import matplotlib.pyplot as plt
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show() 