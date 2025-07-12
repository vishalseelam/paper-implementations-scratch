import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import random


class CBOWModel(nn.Module):
    """
    CBOW (Continuous Bag of Words) model for Word2Vec.
    Predicts target word from context words.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 window_size: int = 5, sparse: bool = False):
        super(CBOWModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        
        # Input embeddings (syn0)
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)
        
        # Output embeddings (syn1) - for hierarchical softmax or negative sampling
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)
        
        # Initialize embeddings
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embedding weights."""
        # Initialize input embeddings with uniform distribution
        nn.init.uniform_(self.input_embeddings.weight, -0.5/self.embedding_dim, 0.5/self.embedding_dim)
        
        # Initialize output embeddings to zero
        nn.init.zeros_(self.output_embeddings.weight)
    
    def forward(self, context_words: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CBOW model.
        
        Args:
            context_words: Tensor of shape (batch_size, context_size) containing context word indices
            
        Returns:
            context_vector: Average of context word embeddings
        """
        # Get embeddings for context words
        context_embeddings = self.input_embeddings(context_words)  # (batch_size, context_size, embedding_dim)
        
        # Average the context embeddings
        context_vector = torch.mean(context_embeddings, dim=1)  # (batch_size, embedding_dim)
        
        return context_vector
    
    def get_input_embeddings(self) -> torch.Tensor:
        """Get input embedding matrix."""
        return self.input_embeddings.weight
    
    def get_output_embeddings(self) -> torch.Tensor:
        """Get output embedding matrix."""
        return self.output_embeddings.weight


class CBOWNegativeSampling(nn.Module):
    """
    CBOW model with negative sampling for efficient training.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 num_negative_samples: int = 5, sparse: bool = False):
        super(CBOWNegativeSampling, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_negative_samples = num_negative_samples
        
        # CBOW model
        self.cbow = CBOWModel(vocab_size, embedding_dim, sparse=sparse)
        
        # Negative sampling table
        self.negative_sampling_table = None
        
    def forward(self, context_words: torch.Tensor, target_words: torch.Tensor, 
                negative_words: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with negative sampling.
        
        Args:
            context_words: Context word indices (batch_size, context_size)
            target_words: Target word indices (batch_size,)
            negative_words: Negative word indices (batch_size, num_negative_samples)
            
        Returns:
            positive_loss: Loss for positive samples
            negative_loss: Loss for negative samples
        """
        # Get context vector
        context_vector = self.cbow(context_words)  # (batch_size, embedding_dim)
        
        # Positive samples
        target_embeddings = self.cbow.output_embeddings(target_words)  # (batch_size, embedding_dim)
        positive_scores = torch.sum(context_vector * target_embeddings, dim=1)  # (batch_size,)
        positive_loss = -F.logsigmoid(positive_scores).mean()
        
        # Negative samples
        negative_embeddings = self.cbow.output_embeddings(negative_words)  # (batch_size, num_neg, embedding_dim)
        
        # Expand context vector for negative sampling
        context_expanded = context_vector.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        negative_scores = torch.sum(context_expanded * negative_embeddings, dim=2)  # (batch_size, num_neg)
        negative_loss = -F.logsigmoid(-negative_scores).mean()
        
        return positive_loss, negative_loss
    
    def set_negative_sampling_table(self, table: np.ndarray):
        """Set the negative sampling table."""
        self.negative_sampling_table = table
    
    def get_negative_samples(self, batch_size: int, exclude_words: List[int] = None) -> torch.Tensor:
        """Sample negative words."""
        if self.negative_sampling_table is None:
            # Random sampling if no table is provided
            negative_samples = torch.randint(0, self.vocab_size, 
                                           (batch_size, self.num_negative_samples))
        else:
            # Use negative sampling table
            indices = np.random.randint(0, len(self.negative_sampling_table), 
                                      (batch_size, self.num_negative_samples))
            negative_samples = torch.tensor(self.negative_sampling_table[indices])
        
        return negative_samples


class CBOWHierarchicalSoftmax(nn.Module):
    """
    CBOW model with hierarchical softmax for efficient training.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 huffman_tree, sparse: bool = False):
        super(CBOWHierarchicalSoftmax, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.huffman_tree = huffman_tree
        
        # CBOW model
        self.cbow = CBOWModel(vocab_size, embedding_dim, sparse=sparse)
        
        # Internal node embeddings for hierarchical softmax
        num_internal_nodes = len(huffman_tree.internal_nodes)
        self.internal_embeddings = nn.Embedding(num_internal_nodes, embedding_dim, sparse=sparse)
        
        # Initialize internal embeddings
        nn.init.zeros_(self.internal_embeddings.weight)
        
        # Get all paths for efficient computation
        self.word_paths = huffman_tree.get_all_paths()
    
    def forward(self, context_words: torch.Tensor, target_words: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hierarchical softmax.
        
        Args:
            context_words: Context word indices (batch_size, context_size)
            target_words: Target word indices (batch_size,)
            
        Returns:
            loss: Hierarchical softmax loss
        """
        # Get context vector
        context_vector = self.cbow(context_words)  # (batch_size, embedding_dim)
        
        batch_size = context_vector.size(0)
        total_loss = 0.0
        
        # Process each sample in the batch
        for i in range(batch_size):
            target_word = target_words[i].item()
            
            if target_word not in self.word_paths:
                continue
                
            code, path = self.word_paths[target_word]
            
            if not code or not path:
                continue
            
            sample_loss = 0.0
            
            # Traverse the path in the Huffman tree
            for j, point in enumerate(path):
                if point >= len(self.huffman_tree.internal_nodes):
                    continue
                
                # Get internal node embedding
                internal_embedding = self.internal_embeddings.weight[point]
                
                # Compute dot product
                score = torch.dot(context_vector[i], internal_embedding)
                
                # Get target (code bit)
                target_bit = code[j]
                
                # Compute binary cross-entropy loss
                if target_bit == 1:
                    sample_loss += -F.logsigmoid(score)
                else:
                    sample_loss += -F.logsigmoid(-score)
            
            total_loss += sample_loss
        
        return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0)


class CBOWTrainer:
    """
    Trainer class for CBOW model.
    """
    
    def __init__(self, model: Union[CBOWNegativeSampling, CBOWHierarchicalSoftmax], 
                 learning_rate: float = 0.025, device: str = 'cpu'):
        self.model = model
        self.learning_rate = learning_rate
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = None
        
    def train_step(self, context_words: torch.Tensor, target_words: torch.Tensor, 
                   negative_words: torch.Tensor = None) -> float:
        """
        Single training step.
        
        Args:
            context_words: Context word indices
            target_words: Target word indices
            negative_words: Negative word indices (for negative sampling)
            
        Returns:
            loss: Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move tensors to device
        context_words = context_words.to(self.device)
        target_words = target_words.to(self.device)
        
        if isinstance(self.model, CBOWNegativeSampling):
            if negative_words is None:
                negative_words = self.model.get_negative_samples(context_words.size(0))
            negative_words = negative_words.to(self.device)
            
            positive_loss, negative_loss = self.model(context_words, target_words, negative_words)
            loss = positive_loss + negative_loss
        else:
            # Hierarchical softmax
            loss = self.model(context_words, target_words)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_learning_rate(self, progress: float):
        """Update learning rate based on training progress."""
        current_lr = self.learning_rate * (1 - progress)
        current_lr = max(current_lr, self.learning_rate * 0.0001)  # Minimum learning rate
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
    
    def save_model(self, filepath: str):
        """Save model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate = checkpoint['learning_rate']
        print(f"Model loaded from {filepath}")
    
    def get_embeddings(self) -> np.ndarray:
        """Get word embeddings."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.model, (CBOWNegativeSampling, CBOWHierarchicalSoftmax)):
                embeddings = self.model.cbow.get_input_embeddings()
            else:
                embeddings = self.model.get_input_embeddings()
        return embeddings.cpu().numpy()


def create_cbow_data_loader(training_pairs: List[Tuple], vocab_size: int, 
                           batch_size: int = 128, shuffle: bool = True):
    """
    Create data loader for CBOW training.
    
    Args:
        training_pairs: List of (context_words, target_word) pairs
        vocab_size: Size of vocabulary
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader for training
    """
    from torch.utils.data import DataLoader, Dataset
    
    class CBOWDataset(Dataset):
        def __init__(self, pairs, vocab_size):
            self.pairs = pairs
            self.vocab_size = vocab_size
            
        def __len__(self):
            return len(self.pairs)
        
        def __getitem__(self, idx):
            context_words, target_word = self.pairs[idx]
            
            # Ensure context_words is a list
            if not isinstance(context_words, list):
                context_words = [context_words]
            
            return torch.tensor(context_words, dtype=torch.long), torch.tensor(target_word, dtype=torch.long)
    
    def collate_fn(batch):
        """Custom collate function to handle variable context sizes."""
        contexts, targets = zip(*batch)
        
        # Pad contexts to the same length
        max_len = max(len(ctx) for ctx in contexts)
        padded_contexts = []
        
        for ctx in contexts:
            padded = ctx.tolist() + [0] * (max_len - len(ctx))  # Pad with zeros
            padded_contexts.append(padded)
        
        return torch.tensor(padded_contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    dataset = CBOWDataset(training_pairs, vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)


if __name__ == "__main__":
    # Test CBOW implementation
    print("Testing CBOW implementation...")
    
    # Parameters
    vocab_size = 1000
    embedding_dim = 100
    window_size = 5
    batch_size = 32
    
    # Create sample data
    sample_pairs = []
    for _ in range(100):
        context_words = [random.randint(0, vocab_size-1) for _ in range(random.randint(2, 8))]
        target_word = random.randint(0, vocab_size-1)
        sample_pairs.append((context_words, target_word))
    
    # Test CBOW with negative sampling
    print("\nTesting CBOW with negative sampling...")
    model_ns = CBOWNegativeSampling(vocab_size, embedding_dim, num_negative_samples=5)
    trainer_ns = CBOWTrainer(model_ns, learning_rate=0.025)
    
    # Create data loader
    data_loader = create_cbow_data_loader(sample_pairs, vocab_size, batch_size=16)
    
    # Training loop
    total_loss = 0.0
    for batch_idx, (context_words, target_words) in enumerate(data_loader):
        loss = trainer_ns.train_step(context_words, target_words)
        total_loss += loss
        
        if batch_idx >= 5:  # Test only a few batches
            break
    
    print(f"Average loss: {total_loss / 6:.4f}")
    
    # Get embeddings
    embeddings = trainer_ns.get_embeddings()
    print(f"Embedding shape: {embeddings.shape}")
    
    print("\nCBOW model implementation completed successfully!") 