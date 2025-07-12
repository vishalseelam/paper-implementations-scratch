import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
import random


class SkipGramModel(nn.Module):
    """
    Skip-gram model for Word2Vec.
    Predicts context words from target word.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 window_size: int = 5, sparse: bool = False):
        super(SkipGramModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        
        # Input embeddings (syn0) - target word embeddings
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)
        
        # Output embeddings (syn1) - context word embeddings
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)
        
        # Initialize embeddings
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embedding weights."""
        # Initialize input embeddings with uniform distribution
        nn.init.uniform_(self.input_embeddings.weight, -0.5/self.embedding_dim, 0.5/self.embedding_dim)
        
        # Initialize output embeddings to zero
        nn.init.zeros_(self.output_embeddings.weight)
    
    def forward(self, target_words: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Skip-gram model.
        
        Args:
            target_words: Tensor of shape (batch_size,) containing target word indices
            
        Returns:
            target_embeddings: Embeddings for target words
        """
        # Get embeddings for target words
        target_embeddings = self.input_embeddings(target_words)  # (batch_size, embedding_dim)
        
        return target_embeddings
    
    def get_input_embeddings(self) -> torch.Tensor:
        """Get input embedding matrix."""
        return self.input_embeddings.weight
    
    def get_output_embeddings(self) -> torch.Tensor:
        """Get output embedding matrix."""
        return self.output_embeddings.weight


class SkipGramNegativeSampling(nn.Module):
    """
    Skip-gram model with negative sampling for efficient training.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 num_negative_samples: int = 5, sparse: bool = False):
        super(SkipGramNegativeSampling, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_negative_samples = num_negative_samples
        
        # Skip-gram model
        self.skipgram = SkipGramModel(vocab_size, embedding_dim, sparse=sparse)
        
        # Negative sampling table
        self.negative_sampling_table = None
        
    def forward(self, target_words: torch.Tensor, context_words: torch.Tensor, 
                negative_words: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with negative sampling.
        
        Args:
            target_words: Target word indices (batch_size,)
            context_words: Context word indices (batch_size,)
            negative_words: Negative word indices (batch_size, num_negative_samples)
            
        Returns:
            positive_loss: Loss for positive samples
            negative_loss: Loss for negative samples
        """
        # Get target embeddings
        target_embeddings = self.skipgram(target_words)  # (batch_size, embedding_dim)
        
        # Positive samples
        context_embeddings = self.skipgram.output_embeddings(context_words)  # (batch_size, embedding_dim)
        positive_scores = torch.sum(target_embeddings * context_embeddings, dim=1)  # (batch_size,)
        positive_loss = -F.logsigmoid(positive_scores).mean()
        
        # Negative samples
        negative_embeddings = self.skipgram.output_embeddings(negative_words)  # (batch_size, num_neg, embedding_dim)
        
        # Expand target embeddings for negative sampling
        target_expanded = target_embeddings.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        negative_scores = torch.sum(target_expanded * negative_embeddings, dim=2)  # (batch_size, num_neg)
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


class SkipGramHierarchicalSoftmax(nn.Module):
    """
    Skip-gram model with hierarchical softmax for efficient training.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 huffman_tree, sparse: bool = False):
        super(SkipGramHierarchicalSoftmax, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.huffman_tree = huffman_tree
        
        # Skip-gram model
        self.skipgram = SkipGramModel(vocab_size, embedding_dim, sparse=sparse)
        
        # Internal node embeddings for hierarchical softmax
        num_internal_nodes = len(huffman_tree.internal_nodes)
        self.internal_embeddings = nn.Embedding(num_internal_nodes, embedding_dim, sparse=sparse)
        
        # Initialize internal embeddings
        nn.init.zeros_(self.internal_embeddings.weight)
        
        # Get all paths for efficient computation
        self.word_paths = huffman_tree.get_all_paths()
    
    def forward(self, target_words: torch.Tensor, context_words: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with hierarchical softmax.
        
        Args:
            target_words: Target word indices (batch_size,)
            context_words: Context word indices (batch_size,)
            
        Returns:
            loss: Hierarchical softmax loss
        """
        # Get target embeddings
        target_embeddings = self.skipgram(target_words)  # (batch_size, embedding_dim)
        
        batch_size = target_embeddings.size(0)
        total_loss = 0.0
        
        # Process each sample in the batch
        for i in range(batch_size):
            context_word = context_words[i].item()
            
            if context_word not in self.word_paths:
                continue
                
            code, path = self.word_paths[context_word]
            
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
                score = torch.dot(target_embeddings[i], internal_embedding)
                
                # Get target (code bit)
                target_bit = code[j]
                
                # Compute binary cross-entropy loss
                if target_bit == 1:
                    sample_loss += -F.logsigmoid(score)
                else:
                    sample_loss += -F.logsigmoid(-score)
            
            total_loss += sample_loss
        
        return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0)


class SkipGramTrainer:
    """
    Trainer class for Skip-gram model.
    """
    
    def __init__(self, model: Union[SkipGramNegativeSampling, SkipGramHierarchicalSoftmax], 
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
        
    def train_step(self, target_words: torch.Tensor, context_words: torch.Tensor, 
                   negative_words: torch.Tensor = None) -> float:
        """
        Single training step.
        
        Args:
            target_words: Target word indices
            context_words: Context word indices
            negative_words: Negative word indices (for negative sampling)
            
        Returns:
            loss: Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move tensors to device
        target_words = target_words.to(self.device)
        context_words = context_words.to(self.device)
        
        if isinstance(self.model, SkipGramNegativeSampling):
            if negative_words is None:
                negative_words = self.model.get_negative_samples(target_words.size(0))
            negative_words = negative_words.to(self.device)
            
            positive_loss, negative_loss = self.model(target_words, context_words, negative_words)
            loss = positive_loss + negative_loss
        else:
            # Hierarchical softmax
            loss = self.model(target_words, context_words)
        
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
            if isinstance(self.model, (SkipGramNegativeSampling, SkipGramHierarchicalSoftmax)):
                embeddings = self.model.skipgram.get_input_embeddings()
            else:
                embeddings = self.model.get_input_embeddings()
        return embeddings.cpu().numpy()


def create_skipgram_data_loader(training_pairs: List[Tuple[int, int]], 
                               batch_size: int = 128, shuffle: bool = True):
    """
    Create data loader for Skip-gram training.
    
    Args:
        training_pairs: List of (target_word, context_word) pairs
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader for training
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    targets, contexts = zip(*training_pairs)
    
    target_tensor = torch.tensor(targets, dtype=torch.long)
    context_tensor = torch.tensor(contexts, dtype=torch.long)
    
    dataset = TensorDataset(target_tensor, context_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class SkipGramMultiContext(nn.Module):
    """
    Skip-gram model that handles multiple context words per target word.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 num_negative_samples: int = 5, sparse: bool = False):
        super(SkipGramMultiContext, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_negative_samples = num_negative_samples
        
        # Skip-gram model
        self.skipgram = SkipGramModel(vocab_size, embedding_dim, sparse=sparse)
        
        # Negative sampling table
        self.negative_sampling_table = None
        
    def forward(self, target_words: torch.Tensor, context_words_list: List[torch.Tensor], 
                negative_words: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multiple context words per target.
        
        Args:
            target_words: Target word indices (batch_size,)
            context_words_list: List of context word tensors for each target
            negative_words: Negative word indices (batch_size, num_negative_samples)
            
        Returns:
            positive_loss: Loss for positive samples
            negative_loss: Loss for negative samples
        """
        # Get target embeddings
        target_embeddings = self.skipgram(target_words)  # (batch_size, embedding_dim)
        
        batch_size = target_embeddings.size(0)
        total_positive_loss = 0.0
        total_negative_loss = 0.0
        
        # Process each target-context pair
        for i in range(batch_size):
            target_emb = target_embeddings[i:i+1]  # (1, embedding_dim)
            context_words = context_words_list[i]  # (num_context_words,)
            
            if len(context_words) == 0:
                continue
                
            # Positive samples
            context_embeddings = self.skipgram.output_embeddings(context_words)  # (num_context, embedding_dim)
            positive_scores = torch.sum(target_emb * context_embeddings, dim=1)  # (num_context,)
            positive_loss = -F.logsigmoid(positive_scores).mean()
            total_positive_loss += positive_loss
            
            # Negative samples
            neg_words = negative_words[i:i+1]  # (1, num_negative_samples)
            negative_embeddings = self.skipgram.output_embeddings(neg_words)  # (1, num_neg, embedding_dim)
            negative_scores = torch.sum(target_emb.unsqueeze(1) * negative_embeddings, dim=2)  # (1, num_neg)
            negative_loss = -F.logsigmoid(-negative_scores).mean()
            total_negative_loss += negative_loss
        
        return total_positive_loss / batch_size, total_negative_loss / batch_size
    
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


def create_skipgram_batches(training_pairs: List[Tuple[int, int]], 
                           batch_size: int = 128, shuffle: bool = True):
    """
    Create batches for Skip-gram training with multiple context words per target.
    
    Args:
        training_pairs: List of (target_word, context_word) pairs
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        List of batches
    """
    if shuffle:
        random.shuffle(training_pairs)
    
    # Group by target word
    target_to_contexts = {}
    for target, context in training_pairs:
        if target not in target_to_contexts:
            target_to_contexts[target] = []
        target_to_contexts[target].append(context)
    
    # Create batches
    batches = []
    current_batch_targets = []
    current_batch_contexts = []
    
    for target, contexts in target_to_contexts.items():
        current_batch_targets.append(target)
        current_batch_contexts.append(contexts)
        
        if len(current_batch_targets) >= batch_size:
            batches.append((current_batch_targets, current_batch_contexts))
            current_batch_targets = []
            current_batch_contexts = []
    
    # Add remaining items
    if current_batch_targets:
        batches.append((current_batch_targets, current_batch_contexts))
    
    return batches


if __name__ == "__main__":
    # Test Skip-gram implementation
    print("Testing Skip-gram implementation...")
    
    # Parameters
    vocab_size = 1000
    embedding_dim = 100
    window_size = 5
    batch_size = 32
    
    # Create sample data
    sample_pairs = []
    for _ in range(1000):
        target_word = random.randint(0, vocab_size-1)
        context_word = random.randint(0, vocab_size-1)
        sample_pairs.append((target_word, context_word))
    
    # Test Skip-gram with negative sampling
    print("\nTesting Skip-gram with negative sampling...")
    model_ns = SkipGramNegativeSampling(vocab_size, embedding_dim, num_negative_samples=5)
    trainer_ns = SkipGramTrainer(model_ns, learning_rate=0.025)
    
    # Create data loader
    data_loader = create_skipgram_data_loader(sample_pairs, batch_size=batch_size)
    
    # Training loop
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (target_words, context_words) in enumerate(data_loader):
        loss = trainer_ns.train_step(target_words, context_words)
        total_loss += loss
        num_batches += 1
        
        if batch_idx >= 5:  # Test only a few batches
            break
    
    print(f"Average loss: {total_loss / num_batches:.4f}")
    
    # Get embeddings
    embeddings = trainer_ns.get_embeddings()
    print(f"Embedding shape: {embeddings.shape}")
    
    # Test multi-context version
    print("\nTesting Skip-gram with multi-context...")
    model_mc = SkipGramMultiContext(vocab_size, embedding_dim, num_negative_samples=5)
    
    # Create batches
    batches = create_skipgram_batches(sample_pairs, batch_size=16)
    
    # Test one batch
    if batches:
        targets, contexts_list = batches[0]
        target_tensor = torch.tensor(targets, dtype=torch.long)
        context_tensors = [torch.tensor(contexts, dtype=torch.long) for contexts in contexts_list]
        negative_tensor = model_mc.get_negative_samples(len(targets))
        
        pos_loss, neg_loss = model_mc(target_tensor, context_tensors, negative_tensor)
        print(f"Multi-context loss: positive={pos_loss:.4f}, negative={neg_loss:.4f}")
    
    print("\nSkip-gram model implementation completed successfully!") 