import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
import random
from collections import Counter


class NegativeSampler:
    """
    Efficient negative sampling implementation for Word2Vec training.
    """
    
    def __init__(self, vocabulary_freq: Counter, power: float = 0.75, 
                 table_size: int = 10**8):
        self.vocabulary_freq = vocabulary_freq
        self.power = power
        self.table_size = table_size
        
        # Build sampling table
        self.sampling_table = self._build_sampling_table()
        
        # Store vocabulary information
        self.vocab_size = len(vocabulary_freq)
        self.word_freq = vocabulary_freq
        
    def _build_sampling_table(self) -> np.ndarray:
        """
        Build negative sampling table based on word frequencies.
        Uses the unigram distribution raised to the power of 3/4.
        """
        # Calculate probabilities (word_freq^power)
        total_pow = sum(freq ** self.power for freq in self.vocabulary_freq.values())
        
        # Create table
        table = np.zeros(self.table_size, dtype=np.int32)
        
        # Fill table
        word_ids = list(range(len(self.vocabulary_freq)))
        word_probs = [(freq ** self.power) / total_pow 
                     for freq in self.vocabulary_freq.values()]
        
        # Create cumulative distribution
        cumulative = np.cumsum(word_probs)
        
        # Fill sampling table
        j = 0
        for i in range(self.table_size):
            while i / self.table_size > cumulative[j]:
                j += 1
                if j >= len(cumulative):
                    j = len(cumulative) - 1
                    break
            table[i] = word_ids[j]
        
        return table
    
    def sample_negative_words(self, num_samples: int, 
                             exclude_words: Optional[List[int]] = None) -> List[int]:
        """
        Sample negative words from the distribution.
        
        Args:
            num_samples: Number of negative samples to generate
            exclude_words: Words to exclude from sampling
            
        Returns:
            List of negative word IDs
        """
        if exclude_words is None:
            exclude_words = []
        
        exclude_set = set(exclude_words)
        negative_samples = []
        
        max_attempts = num_samples * 10  # Prevent infinite loops
        attempts = 0
        
        while len(negative_samples) < num_samples and attempts < max_attempts:
            # Random index in the table
            idx = random.randint(0, self.table_size - 1)
            word_id = self.sampling_table[idx]
            
            if word_id not in exclude_set and word_id not in negative_samples:
                negative_samples.append(word_id)
            
            attempts += 1
        
        # If we couldn't get enough samples, fill with random words
        while len(negative_samples) < num_samples:
            word_id = random.randint(0, self.vocab_size - 1)
            if word_id not in exclude_set and word_id not in negative_samples:
                negative_samples.append(word_id)
        
        return negative_samples
    
    def batch_sample_negative_words(self, batch_size: int, num_samples: int,
                                   exclude_words: Optional[List[List[int]]] = None) -> List[List[int]]:
        """
        Sample negative words for a batch.
        
        Args:
            batch_size: Size of the batch
            num_samples: Number of negative samples per example
            exclude_words: Words to exclude for each example in the batch
            
        Returns:
            List of negative word lists for each example
        """
        if exclude_words is None:
            exclude_words = [[] for _ in range(batch_size)]
        
        batch_negatives = []
        for i in range(batch_size):
            negatives = self.sample_negative_words(num_samples, exclude_words[i])
            batch_negatives.append(negatives)
        
        return batch_negatives
    
    def get_word_probability(self, word_id: int) -> float:
        """Get the probability of a word being sampled."""
        if word_id >= len(self.vocabulary_freq):
            return 0.0
        
        word_freq = list(self.vocabulary_freq.values())[word_id]
        total_pow = sum(freq ** self.power for freq in self.vocabulary_freq.values())
        
        return (word_freq ** self.power) / total_pow


class NegativeSamplingTrainer:
    """
    Trainer for negative sampling based Word2Vec.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 negative_sampler: NegativeSampler, num_negative_samples: int = 5,
                 learning_rate: float = 0.025, device: str = 'cpu'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.negative_sampler = negative_sampler
        self.num_negative_samples = num_negative_samples
        self.learning_rate = learning_rate
        self.device = device
        
        # Initialize embedding matrices
        self.input_embeddings = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, 
                                                 (vocab_size, embedding_dim)).astype(np.float32)
        self.output_embeddings = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation with numerical stability."""
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def train_pair(self, target_word: int, context_word: int) -> float:
        """
        Train on a single target-context pair using negative sampling.
        
        Args:
            target_word: Target word ID
            context_word: Context word ID
            
        Returns:
            Loss for this pair
        """
        # Get embeddings
        target_embedding = self.input_embeddings[target_word]
        context_embedding = self.output_embeddings[context_word]
        
        # Positive sample
        positive_score = np.dot(target_embedding, context_embedding)
        positive_sigmoid = self.sigmoid(positive_score)
        positive_loss = -np.log(positive_sigmoid + 1e-10)
        
        # Gradient for positive sample
        positive_error = positive_sigmoid - 1.0
        target_gradient = positive_error * context_embedding
        context_gradient = positive_error * target_embedding
        
        # Update context embedding
        self.output_embeddings[context_word] -= self.learning_rate * context_gradient
        
        # Negative samples
        negative_words = self.negative_sampler.sample_negative_words(
            self.num_negative_samples, [target_word, context_word])
        
        negative_loss = 0.0
        
        for neg_word in negative_words:
            neg_embedding = self.output_embeddings[neg_word]
            neg_score = np.dot(target_embedding, neg_embedding)
            neg_sigmoid = self.sigmoid(neg_score)
            
            # Loss for negative sample
            negative_loss += -np.log(1.0 - neg_sigmoid + 1e-10)
            
            # Gradient for negative sample
            neg_error = neg_sigmoid
            target_gradient += neg_error * neg_embedding
            
            # Update negative word embedding
            self.output_embeddings[neg_word] -= self.learning_rate * neg_error * target_embedding
        
        # Update target embedding
        self.input_embeddings[target_word] -= self.learning_rate * target_gradient
        
        return positive_loss + negative_loss
    
    def train_batch(self, target_words: List[int], context_words: List[int]) -> float:
        """
        Train on a batch of target-context pairs.
        
        Args:
            target_words: List of target word IDs
            context_words: List of context word IDs
            
        Returns:
            Average loss for the batch
        """
        total_loss = 0.0
        batch_size = len(target_words)
        
        for i in range(batch_size):
            loss = self.train_pair(target_words[i], context_words[i])
            total_loss += loss
        
        return total_loss / batch_size
    
    def update_learning_rate(self, progress: float):
        """Update learning rate based on training progress."""
        self.learning_rate = max(0.025 * (1 - progress), 0.0001)
    
    def get_embeddings(self) -> np.ndarray:
        """Get input embeddings."""
        return self.input_embeddings.copy()
    
    def get_output_embeddings(self) -> np.ndarray:
        """Get output embeddings."""
        return self.output_embeddings.copy()
    
    def save_embeddings(self, input_file: str, output_file: str):
        """Save embeddings to files."""
        np.save(input_file, self.input_embeddings)
        np.save(output_file, self.output_embeddings)
        print(f"Embeddings saved to {input_file} and {output_file}")
    
    def load_embeddings(self, input_file: str, output_file: str):
        """Load embeddings from files."""
        self.input_embeddings = np.load(input_file)
        self.output_embeddings = np.load(output_file)
        print(f"Embeddings loaded from {input_file} and {output_file}")


class NegativeSamplingPyTorch(nn.Module):
    """
    PyTorch implementation of negative sampling for GPU acceleration.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 num_negative_samples: int = 5, sparse: bool = False):
        super(NegativeSamplingPyTorch, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_negative_samples = num_negative_samples
        
        # Embedding layers
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=sparse)
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Negative sampling table (will be set externally)
        self.negative_sampling_table = None
        
    def _init_embeddings(self):
        """Initialize embedding weights."""
        nn.init.uniform_(self.input_embeddings.weight, -0.5/self.embedding_dim, 0.5/self.embedding_dim)
        nn.init.zeros_(self.output_embeddings.weight)
    
    def forward(self, target_words: torch.Tensor, context_words: torch.Tensor,
                negative_words: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for negative sampling.
        
        Args:
            target_words: Target word IDs (batch_size,)
            context_words: Context word IDs (batch_size,)
            negative_words: Negative word IDs (batch_size, num_negative_samples)
            
        Returns:
            positive_loss: Loss for positive samples
            negative_loss: Loss for negative samples
        """
        # Get embeddings
        target_embeds = self.input_embeddings(target_words)  # (batch_size, embedding_dim)
        context_embeds = self.output_embeddings(context_words)  # (batch_size, embedding_dim)
        
        # Positive samples
        positive_scores = torch.sum(target_embeds * context_embeds, dim=1)  # (batch_size,)
        positive_loss = -F.logsigmoid(positive_scores).mean()
        
        # Negative samples
        negative_embeds = self.output_embeddings(negative_words)  # (batch_size, num_neg, embedding_dim)
        
        # Expand target embeddings for broadcasting
        target_expanded = target_embeds.unsqueeze(1)  # (batch_size, 1, embedding_dim)
        negative_scores = torch.sum(target_expanded * negative_embeds, dim=2)  # (batch_size, num_neg)
        negative_loss = -F.logsigmoid(-negative_scores).mean()
        
        return positive_loss, negative_loss
    
    def set_negative_sampling_table(self, table: np.ndarray):
        """Set the negative sampling table."""
        self.negative_sampling_table = table
    
    def sample_negative_words(self, batch_size: int, exclude_words: Optional[List[int]] = None) -> torch.Tensor:
        """Sample negative words for a batch."""
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


class AdvancedNegativeSampler:
    """
    Advanced negative sampling with additional optimizations.
    """
    
    def __init__(self, vocabulary_freq: Counter, power: float = 0.75, 
                 table_size: int = 10**8, subsample_threshold: float = 1e-3):
        self.vocabulary_freq = vocabulary_freq
        self.power = power
        self.table_size = table_size
        self.subsample_threshold = subsample_threshold
        
        # Build sampling table
        self.sampling_table = self._build_sampling_table()
        
        # Calculate subsampling probabilities
        self.subsampling_probs = self._calculate_subsampling_probs()
        
        # Store vocabulary information
        self.vocab_size = len(vocabulary_freq)
        self.word_freq = vocabulary_freq
        
    def _build_sampling_table(self) -> np.ndarray:
        """Build negative sampling table with improved efficiency."""
        # Calculate probabilities
        word_probs = {}
        total_pow = 0
        
        for word_id, freq in enumerate(self.vocabulary_freq.values()):
            prob = freq ** self.power
            word_probs[word_id] = prob
            total_pow += prob
        
        # Normalize probabilities
        for word_id in word_probs:
            word_probs[word_id] /= total_pow
        
        # Create table using alias method for faster sampling
        return self._create_alias_table(word_probs)
    
    def _create_alias_table(self, word_probs: Dict[int, float]) -> np.ndarray:
        """Create alias table for O(1) sampling."""
        n = len(word_probs)
        prob_table = np.zeros(n, dtype=np.float32)
        alias_table = np.zeros(n, dtype=np.int32)
        
        # Scale probabilities
        scaled_probs = [prob * n for prob in word_probs.values()]
        word_ids = list(word_probs.keys())
        
        # Separate small and large probabilities
        small = []
        large = []
        
        for i, prob in enumerate(scaled_probs):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)
        
        # Fill alias table
        while small and large:
            small_idx = small.pop()
            large_idx = large.pop()
            
            prob_table[small_idx] = scaled_probs[small_idx]
            alias_table[small_idx] = large_idx
            
            scaled_probs[large_idx] = scaled_probs[large_idx] + scaled_probs[small_idx] - 1.0
            
            if scaled_probs[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)
        
        # Fill remaining
        while large:
            large_idx = large.pop()
            prob_table[large_idx] = 1.0
        
        while small:
            small_idx = small.pop()
            prob_table[small_idx] = 1.0
        
        # Store tables for sampling
        self.prob_table = prob_table
        self.alias_table = alias_table
        self.word_ids = word_ids
        
        return np.arange(n)  # Return simple array as table reference
    
    def _calculate_subsampling_probs(self) -> Dict[int, float]:
        """Calculate subsampling probabilities for frequent words."""
        total_words = sum(self.vocabulary_freq.values())
        subsampling_probs = {}
        
        for word_id, freq in enumerate(self.vocabulary_freq.values()):
            word_prob = freq / total_words
            if word_prob > self.subsample_threshold:
                keep_prob = (np.sqrt(word_prob / self.subsample_threshold) + 1) * (self.subsample_threshold / word_prob)
                subsampling_probs[word_id] = min(keep_prob, 1.0)
            else:
                subsampling_probs[word_id] = 1.0
        
        return subsampling_probs
    
    def sample_negative_words_fast(self, num_samples: int) -> List[int]:
        """Fast negative sampling using alias method."""
        negative_samples = []
        
        for _ in range(num_samples):
            # Sample using alias table
            idx = random.randint(0, len(self.word_ids) - 1)
            
            if random.random() < self.prob_table[idx]:
                word_id = self.word_ids[idx]
            else:
                word_id = self.word_ids[self.alias_table[idx]]
            
            negative_samples.append(word_id)
        
        return negative_samples
    
    def should_subsample(self, word_id: int) -> bool:
        """Determine if a word should be subsampled."""
        if word_id in self.subsampling_probs:
            return random.random() > self.subsampling_probs[word_id]
        return False


def create_negative_sampler(vocabulary_freq: Counter, 
                          implementation: str = 'basic',
                          **kwargs) -> Union[NegativeSampler, AdvancedNegativeSampler]:
    """
    Factory function to create negative sampler.
    
    Args:
        vocabulary_freq: Word frequency counter
        implementation: Implementation type ('basic' or 'advanced')
        **kwargs: Additional arguments for the sampler
        
    Returns:
        Negative sampler instance
    """
    if implementation == 'basic':
        return NegativeSampler(vocabulary_freq, **kwargs)
    elif implementation == 'advanced':
        return AdvancedNegativeSampler(vocabulary_freq, **kwargs)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")


if __name__ == "__main__":
    # Test negative sampling implementations
    print("Testing negative sampling implementations...")
    
    # Create sample vocabulary
    vocab_freq = Counter({
        'the': 100, 'of': 80, 'and': 70, 'to': 60, 'a': 50,
        'in': 40, 'is': 30, 'it': 25, 'you': 20, 'that': 15,
        'he': 12, 'was': 10, 'for': 8, 'on': 6, 'are': 5
    })
    
    # Test basic negative sampler
    print("\nTesting basic negative sampler...")
    sampler = NegativeSampler(vocab_freq, power=0.75)
    
    # Sample negative words
    negative_samples = sampler.sample_negative_words(5)
    print(f"Negative samples: {negative_samples}")
    
    # Test batch sampling
    batch_negatives = sampler.batch_sample_negative_words(3, 5)
    print(f"Batch negative samples: {batch_negatives}")
    
    # Test negative sampling trainer
    print("\nTesting negative sampling trainer...")
    trainer = NegativeSamplingTrainer(len(vocab_freq), 100, sampler, num_negative_samples=5)
    
    # Train on sample pairs
    target_words = [0, 1, 2, 3, 4]
    context_words = [5, 6, 7, 8, 9]
    
    loss = trainer.train_batch(target_words, context_words)
    print(f"Training loss: {loss:.4f}")
    
    # Test advanced sampler
    print("\nTesting advanced negative sampler...")
    advanced_sampler = AdvancedNegativeSampler(vocab_freq, power=0.75)
    
    # Test fast sampling
    fast_samples = advanced_sampler.sample_negative_words_fast(10)
    print(f"Fast negative samples: {fast_samples}")
    
    # Test subsampling
    subsample_decisions = [advanced_sampler.should_subsample(i) for i in range(5)]
    print(f"Subsampling decisions: {subsample_decisions}")
    
    print("\nNegative sampling implementation completed successfully!") 