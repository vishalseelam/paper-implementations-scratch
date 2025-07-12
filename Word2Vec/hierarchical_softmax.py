import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union
from huffman_tree import HuffmanTree, HuffmanNode


class HierarchicalSoftmaxTrainer:
    """
    Efficient hierarchical softmax trainer for Word2Vec models.
    """
    
    def __init__(self, huffman_tree: HuffmanTree, embedding_dim: int, 
                 learning_rate: float = 0.025, device: str = 'cpu'):
        self.huffman_tree = huffman_tree
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.device = device
        
        # Internal node embeddings
        self.num_internal_nodes = len(huffman_tree.internal_nodes)
        self.internal_embeddings = np.random.uniform(-0.25, 0.25, 
                                                    (self.num_internal_nodes, embedding_dim)).astype(np.float32)
        
        # Get all paths for efficient computation
        self.word_paths = huffman_tree.get_all_paths()
        
        # Precompute paths for batch processing
        self._precompute_paths()
    
    def _precompute_paths(self):
        """Precompute and organize paths for efficient batch processing."""
        self.max_path_length = 0
        self.word_codes = {}
        self.word_points = {}
        
        for word_id, (code, path) in self.word_paths.items():
            self.word_codes[word_id] = np.array(code, dtype=np.int32)
            self.word_points[word_id] = np.array(path, dtype=np.int32)
            self.max_path_length = max(self.max_path_length, len(code))
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation with numerical stability."""
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def compute_loss_and_gradients(self, input_vector: np.ndarray, 
                                  target_word: int) -> Tuple[float, np.ndarray]:
        """
        Compute hierarchical softmax loss and gradients.
        
        Args:
            input_vector: Input embedding vector
            target_word: Target word ID
            
        Returns:
            loss: Cross-entropy loss
            input_gradient: Gradient with respect to input vector
        """
        if target_word not in self.word_paths:
            return 0.0, np.zeros_like(input_vector)
        
        codes = self.word_codes[target_word]
        points = self.word_points[target_word]
        
        if len(codes) == 0 or len(points) == 0:
            return 0.0, np.zeros_like(input_vector)
        
        loss = 0.0
        input_gradient = np.zeros_like(input_vector)
        
        # Traverse the path in the Huffman tree
        for i, point in enumerate(points):
            if point >= self.num_internal_nodes:
                continue
            
            # Get internal node embedding
            internal_vector = self.internal_embeddings[point]
            
            # Compute dot product
            dot_product = np.dot(input_vector, internal_vector)
            
            # Apply sigmoid
            sigmoid_val = self.sigmoid(dot_product)
            
            # Get target (code bit)
            target_bit = codes[i]
            
            # Compute loss and error
            if target_bit == 1:
                loss += -np.log(sigmoid_val + 1e-10)
                error = sigmoid_val - 1.0
            else:
                loss += -np.log(1.0 - sigmoid_val + 1e-10)
                error = sigmoid_val
            
            # Update gradients
            input_gradient += error * internal_vector
            
            # Update internal node embedding
            self.internal_embeddings[point] -= self.learning_rate * error * input_vector
        
        return loss, input_gradient
    
    def batch_compute_loss_and_gradients(self, input_vectors: np.ndarray, 
                                        target_words: List[int]) -> Tuple[float, np.ndarray]:
        """
        Batch computation of hierarchical softmax loss and gradients.
        
        Args:
            input_vectors: Batch of input vectors (batch_size, embedding_dim)
            target_words: List of target word IDs
            
        Returns:
            total_loss: Total loss for the batch
            input_gradients: Gradients for input vectors
        """
        batch_size = len(target_words)
        total_loss = 0.0
        input_gradients = np.zeros_like(input_vectors)
        
        for i in range(batch_size):
            loss, gradient = self.compute_loss_and_gradients(input_vectors[i], target_words[i])
            total_loss += loss
            input_gradients[i] = gradient
        
        return total_loss, input_gradients
    
    def update_learning_rate(self, progress: float):
        """Update learning rate based on training progress."""
        self.learning_rate = max(0.025 * (1 - progress), 0.0001)
    
    def save_internal_embeddings(self, filename: str):
        """Save internal node embeddings."""
        np.save(filename, self.internal_embeddings)
        print(f"Internal embeddings saved to {filename}")
    
    def load_internal_embeddings(self, filename: str):
        """Load internal node embeddings."""
        self.internal_embeddings = np.load(filename)
        print(f"Internal embeddings loaded from {filename}")


class HierarchicalSoftmaxPyTorch(nn.Module):
    """
    PyTorch implementation of hierarchical softmax for better GPU utilization.
    """
    
    def __init__(self, huffman_tree: HuffmanTree, embedding_dim: int):
        super(HierarchicalSoftmaxPyTorch, self).__init__()
        
        self.huffman_tree = huffman_tree
        self.embedding_dim = embedding_dim
        self.num_internal_nodes = len(huffman_tree.internal_nodes)
        
        # Internal node embeddings
        self.internal_embeddings = nn.Parameter(
            torch.randn(self.num_internal_nodes, embedding_dim) * 0.1
        )
        
        # Precompute paths
        self.word_paths = huffman_tree.get_all_paths()
        self._create_path_tensors()
    
    def _create_path_tensors(self):
        """Create tensors for efficient path processing."""
        self.max_path_length = max(len(code) for code, _ in self.word_paths.values()) if self.word_paths else 0
        
        # Create padded tensors for codes and points
        self.codes_tensor = {}
        self.points_tensor = {}
        self.path_lengths = {}
        
        for word_id, (code, path) in self.word_paths.items():
            # Pad codes and paths to max length
            padded_code = code + [0] * (self.max_path_length - len(code))
            padded_path = path + [0] * (self.max_path_length - len(path))
            
            self.codes_tensor[word_id] = torch.tensor(padded_code, dtype=torch.float32)
            self.points_tensor[word_id] = torch.tensor(padded_path, dtype=torch.long)
            self.path_lengths[word_id] = len(code)
    
    def forward(self, input_vectors: torch.Tensor, target_words: List[int]) -> torch.Tensor:
        """
        Forward pass for hierarchical softmax.
        
        Args:
            input_vectors: Input embedding vectors (batch_size, embedding_dim)
            target_words: List of target word IDs
            
        Returns:
            loss: Total hierarchical softmax loss
        """
        batch_size = input_vectors.size(0)
        total_loss = 0.0
        
        for i in range(batch_size):
            word_id = target_words[i]
            
            if word_id not in self.word_paths:
                continue
            
            # Get codes and paths for this word
            codes = self.codes_tensor[word_id].to(input_vectors.device)
            points = self.points_tensor[word_id].to(input_vectors.device)
            path_length = self.path_lengths[word_id]
            
            if path_length == 0:
                continue
            
            # Get relevant internal embeddings
            relevant_points = points[:path_length]
            relevant_codes = codes[:path_length]
            
            # Mask for valid internal nodes
            valid_mask = relevant_points < self.num_internal_nodes
            
            if not valid_mask.any():
                continue
            
            # Filter valid points and codes
            valid_points = relevant_points[valid_mask]
            valid_codes = relevant_codes[valid_mask]
            
            # Get internal embeddings for valid points
            internal_embeds = self.internal_embeddings[valid_points]  # (valid_points, embedding_dim)
            
            # Compute dot products
            input_vec = input_vectors[i]  # (embedding_dim,)
            dot_products = torch.sum(input_vec * internal_embeds, dim=1)  # (valid_points,)
            
            # Apply sigmoid
            sigmoid_vals = torch.sigmoid(dot_products)
            
            # Compute loss
            sample_loss = -(valid_codes * torch.log(sigmoid_vals + 1e-10) + 
                           (1 - valid_codes) * torch.log(1 - sigmoid_vals + 1e-10))
            
            total_loss += sample_loss.sum()
        
        return total_loss / batch_size if batch_size > 0 else torch.tensor(0.0)


class HierarchicalSoftmaxOptimized:
    """
    Optimized hierarchical softmax implementation with vectorized operations.
    """
    
    def __init__(self, huffman_tree: HuffmanTree, embedding_dim: int):
        self.huffman_tree = huffman_tree
        self.embedding_dim = embedding_dim
        self.num_internal_nodes = len(huffman_tree.internal_nodes)
        
        # Internal node embeddings
        self.internal_embeddings = np.random.uniform(-0.1, 0.1, 
                                                    (self.num_internal_nodes, embedding_dim)).astype(np.float32)
        
        # Precompute and organize paths for vectorized operations
        self.word_paths = huffman_tree.get_all_paths()
        self._prepare_vectorized_data()
    
    def _prepare_vectorized_data(self):
        """Prepare data structures for vectorized operations."""
        # Find maximum path length
        self.max_path_length = max(len(code) for code, _ in self.word_paths.values()) if self.word_paths else 0
        
        # Create arrays for all words
        vocab_size = len(self.word_paths)
        self.all_codes = np.zeros((vocab_size, self.max_path_length), dtype=np.int32)
        self.all_points = np.zeros((vocab_size, self.max_path_length), dtype=np.int32)
        self.path_lengths = np.zeros(vocab_size, dtype=np.int32)
        
        # Fill arrays
        for word_id, (code, path) in self.word_paths.items():
            path_len = len(code)
            self.path_lengths[word_id] = path_len
            
            if path_len > 0:
                self.all_codes[word_id, :path_len] = code
                self.all_points[word_id, :path_len] = path
    
    def compute_batch_loss_vectorized(self, input_vectors: np.ndarray, 
                                     target_words: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Vectorized computation of hierarchical softmax loss.
        
        Args:
            input_vectors: Input vectors (batch_size, embedding_dim)
            target_words: Target word IDs (batch_size,)
            
        Returns:
            total_loss: Total loss
            input_gradients: Gradients for input vectors
        """
        batch_size = len(target_words)
        total_loss = 0.0
        input_gradients = np.zeros_like(input_vectors)
        
        for i in range(batch_size):
            word_id = target_words[i]
            
            if word_id >= len(self.path_lengths):
                continue
                
            path_length = self.path_lengths[word_id]
            
            if path_length == 0:
                continue
            
            # Get codes and points for this word
            codes = self.all_codes[word_id, :path_length]
            points = self.all_points[word_id, :path_length]
            
            # Filter valid points
            valid_mask = points < self.num_internal_nodes
            
            if not valid_mask.any():
                continue
            
            valid_points = points[valid_mask]
            valid_codes = codes[valid_mask]
            
            # Get internal embeddings
            internal_embeds = self.internal_embeddings[valid_points]  # (valid_points, embedding_dim)
            
            # Compute dot products
            input_vec = input_vectors[i]  # (embedding_dim,)
            dot_products = np.sum(input_vec * internal_embeds, axis=1)  # (valid_points,)
            
            # Apply sigmoid
            sigmoid_vals = 1.0 / (1.0 + np.exp(-np.clip(dot_products, -500, 500)))
            
            # Compute loss
            eps = 1e-10
            sample_loss = -(valid_codes * np.log(sigmoid_vals + eps) + 
                           (1 - valid_codes) * np.log(1 - sigmoid_vals + eps))
            total_loss += np.sum(sample_loss)
            
            # Compute gradients
            errors = sigmoid_vals - valid_codes
            input_gradients[i] = np.sum(errors[:, np.newaxis] * internal_embeds, axis=0)
        
        return total_loss, input_gradients
    
    def update_internal_embeddings(self, input_vectors: np.ndarray, 
                                  target_words: np.ndarray, learning_rate: float):
        """
        Update internal node embeddings.
        
        Args:
            input_vectors: Input vectors (batch_size, embedding_dim)
            target_words: Target word IDs (batch_size,)
            learning_rate: Learning rate
        """
        batch_size = len(target_words)
        
        for i in range(batch_size):
            word_id = target_words[i]
            
            if word_id >= len(self.path_lengths):
                continue
                
            path_length = self.path_lengths[word_id]
            
            if path_length == 0:
                continue
            
            # Get codes and points for this word
            codes = self.all_codes[word_id, :path_length]
            points = self.all_points[word_id, :path_length]
            
            # Filter valid points
            valid_mask = points < self.num_internal_nodes
            
            if not valid_mask.any():
                continue
            
            valid_points = points[valid_mask]
            valid_codes = codes[valid_mask]
            
            # Get internal embeddings
            internal_embeds = self.internal_embeddings[valid_points]
            
            # Compute dot products and sigmoid
            input_vec = input_vectors[i]
            dot_products = np.sum(input_vec * internal_embeds, axis=1)
            sigmoid_vals = 1.0 / (1.0 + np.exp(-np.clip(dot_products, -500, 500)))
            
            # Compute errors
            errors = sigmoid_vals - valid_codes
            
            # Update internal embeddings
            for j, point in enumerate(valid_points):
                self.internal_embeddings[point] -= learning_rate * errors[j] * input_vec


def create_hierarchical_softmax_trainer(huffman_tree: HuffmanTree, 
                                       embedding_dim: int, 
                                       learning_rate: float = 0.025,
                                       implementation: str = 'numpy') -> Union[HierarchicalSoftmaxTrainer, 
                                                                            HierarchicalSoftmaxOptimized]:
    """
    Factory function to create hierarchical softmax trainer.
    
    Args:
        huffman_tree: Huffman tree for the vocabulary
        embedding_dim: Embedding dimension
        learning_rate: Learning rate
        implementation: Implementation type ('numpy' or 'optimized')
        
    Returns:
        Hierarchical softmax trainer instance
    """
    if implementation == 'numpy':
        return HierarchicalSoftmaxTrainer(huffman_tree, embedding_dim, learning_rate)
    elif implementation == 'optimized':
        return HierarchicalSoftmaxOptimized(huffman_tree, embedding_dim)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")


if __name__ == "__main__":
    # Test hierarchical softmax implementations
    from collections import Counter
    from huffman_tree import HuffmanTree
    
    # Create sample word frequencies
    word_freq = Counter({
        'the': 100, 'of': 80, 'and': 70, 'to': 60, 'a': 50,
        'in': 40, 'is': 30, 'it': 25, 'you': 20, 'that': 15
    })
    
    print("Testing hierarchical softmax implementations...")
    
    # Create Huffman tree
    huffman_tree = HuffmanTree(word_freq)
    
    # Test basic implementation
    print("\nTesting basic implementation...")
    hs_trainer = HierarchicalSoftmaxTrainer(huffman_tree, embedding_dim=100)
    
    # Test with random input
    input_vector = np.random.randn(100).astype(np.float32)
    target_word = 0
    
    loss, gradient = hs_trainer.compute_loss_and_gradients(input_vector, target_word)
    print(f"Loss: {loss:.4f}, Gradient norm: {np.linalg.norm(gradient):.4f}")
    
    # Test batch processing
    batch_size = 5
    input_vectors = np.random.randn(batch_size, 100).astype(np.float32)
    target_words = [0, 1, 2, 3, 4]
    
    total_loss, gradients = hs_trainer.batch_compute_loss_and_gradients(input_vectors, target_words)
    print(f"Batch loss: {total_loss:.4f}, Gradients shape: {gradients.shape}")
    
    # Test optimized implementation
    print("\nTesting optimized implementation...")
    hs_optimized = HierarchicalSoftmaxOptimized(huffman_tree, embedding_dim=100)
    
    target_words_array = np.array(target_words)
    loss_opt, gradients_opt = hs_optimized.compute_batch_loss_vectorized(input_vectors, target_words_array)
    print(f"Optimized loss: {loss_opt:.4f}, Gradients shape: {gradients_opt.shape}")
    
    print("\nHierarchical softmax implementation completed successfully!") 