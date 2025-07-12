import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from collections import Counter


class HuffmanNode:
    """Node in the Huffman tree for hierarchical softmax."""
    
    def __init__(self, word_id: Optional[int] = None, freq: int = 0, 
                 left: 'HuffmanNode' = None, right: 'HuffmanNode' = None):
        self.word_id = word_id  # Only leaf nodes have word_id
        self.freq = freq
        self.left = left
        self.right = right
        
        # For hierarchical softmax
        self.code = []  # Huffman code (path from root to this node)
        self.path = []  # Internal nodes along the path
        self.point = []  # Point indices for internal nodes
        
    def __lt__(self, other):
        """For priority queue comparison."""
        return self.freq < other.freq
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.word_id is not None


class HuffmanTree:
    """Huffman tree for hierarchical softmax in Word2Vec."""
    
    def __init__(self, word_freq: Counter):
        self.word_freq = word_freq
        self.root = None
        self.word_nodes = {}  # word_id -> HuffmanNode
        self.internal_nodes = []  # List of internal nodes
        self.vocab_size = len(word_freq)
        
        # Build the tree
        self._build_tree()
        self._assign_codes()
    
    def _build_tree(self) -> None:
        """Build the Huffman tree from word frequencies."""
        if not self.word_freq:
            return
            
        # Create a priority queue with leaf nodes
        heap = []
        
        # Create leaf nodes for each word
        for word_id, freq in enumerate(self.word_freq.values()):
            node = HuffmanNode(word_id=word_id, freq=freq)
            self.word_nodes[word_id] = node
            heapq.heappush(heap, node)
        
        # Build tree bottom-up
        internal_node_id = self.vocab_size  # Start internal node IDs after vocabulary
        
        while len(heap) > 1:
            # Take two nodes with smallest frequencies
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            # Create internal node
            merged_freq = left.freq + right.freq
            internal_node = HuffmanNode(freq=merged_freq, left=left, right=right)
            
            # Store internal node with unique ID
            internal_node.word_id = internal_node_id
            self.internal_nodes.append(internal_node)
            internal_node_id += 1
            
            # Add back to heap
            heapq.heappush(heap, internal_node)
        
        # The remaining node is the root
        if heap:
            self.root = heap[0]
    
    def _assign_codes(self) -> None:
        """Assign Huffman codes and paths to all words."""
        if not self.root:
            return
            
        # Traverse tree to assign codes
        self._assign_codes_recursive(self.root, [], [])
    
    def _assign_codes_recursive(self, node: HuffmanNode, 
                               code: List[int], path: List[int]) -> None:
        """Recursively assign codes and paths."""
        if node.is_leaf():
            # Leaf node: assign code and path
            node.code = code.copy()
            node.path = path.copy()
            node.point = [n - self.vocab_size for n in path]  # Convert to 0-based internal node indices
        else:
            # Internal node: continue traversal
            if node.left:
                self._assign_codes_recursive(node.left, code + [0], path + [node.word_id])
            if node.right:
                self._assign_codes_recursive(node.right, code + [1], path + [node.word_id])
    
    def get_word_path(self, word_id: int) -> Tuple[List[int], List[int]]:
        """Get Huffman code and path for a word."""
        if word_id in self.word_nodes:
            node = self.word_nodes[word_id]
            return node.code, node.point
        return [], []
    
    def get_all_paths(self) -> Dict[int, Tuple[List[int], List[int]]]:
        """Get all word paths for efficient batch processing."""
        paths = {}
        for word_id, node in self.word_nodes.items():
            paths[word_id] = (node.code, node.point)
        return paths
    
    def print_tree_info(self) -> None:
        """Print information about the Huffman tree."""
        if not self.root:
            print("Empty tree")
            return
            
        total_nodes = len(self.word_nodes) + len(self.internal_nodes)
        print(f"Huffman Tree Info:")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Internal nodes: {len(self.internal_nodes)}")
        print(f"  Tree depth: {self._get_tree_depth()}")
        
        # Print some sample codes
        print("\nSample Huffman codes:")
        word_items = list(self.word_freq.items())
        for i, (word, freq) in enumerate(word_items[:5]):
            if i < len(self.word_nodes):
                node = self.word_nodes[i]
                print(f"  Word {i} (freq: {freq}): code = {node.code}, path = {node.point}")
    
    def _get_tree_depth(self) -> int:
        """Get the maximum depth of the tree."""
        if not self.root:
            return 0
        return self._get_depth_recursive(self.root)
    
    def _get_depth_recursive(self, node: HuffmanNode) -> int:
        """Recursively calculate tree depth."""
        if node.is_leaf():
            return 0
        
        left_depth = self._get_depth_recursive(node.left) if node.left else 0
        right_depth = self._get_depth_recursive(node.right) if node.right else 0
        
        return 1 + max(left_depth, right_depth)


class HierarchicalSoftmax:
    """Hierarchical softmax implementation using Huffman tree."""
    
    def __init__(self, huffman_tree: HuffmanTree, embedding_dim: int):
        self.huffman_tree = huffman_tree
        self.embedding_dim = embedding_dim
        self.vocab_size = huffman_tree.vocab_size
        self.num_internal_nodes = len(huffman_tree.internal_nodes)
        
        # Initialize parameters for internal nodes
        # Each internal node has a vector of size embedding_dim
        self.syn1 = np.random.uniform(-0.25, 0.25, 
                                     (self.num_internal_nodes, embedding_dim)).astype(np.float32)
        
        # Get all paths once for efficiency
        self.word_paths = huffman_tree.get_all_paths()
    
    def forward(self, input_vector: np.ndarray, target_word: int) -> Tuple[float, np.ndarray]:
        """
        Forward pass for hierarchical softmax.
        
        Args:
            input_vector: Input embedding vector
            target_word: Target word ID
            
        Returns:
            loss: Cross-entropy loss
            gradient: Gradient with respect to input vector
        """
        if target_word not in self.word_paths:
            return 0.0, np.zeros_like(input_vector)
        
        code, path = self.word_paths[target_word]
        
        if not code or not path:
            return 0.0, np.zeros_like(input_vector)
        
        loss = 0.0
        gradient = np.zeros_like(input_vector)
        
        # Traverse the path in the Huffman tree
        for i, point in enumerate(path):
            if point >= self.num_internal_nodes:
                continue
                
            # Get the vector for this internal node
            syn1_vector = self.syn1[point]
            
            # Compute dot product
            dot_product = np.dot(input_vector, syn1_vector)
            
            # Apply sigmoid
            sigmoid_val = self._sigmoid(dot_product)
            
            # Get the target (code bit)
            target = code[i]
            
            # Compute loss (binary cross-entropy)
            if target == 1:
                loss += -np.log(sigmoid_val + 1e-10)
                error = sigmoid_val - 1
            else:
                loss += -np.log(1 - sigmoid_val + 1e-10)
                error = sigmoid_val
            
            # Update gradients
            gradient += error * syn1_vector
            
            # Update internal node vector
            self.syn1[point] -= self.learning_rate * error * input_vector
        
        return loss, gradient
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function with numerical stability."""
        x = np.clip(x, -500, 500)  # Prevent overflow
        return 1.0 / (1.0 + np.exp(-x))
    
    def set_learning_rate(self, learning_rate: float) -> None:
        """Set the learning rate for parameter updates."""
        self.learning_rate = learning_rate
    
    def get_internal_vectors(self) -> np.ndarray:
        """Get internal node vectors for inspection."""
        return self.syn1.copy()
    
    def save_vectors(self, filename: str) -> None:
        """Save internal node vectors to file."""
        np.save(filename, self.syn1)
        print(f"Internal vectors saved to {filename}")
    
    def load_vectors(self, filename: str) -> None:
        """Load internal node vectors from file."""
        self.syn1 = np.load(filename)
        print(f"Internal vectors loaded from {filename}")


def create_huffman_tree_from_vocabulary(vocabulary) -> HuffmanTree:
    """Create Huffman tree from vocabulary object."""
    return HuffmanTree(vocabulary.word_freq)


if __name__ == "__main__":
    # Test the Huffman tree implementation
    from collections import Counter
    
    # Create sample word frequencies
    word_freq = Counter({
        'the': 100,
        'of': 80,
        'and': 70,
        'to': 60,
        'a': 50,
        'in': 40,
        'is': 30,
        'it': 25,
        'you': 20,
        'that': 15,
        'he': 12,
        'was': 10,
        'for': 8,
        'on': 6,
        'are': 5
    })
    
    print("Creating Huffman tree...")
    huffman_tree = HuffmanTree(word_freq)
    huffman_tree.print_tree_info()
    
    # Test hierarchical softmax
    print("\nTesting hierarchical softmax...")
    hs = HierarchicalSoftmax(huffman_tree, embedding_dim=100)
    hs.set_learning_rate(0.01)
    
    # Test with random input
    input_vector = np.random.randn(100).astype(np.float32)
    target_word = 0  # 'the'
    
    loss, gradient = hs.forward(input_vector, target_word)
    print(f"Loss: {loss:.4f}")
    print(f"Gradient norm: {np.linalg.norm(gradient):.4f}")
    
    print("\nHuffman tree implementation completed successfully!") 