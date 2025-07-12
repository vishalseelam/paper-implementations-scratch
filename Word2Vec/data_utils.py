import re
import math
import random
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional, Set


class Vocabulary:
    """Vocabulary class for Word2Vec implementation."""
    
    def __init__(self, min_count: int = 5, sample: float = 1e-3):
        self.min_count = min_count
        self.sample = sample  # Subsampling threshold
        
        self.word2id = {}
        self.id2word = {}
        self.word_freq = Counter()
        self.word_count = 0
        self.vocab_size = 0
        
        # Subsampling probabilities
        self.subsampling_probs = {}
        
    def build_vocabulary(self, sentences: List[List[str]]) -> None:
        """Build vocabulary from sentences."""
        print("Building vocabulary...")
        
        # Count word frequencies
        for sentence in sentences:
            for word in sentence:
                self.word_freq[word] += 1
                self.word_count += 1
        
        # Filter by minimum count
        filtered_words = {word: count for word, count in self.word_freq.items() 
                         if count >= self.min_count}
        
        # Create word-to-id mapping
        self.word2id = {word: i for i, word in enumerate(filtered_words.keys())}
        self.id2word = {i: word for word, i in self.word2id.items()}
        self.vocab_size = len(self.word2id)
        
        # Update word frequencies to only include vocabulary words
        self.word_freq = Counter(filtered_words)
        
        # Calculate subsampling probabilities
        self._calculate_subsampling_probs()
        
        print(f"Vocabulary built: {self.vocab_size} words from {self.word_count} total words")
        
    def _calculate_subsampling_probs(self) -> None:
        """Calculate subsampling probabilities for frequent words."""
        if self.sample <= 0:
            return
            
        for word, freq in self.word_freq.items():
            # Calculate probability of keeping the word
            word_prob = freq / self.word_count
            keep_prob = (math.sqrt(word_prob / self.sample) + 1) * (self.sample / word_prob)
            self.subsampling_probs[word] = min(keep_prob, 1.0)
    
    def subsample_word(self, word: str) -> bool:
        """Determine if a word should be kept based on subsampling."""
        if word not in self.subsampling_probs:
            return False
        return random.random() < self.subsampling_probs[word]
    
    def get_word_id(self, word: str) -> Optional[int]:
        """Get word ID, return None if not in vocabulary."""
        return self.word2id.get(word)
    
    def get_word(self, word_id: int) -> Optional[str]:
        """Get word from ID."""
        return self.id2word.get(word_id)


class DataProcessor:
    """Data preprocessing utilities for Word2Vec."""
    
    def __init__(self, vocabulary: Vocabulary):
        self.vocabulary = vocabulary
        
    @staticmethod
    def preprocess_text(text: str) -> List[str]:
        """Preprocess text: lowercase, remove punctuation, tokenize."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters, keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize by splitting on whitespace
        tokens = text.split()
        
        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    @staticmethod
    def read_corpus(file_path: str) -> List[List[str]]:
        """Read corpus from file and return list of tokenized sentences."""
        sentences = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tokens = DataProcessor.preprocess_text(line)
                        if tokens:
                            sentences.append(tokens)
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found. Using sample sentences.")
            # Provide sample sentences for demonstration
            sample_text = [
                "the quick brown fox jumps over the lazy dog",
                "word embeddings are useful for natural language processing",
                "machine learning models can learn representations of words",
                "neural networks can capture semantic relationships between words",
                "word2vec is a popular algorithm for learning word embeddings"
            ]
            sentences = [DataProcessor.preprocess_text(text) for text in sample_text]
        
        return sentences
    
    def filter_sentence(self, sentence: List[str]) -> List[int]:
        """Filter sentence by vocabulary and subsampling, return word IDs."""
        filtered_ids = []
        
        for word in sentence:
            word_id = self.vocabulary.get_word_id(word)
            if word_id is not None:
                # Apply subsampling
                if self.vocabulary.subsample_word(word):
                    filtered_ids.append(word_id)
        
        return filtered_ids
    
    def generate_training_pairs(self, sentences: List[List[str]], 
                               window_size: int = 5, 
                               model_type: str = 'skipgram') -> List[Tuple[int, int]]:
        """Generate training pairs for Word2Vec training."""
        training_pairs = []
        
        for sentence in sentences:
            # Filter sentence and convert to IDs
            word_ids = self.filter_sentence(sentence)
            
            if len(word_ids) < 2:
                continue
                
            # Generate context-target pairs
            for i, target_id in enumerate(word_ids):
                # Dynamic window size (randomly choose from 1 to window_size)
                dynamic_window = random.randint(1, window_size)
                
                # Get context words within the window
                start = max(0, i - dynamic_window)
                end = min(len(word_ids), i + dynamic_window + 1)
                
                context_ids = word_ids[start:i] + word_ids[i+1:end]
                
                if model_type == 'skipgram':
                    # Skip-gram: predict context from target
                    for context_id in context_ids:
                        training_pairs.append((target_id, context_id))
                        
                elif model_type == 'cbow':
                    # CBOW: predict target from context
                    if context_ids:
                        training_pairs.append((context_ids, target_id))
        
        return training_pairs
    
    def create_negative_sampling_table(self, table_size: int = 1e8) -> np.ndarray:
        """Create negative sampling table based on word frequencies."""
        table_size = int(table_size)
        table = np.zeros(table_size, dtype=np.int32)
        
        # Calculate probabilities (word_freq^0.75)
        total_pow = 0
        word_probs = {}
        
        for word, freq in self.vocabulary.word_freq.items():
            word_id = self.vocabulary.get_word_id(word)
            if word_id is not None:
                prob = freq ** 0.75
                word_probs[word_id] = prob
                total_pow += prob
        
        # Normalize probabilities
        for word_id in word_probs:
            word_probs[word_id] /= total_pow
        
        # Fill the table
        word_ids = list(word_probs.keys())
        probs = list(word_probs.values())
        
        # Create cumulative distribution
        cumulative = np.cumsum(probs)
        
        # Fill table
        j = 0
        for i in range(table_size):
            while i / table_size > cumulative[j]:
                j += 1
            table[i] = word_ids[j]
        
        return table
    
    def get_negative_samples(self, positive_word: int, 
                           negative_table: np.ndarray, 
                           num_negative: int = 5) -> List[int]:
        """Get negative samples for negative sampling training."""
        negative_samples = []
        
        while len(negative_samples) < num_negative:
            # Random index in the table
            idx = random.randint(0, len(negative_table) - 1)
            negative_word = negative_table[idx]
            
            # Make sure it's not the positive word
            if negative_word != positive_word and negative_word not in negative_samples:
                negative_samples.append(negative_word)
        
        return negative_samples


def create_sample_corpus(filename: str = "sample_corpus.txt") -> None:
    """Create a sample corpus file for demonstration."""
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Word embeddings are dense vector representations of words",
        "Machine learning algorithms can learn from data automatically",
        "Neural networks consist of interconnected nodes called neurons",
        "Natural language processing deals with human language understanding",
        "Deep learning models can capture complex patterns in data",
        "Word2Vec is an algorithm for learning word embeddings",
        "Skip-gram model predicts context words from target words",
        "CBOW model predicts target words from context words",
        "Hierarchical softmax is used for efficient training",
        "Negative sampling is an alternative to hierarchical softmax",
        "Vector representations capture semantic relationships between words",
        "Similar words have similar vector representations in embedding space",
        "Word analogies can be solved using vector arithmetic",
        "King minus man plus woman equals queen in word embeddings",
        "Cosine similarity measures the angle between word vectors",
        "Dimensionality reduction techniques can visualize word embeddings",
        "Transfer learning uses pre-trained embeddings for downstream tasks",
        "Word embeddings improve performance in many NLP applications",
        "Semantic similarity between words can be computed using embeddings"
    ]
    
    with open(filename, 'w', encoding='utf-8') as f:
        for text in sample_texts:
            f.write(text + '\n')
    
    print(f"Sample corpus created: {filename}")


if __name__ == "__main__":
    # Create sample corpus
    create_sample_corpus()
    
    # Test the data processing pipeline
    sentences = DataProcessor.read_corpus("sample_corpus.txt")
    print(f"Read {len(sentences)} sentences")
    
    # Build vocabulary
    vocab = Vocabulary(min_count=1, sample=1e-3)
    vocab.build_vocabulary(sentences)
    
    # Test data processor
    processor = DataProcessor(vocab)
    
    # Generate training pairs
    skipgram_pairs = processor.generate_training_pairs(sentences, window_size=5, model_type='skipgram')
    cbow_pairs = processor.generate_training_pairs(sentences, window_size=5, model_type='cbow')
    
    print(f"Generated {len(skipgram_pairs)} Skip-gram pairs")
    print(f"Generated {len(cbow_pairs)} CBOW pairs")
    
    # Create negative sampling table
    neg_table = processor.create_negative_sampling_table()
    print(f"Created negative sampling table with {len(neg_table)} entries")
    
    # Test negative sampling
    if skipgram_pairs:
        target_word = skipgram_pairs[0][0]
        negative_samples = processor.get_negative_samples(target_word, neg_table, num_negative=5)
        print(f"Negative samples for word '{vocab.get_word(target_word)}': {[vocab.get_word(w) for w in negative_samples]}") 