import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
from typing import List, Tuple, Dict, Optional, Union, Any
from collections import Counter
import pickle
import logging

# Import our custom modules
from data_utils import Vocabulary, DataProcessor, create_sample_corpus
from huffman_tree import HuffmanTree
from hierarchical_softmax import HierarchicalSoftmaxTrainer, HierarchicalSoftmaxOptimized
from negative_sampling import NegativeSampler, NegativeSamplingTrainer, AdvancedNegativeSampler
from cbow_model import CBOWNegativeSampling, CBOWHierarchicalSoftmax, CBOWTrainer
from skipgram_model import SkipGramNegativeSampling, SkipGramHierarchicalSoftmax, SkipGramTrainer


class Word2VecConfig:
    """Configuration class for Word2Vec training."""
    
    def __init__(self):
        # Model parameters
        self.model_type = 'skipgram'  # 'skipgram' or 'cbow'
        self.training_method = 'negative_sampling'  # 'negative_sampling' or 'hierarchical_softmax'
        self.embedding_dim = 100
        self.window_size = 5
        self.min_count = 5
        self.sample = 1e-3  # Subsampling threshold
        
        # Training parameters
        self.num_epochs = 5
        self.batch_size = 128
        self.learning_rate = 0.025
        self.min_learning_rate = 0.0001
        self.num_negative_samples = 5
        self.negative_sampling_power = 0.75
        
        # System parameters
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_workers = 4
        self.save_interval = 10000  # Save model every N steps
        self.eval_interval = 5000   # Evaluate model every N steps
        self.log_interval = 1000    # Log progress every N steps
        
        # File paths
        self.corpus_file = 'corpus.txt'
        self.vocab_file = 'vocabulary.pkl'
        self.model_file = 'word2vec_model.pth'
        self.embeddings_file = 'embeddings.npy'
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Word2VecConfig':
        """Create config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config


class Word2VecTrainer:
    """
    Main Word2Vec trainer that integrates all components.
    """
    
    def __init__(self, config: Word2VecConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        # Initialize components
        self.vocabulary = None
        self.data_processor = None
        self.model = None
        self.trainer = None
        self.huffman_tree = None
        self.negative_sampler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Statistics
        self.training_stats = {
            'total_words_processed': 0,
            'total_training_time': 0,
            'losses': [],
            'learning_rates': []
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for training."""
        logger = logging.getLogger('Word2VecTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def prepare_data(self, corpus_file: Optional[str] = None) -> None:
        """
        Prepare data for training.
        
        Args:
            corpus_file: Path to corpus file. If None, uses config.corpus_file
        """
        if corpus_file is None:
            corpus_file = self.config.corpus_file
        
        self.logger.info("Preparing data...")
        
        # Check if corpus file exists, create sample if not
        if not os.path.exists(corpus_file):
            self.logger.info(f"Corpus file {corpus_file} not found. Creating sample corpus...")
            create_sample_corpus(corpus_file)
        
        # Read corpus
        sentences = DataProcessor.read_corpus(corpus_file)
        self.logger.info(f"Read {len(sentences)} sentences from corpus")
        
        # Build vocabulary
        self.vocabulary = Vocabulary(min_count=self.config.min_count, 
                                   sample=self.config.sample)
        self.vocabulary.build_vocabulary(sentences)
        
        # Initialize data processor
        self.data_processor = DataProcessor(self.vocabulary)
        
        # Save vocabulary
        self._save_vocabulary()
        
        self.logger.info(f"Vocabulary size: {self.vocabulary.vocab_size}")
        
    def _save_vocabulary(self) -> None:
        """Save vocabulary to file."""
        with open(self.config.vocab_file, 'wb') as f:
            pickle.dump(self.vocabulary, f)
        self.logger.info(f"Vocabulary saved to {self.config.vocab_file}")
    
    def _load_vocabulary(self) -> None:
        """Load vocabulary from file."""
        with open(self.config.vocab_file, 'rb') as f:
            self.vocabulary = pickle.load(f)
        self.data_processor = DataProcessor(self.vocabulary)
        self.logger.info(f"Vocabulary loaded from {self.config.vocab_file}")
    
    def build_model(self) -> None:
        """Build Word2Vec model based on configuration."""
        if self.vocabulary is None:
            raise ValueError("Vocabulary not initialized. Call prepare_data() first.")
        
        self.logger.info(f"Building {self.config.model_type} model with {self.config.training_method}")
        
        vocab_size = self.vocabulary.vocab_size
        embedding_dim = self.config.embedding_dim
        
        if self.config.training_method == 'negative_sampling':
            # Build negative sampler
            self.negative_sampler = NegativeSampler(
                self.vocabulary.word_freq,
                power=self.config.negative_sampling_power
            )
            
            # Build model
            if self.config.model_type == 'skipgram':
                self.model = SkipGramNegativeSampling(
                    vocab_size, embedding_dim, 
                    num_negative_samples=self.config.num_negative_samples
                )
                self.trainer = SkipGramTrainer(
                    self.model, self.config.learning_rate, self.config.device
                )
            else:  # cbow
                self.model = CBOWNegativeSampling(
                    vocab_size, embedding_dim,
                    num_negative_samples=self.config.num_negative_samples
                )
                self.trainer = CBOWTrainer(
                    self.model, self.config.learning_rate, self.config.device
                )
            
            # Set negative sampling table
            neg_table = self.data_processor.create_negative_sampling_table()
            self.model.set_negative_sampling_table(neg_table)
            
        elif self.config.training_method == 'hierarchical_softmax':
            # Build Huffman tree
            self.huffman_tree = HuffmanTree(self.vocabulary.word_freq)
            
            # Build model
            if self.config.model_type == 'skipgram':
                self.model = SkipGramHierarchicalSoftmax(
                    vocab_size, embedding_dim, self.huffman_tree
                )
                self.trainer = SkipGramTrainer(
                    self.model, self.config.learning_rate, self.config.device
                )
            else:  # cbow
                self.model = CBOWHierarchicalSoftmax(
                    vocab_size, embedding_dim, self.huffman_tree
                )
                self.trainer = CBOWTrainer(
                    self.model, self.config.learning_rate, self.config.device
                )
        
        self.logger.info(f"Model built with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def generate_training_data(self, corpus_file: Optional[str] = None) -> List[Tuple]:
        """
        Generate training pairs from corpus.
        
        Args:
            corpus_file: Path to corpus file
            
        Returns:
            List of training pairs
        """
        if corpus_file is None:
            corpus_file = self.config.corpus_file
        
        sentences = DataProcessor.read_corpus(corpus_file)
        
        training_pairs = self.data_processor.generate_training_pairs(
            sentences, 
            window_size=self.config.window_size,
            model_type=self.config.model_type
        )
        
        self.logger.info(f"Generated {len(training_pairs)} training pairs")
        return training_pairs
    
    def train(self, corpus_file: Optional[str] = None) -> None:
        """
        Main training loop.
        
        Args:
            corpus_file: Path to corpus file
        """
        self.logger.info("Starting Word2Vec training...")
        
        # Prepare data if not already done
        if self.vocabulary is None:
            self.prepare_data(corpus_file)
        
        # Build model if not already done
        if self.model is None:
            self.build_model()
        
        # Generate training data
        training_pairs = self.generate_training_data(corpus_file)
        
        # Create data loader
        if self.config.model_type == 'skipgram':
            from skipgram_model import create_skipgram_data_loader
            data_loader = create_skipgram_data_loader(
                training_pairs, self.config.batch_size, shuffle=True
            )
        else:  # cbow
            from cbow_model import create_cbow_data_loader
            data_loader = create_cbow_data_loader(
                training_pairs, self.vocabulary.vocab_size, 
                self.config.batch_size, shuffle=True
            )
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            for batch_idx, batch in enumerate(data_loader):
                # Update learning rate
                progress = (epoch * len(data_loader) + batch_idx) / (self.config.num_epochs * len(data_loader))
                self.trainer.update_learning_rate(progress)
                
                # Training step
                if self.config.model_type == 'skipgram':
                    target_words, context_words = batch
                    loss = self.trainer.train_step(target_words, context_words)
                else:  # cbow
                    context_words, target_words = batch
                    loss = self.trainer.train_step(context_words, target_words)
                
                epoch_loss += loss
                num_batches += 1
                self.global_step += 1
                
                # Update statistics
                self.training_stats['losses'].append(loss)
                current_lr = self.trainer.optimizer.param_groups[0]['lr']
                self.training_stats['learning_rates'].append(current_lr)
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    self.logger.info(
                        f"Step {self.global_step}, Epoch {epoch + 1}, "
                        f"Batch {batch_idx + 1}, Loss: {loss:.4f}, "
                        f"Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
                    )
                
                # Save model
                if self.global_step % self.config.save_interval == 0:
                    self.save_model()
                
                # Evaluation
                if self.global_step % self.config.eval_interval == 0:
                    self.evaluate()
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            self.logger.info(
                f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}"
            )
            
            # Save best model
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self.save_model(suffix='_best')
        
        # Training completed
        end_time = time.time()
        self.training_stats['total_training_time'] = end_time - start_time
        
        self.logger.info(f"Training completed in {self.training_stats['total_training_time']:.2f} seconds")
        
        # Save final model and embeddings
        self.save_model()
        self.save_embeddings()
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model (placeholder for now).
        
        Returns:
            Dictionary of evaluation metrics
        """
        # For now, just return basic statistics
        recent_losses = self.training_stats['losses'][-100:]  # Last 100 losses
        avg_recent_loss = np.mean(recent_losses) if recent_losses else 0.0
        
        metrics = {
            'avg_recent_loss': avg_recent_loss,
            'total_steps': self.global_step,
            'vocab_size': self.vocabulary.vocab_size if self.vocabulary else 0
        }
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def save_model(self, suffix: str = '') -> None:
        """
        Save model state.
        
        Args:
            suffix: Suffix to add to filename
        """
        if self.trainer is None:
            self.logger.warning("No trainer to save")
            return
        
        filename = self.config.model_file
        if suffix:
            name, ext = os.path.splitext(filename)
            filename = f"{name}{suffix}{ext}"
        
        # Save model and training state
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'trainer_state_dict': self.trainer.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'training_stats': self.training_stats
        }
        
        torch.save(save_dict, filename)
        self.logger.info(f"Model saved to {filename}")
    
    def load_model(self, filename: Optional[str] = None) -> None:
        """
        Load model state.
        
        Args:
            filename: Path to model file
        """
        if filename is None:
            filename = self.config.model_file
        
        if not os.path.exists(filename):
            self.logger.error(f"Model file {filename} not found")
            return
        
        checkpoint = torch.load(filename, map_location=self.config.device)
        
        # Load config
        self.config = Word2VecConfig.from_dict(checkpoint['config'])
        
        # Load vocabulary
        self._load_vocabulary()
        
        # Build model
        self.build_model()
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['trainer_state_dict'])
        
        # Load training state
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_stats = checkpoint.get('training_stats', {})
        
        self.logger.info(f"Model loaded from {filename}")
    
    def save_embeddings(self, filename: Optional[str] = None) -> None:
        """
        Save word embeddings.
        
        Args:
            filename: Path to save embeddings
        """
        if filename is None:
            filename = self.config.embeddings_file
        
        if self.trainer is None:
            self.logger.warning("No trainer to get embeddings from")
            return
        
        embeddings = self.trainer.get_embeddings()
        np.save(filename, embeddings)
        
        # Also save vocabulary for reference
        vocab_dict = {
            'word2id': self.vocabulary.word2id,
            'id2word': self.vocabulary.id2word
        }
        
        vocab_filename = filename.replace('.npy', '_vocab.pkl')
        with open(vocab_filename, 'wb') as f:
            pickle.dump(vocab_dict, f)
        
        self.logger.info(f"Embeddings saved to {filename}")
        self.logger.info(f"Vocabulary mapping saved to {vocab_filename}")
    
    def get_word_embedding(self, word: str) -> Optional[np.ndarray]:
        """
        Get embedding for a specific word.
        
        Args:
            word: Word to get embedding for
            
        Returns:
            Word embedding or None if word not in vocabulary
        """
        if self.vocabulary is None or self.trainer is None:
            return None
        
        word_id = self.vocabulary.get_word_id(word)
        if word_id is None:
            return None
        
        embeddings = self.trainer.get_embeddings()
        return embeddings[word_id]
    
    def find_similar_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find words similar to the given word.
        
        Args:
            word: Query word
            top_k: Number of similar words to return
            
        Returns:
            List of (word, similarity) tuples
        """
        if self.vocabulary is None or self.trainer is None:
            return []
        
        word_embedding = self.get_word_embedding(word)
        if word_embedding is None:
            return []
        
        # Get all embeddings
        all_embeddings = self.trainer.get_embeddings()
        
        # Compute cosine similarities
        similarities = []
        for word_id in range(self.vocabulary.vocab_size):
            if word_id == self.vocabulary.get_word_id(word):
                continue  # Skip the query word itself
            
            other_embedding = all_embeddings[word_id]
            similarity = np.dot(word_embedding, other_embedding) / (
                np.linalg.norm(word_embedding) * np.linalg.norm(other_embedding)
            )
            
            other_word = self.vocabulary.get_word(word_id)
            if other_word:
                similarities.append((other_word, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def create_default_config() -> Word2VecConfig:
    """Create default configuration."""
    return Word2VecConfig()


def train_word2vec(corpus_file: str, config: Optional[Word2VecConfig] = None) -> Word2VecTrainer:
    """
    Convenience function to train Word2Vec model.
    
    Args:
        corpus_file: Path to corpus file
        config: Configuration object
        
    Returns:
        Trained Word2Vec trainer
    """
    if config is None:
        config = create_default_config()
    
    trainer = Word2VecTrainer(config)
    trainer.train(corpus_file)
    
    return trainer


if __name__ == "__main__":
    # Test the Word2Vec trainer
    print("Testing Word2Vec trainer...")
    
    # Create configuration
    config = Word2VecConfig()
    config.num_epochs = 2
    config.batch_size = 32
    config.embedding_dim = 50
    config.log_interval = 100
    config.save_interval = 1000
    config.eval_interval = 500
    
    # Test Skip-gram with negative sampling
    print("\nTesting Skip-gram with negative sampling...")
    config.model_type = 'skipgram'
    config.training_method = 'negative_sampling'
    
    trainer = Word2VecTrainer(config)
    trainer.train()
    
    # Test similarity search
    print("\nTesting similarity search...")
    similar_words = trainer.find_similar_words('word', top_k=5)
    print(f"Words similar to 'word': {similar_words}")
    
    # Test CBOW with hierarchical softmax
    print("\nTesting CBOW with hierarchical softmax...")
    config.model_type = 'cbow'
    config.training_method = 'hierarchical_softmax'
    
    trainer2 = Word2VecTrainer(config)
    trainer2.train()
    
    print("\nWord2Vec trainer testing completed successfully!") 