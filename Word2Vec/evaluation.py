import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
import pickle


class Word2VecEvaluator:
    """
    Comprehensive evaluation suite for Word2Vec embeddings.
    """
    
    def __init__(self, embeddings: np.ndarray, vocabulary: Dict[str, int], 
                 word_to_id: Dict[str, int], id_to_word: Dict[int, str]):
        self.embeddings = embeddings
        self.vocabulary = vocabulary
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.vocab_size = len(vocabulary)
        
        # Normalize embeddings for cosine similarity
        self.normalized_embeddings = self._normalize_embeddings()
        
    def _normalize_embeddings(self) -> np.ndarray:
        """Normalize embeddings for cosine similarity computation."""
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return self.embeddings / norms
    
    def get_word_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding for a word."""
        word_id = self.word_to_id.get(word)
        if word_id is None:
            return None
        return self.embeddings[word_id]
    
    def cosine_similarity(self, word1: str, word2: str) -> Optional[float]:
        """Compute cosine similarity between two words."""
        emb1 = self.get_word_embedding(word1)
        emb2 = self.get_word_embedding(word2)
        
        if emb1 is None or emb2 is None:
            return None
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(emb1, emb2) / (norm1 * norm2)
    
    def find_similar_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words to a given word."""
        word_id = self.word_to_id.get(word)
        if word_id is None:
            return []
        
        word_embedding = self.normalized_embeddings[word_id]
        similarities = np.dot(self.normalized_embeddings, word_embedding)
        
        # Get top-k similar words (excluding the word itself)
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            if idx == word_id:
                continue  # Skip the word itself
            
            similar_word = self.id_to_word.get(idx)
            if similar_word and len(results) < top_k:
                results.append((similar_word, similarities[idx]))
        
        return results
    
    def analogy_task(self, word_a: str, word_b: str, word_c: str, 
                    top_k: int = 1) -> List[Tuple[str, float]]:
        """
        Solve analogy task: word_a is to word_b as word_c is to ?
        
        Args:
            word_a: First word in analogy
            word_b: Second word in analogy
            word_c: Third word in analogy
            top_k: Number of candidate answers to return
            
        Returns:
            List of (word, similarity) tuples
        """
        # Get embeddings
        emb_a = self.get_word_embedding(word_a)
        emb_b = self.get_word_embedding(word_b)
        emb_c = self.get_word_embedding(word_c)
        
        if emb_a is None or emb_b is None or emb_c is None:
            return []
        
        # Compute analogy vector: b - a + c
        analogy_vector = emb_b - emb_a + emb_c
        
        # Normalize
        analogy_vector = analogy_vector / np.linalg.norm(analogy_vector)
        
        # Find most similar words
        similarities = np.dot(self.normalized_embeddings, analogy_vector)
        
        # Get top-k candidates (excluding input words)
        top_indices = np.argsort(similarities)[::-1]
        exclude_words = {word_a, word_b, word_c}
        
        results = []
        for idx in top_indices:
            candidate_word = self.id_to_word.get(idx)
            if candidate_word and candidate_word not in exclude_words:
                results.append((candidate_word, similarities[idx]))
                if len(results) >= top_k:
                    break
        
        return results
    
    def evaluate_word_similarity(self, similarity_file: str) -> Dict[str, float]:
        """
        Evaluate on word similarity datasets.
        
        Args:
            similarity_file: Path to similarity dataset file
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not os.path.exists(similarity_file):
            return {'error': f'Similarity file {similarity_file} not found'}
        
        # Load similarity data
        try:
            data = pd.read_csv(similarity_file, sep='\t', header=None,
                             names=['word1', 'word2', 'similarity'])
        except:
            try:
                data = pd.read_csv(similarity_file, sep=',', header=None,
                                 names=['word1', 'word2', 'similarity'])
            except:
                return {'error': 'Could not parse similarity file'}
        
        # Compute similarities
        predicted_similarities = []
        actual_similarities = []
        
        for _, row in data.iterrows():
            word1, word2, actual_sim = row['word1'], row['word2'], row['similarity']
            
            predicted_sim = self.cosine_similarity(word1, word2)
            
            if predicted_sim is not None:
                predicted_similarities.append(predicted_sim)
                actual_similarities.append(actual_sim)
        
        if len(predicted_similarities) == 0:
            return {'error': 'No valid word pairs found'}
        
        # Compute correlation
        correlation, p_value = spearmanr(actual_similarities, predicted_similarities)
        
        return {
            'spearman_correlation': correlation,
            'p_value': p_value,
            'num_pairs': len(predicted_similarities),
            'coverage': len(predicted_similarities) / len(data)
        }
    
    def evaluate_analogies(self, analogy_file: str) -> Dict[str, float]:
        """
        Evaluate on analogy datasets.
        
        Args:
            analogy_file: Path to analogy dataset file
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not os.path.exists(analogy_file):
            return {'error': f'Analogy file {analogy_file} not found'}
        
        # Load analogy data
        analogies = []
        try:
            with open(analogy_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # Skip comments
                        parts = line.split()
                        if len(parts) >= 4:
                            analogies.append(parts[:4])
        except:
            return {'error': 'Could not parse analogy file'}
        
        # Evaluate analogies
        correct = 0
        total = 0
        
        for analogy in analogies:
            word_a, word_b, word_c, expected_d = analogy
            
            # Get top-1 prediction
            predictions = self.analogy_task(word_a, word_b, word_c, top_k=1)
            
            if predictions:
                predicted_d = predictions[0][0]
                if predicted_d == expected_d:
                    correct += 1
                total += 1
        
        if total == 0:
            return {'error': 'No valid analogies found'}
        
        accuracy = correct / total
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'coverage': total / len(analogies)
        }
    
    def cluster_analysis(self, words: List[str], n_clusters: int = 5) -> Dict[str, Any]:
        """
        Perform clustering analysis on a set of words.
        
        Args:
            words: List of words to cluster
            n_clusters: Number of clusters
            
        Returns:
            Dictionary with clustering results
        """
        # Get embeddings for words
        word_embeddings = []
        valid_words = []
        
        for word in words:
            embedding = self.get_word_embedding(word)
            if embedding is not None:
                word_embeddings.append(embedding)
                valid_words.append(word)
        
        if len(word_embeddings) < n_clusters:
            return {'error': f'Not enough valid words for clustering (need at least {n_clusters})'}
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(word_embeddings)
        
        # Organize results
        clusters = {}
        for i, word in enumerate(valid_words):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(word)
        
        return {
            'clusters': clusters,
            'cluster_labels': cluster_labels.tolist(),
            'valid_words': valid_words,
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_
        }
    
    def visualize_embeddings(self, words: List[str], method: str = 'tsne', 
                           save_path: Optional[str] = None) -> None:
        """
        Visualize word embeddings in 2D space.
        
        Args:
            words: List of words to visualize
            method: Dimensionality reduction method ('tsne' or 'pca')
            save_path: Path to save the plot
        """
        # Get embeddings for words
        word_embeddings = []
        valid_words = []
        
        for word in words:
            embedding = self.get_word_embedding(word)
            if embedding is not None:
                word_embeddings.append(embedding)
                valid_words.append(word)
        
        if len(word_embeddings) < 2:
            print("Not enough valid words for visualization")
            return
        
        word_embeddings = np.array(word_embeddings)
        
        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(word_embeddings)-1))
        else:  # pca
            reducer = PCA(n_components=2, random_state=42)
        
        embeddings_2d = reducer.fit_transform(word_embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        # Add word labels
        for i, word in enumerate(valid_words):
            plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title(f'Word Embeddings Visualization ({method.upper()})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def generate_evaluation_report(self, output_file: str = 'evaluation_report.txt') -> None:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_file: Path to save the report
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Word2Vec Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Vocabulary Size: {self.vocab_size}\n")
            f.write(f"Embedding Dimension: {self.embeddings.shape[1]}\n\n")
            
            # Sample similar words
            f.write("Sample Similar Words:\n")
            f.write("-" * 20 + "\n")
            
            sample_words = ['king', 'queen', 'man', 'woman', 'computer', 'science']
            for word in sample_words:
                if word in self.word_to_id:
                    similar = self.find_similar_words(word, top_k=5)
                    f.write(f"{word}: {[w for w, _ in similar]}\n")
            
            f.write("\n")
            
            # Sample analogies
            f.write("Sample Analogies:\n")
            f.write("-" * 15 + "\n")
            
            analogies = [
                ('king', 'queen', 'man'),
                ('paris', 'france', 'london'),
                ('good', 'better', 'bad'),
                ('big', 'bigger', 'small')
            ]
            
            for word_a, word_b, word_c in analogies:
                if all(w in self.word_to_id for w in [word_a, word_b, word_c]):
                    result = self.analogy_task(word_a, word_b, word_c, top_k=3)
                    f.write(f"{word_a} : {word_b} :: {word_c} : {[w for w, _ in result]}\n")
            
            f.write("\n")
            
            # Clustering analysis
            f.write("Clustering Analysis:\n")
            f.write("-" * 18 + "\n")
            
            color_words = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'white']
            animal_words = ['cat', 'dog', 'bird', 'fish', 'horse', 'cow', 'pig', 'sheep']
            
            for category, words in [('Colors', color_words), ('Animals', animal_words)]:
                valid_words = [w for w in words if w in self.word_to_id]
                if len(valid_words) >= 3:
                    cluster_result = self.cluster_analysis(valid_words, n_clusters=min(3, len(valid_words)))
                    if 'clusters' in cluster_result:
                        f.write(f"{category}: {cluster_result['clusters']}\n")
            
        print(f"Evaluation report saved to {output_file}")


def create_sample_similarity_dataset(filename: str = 'sample_similarity.txt') -> None:
    """Create a sample word similarity dataset for testing."""
    sample_pairs = [
        ('king', 'queen', 8.5),
        ('man', 'woman', 7.3),
        ('computer', 'machine', 6.8),
        ('car', 'automobile', 9.2),
        ('happy', 'sad', 2.1),
        ('good', 'bad', 1.8),
        ('big', 'large', 8.7),
        ('small', 'tiny', 7.9),
        ('fast', 'quick', 8.3),
        ('slow', 'sluggish', 7.2)
    ]
    
    with open(filename, 'w', encoding='utf-8') as f:
        for word1, word2, similarity in sample_pairs:
            f.write(f"{word1}\t{word2}\t{similarity}\n")
    
    print(f"Sample similarity dataset created: {filename}")


def create_sample_analogy_dataset(filename: str = 'sample_analogies.txt') -> None:
    """Create a sample analogy dataset for testing."""
    sample_analogies = [
        ('king', 'queen', 'man', 'woman'),
        ('good', 'better', 'bad', 'worse'),
        ('big', 'bigger', 'small', 'smaller'),
        ('fast', 'faster', 'slow', 'slower'),
        ('happy', 'happiness', 'sad', 'sadness'),
        ('strong', 'stronger', 'weak', 'weaker'),
        ('hot', 'hotter', 'cold', 'colder'),
        ('high', 'higher', 'low', 'lower'),
        ('long', 'longer', 'short', 'shorter'),
        ('deep', 'deeper', 'shallow', 'shallower')
    ]
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Sample analogy dataset\n")
        f.write("# Format: word1 word2 word3 word4\n")
        f.write("# Represents: word1 is to word2 as word3 is to word4\n\n")
        
        for word1, word2, word3, word4 in sample_analogies:
            f.write(f"{word1} {word2} {word3} {word4}\n")
    
    print(f"Sample analogy dataset created: {filename}")


def evaluate_embeddings(embeddings_file: str, vocab_file: str, 
                       similarity_file: Optional[str] = None,
                       analogy_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive evaluation of word embeddings.
    
    Args:
        embeddings_file: Path to embeddings file
        vocab_file: Path to vocabulary file
        similarity_file: Path to similarity dataset
        analogy_file: Path to analogy dataset
        
    Returns:
        Dictionary of evaluation results
    """
    # Load embeddings and vocabulary
    embeddings = np.load(embeddings_file)
    
    with open(vocab_file, 'rb') as f:
        vocab_data = pickle.load(f)
    
    if isinstance(vocab_data, dict):
        word_to_id = vocab_data.get('word2id', {})
        id_to_word = vocab_data.get('id2word', {})
    else:
        # Assuming vocab_data is a Vocabulary object
        word_to_id = vocab_data.word2id
        id_to_word = vocab_data.id2word
    
    # Create evaluator
    evaluator = Word2VecEvaluator(embeddings, word_to_id, word_to_id, id_to_word)
    
    results = {}
    
    # Word similarity evaluation
    if similarity_file and os.path.exists(similarity_file):
        similarity_results = evaluator.evaluate_word_similarity(similarity_file)
        results['word_similarity'] = similarity_results
    
    # Analogy evaluation
    if analogy_file and os.path.exists(analogy_file):
        analogy_results = evaluator.evaluate_analogies(analogy_file)
        results['analogies'] = analogy_results
    
    # Generate report
    evaluator.generate_evaluation_report()
    
    return results


if __name__ == "__main__":
    # Test evaluation utilities
    print("Testing Word2Vec evaluation utilities...")
    
    # Create sample datasets
    create_sample_similarity_dataset()
    create_sample_analogy_dataset()
    
    # Create dummy embeddings for testing
    vocab_size = 100
    embedding_dim = 50
    
    # Generate random embeddings
    embeddings = np.random.randn(vocab_size, embedding_dim)
    
    # Create vocabulary mapping
    sample_words = ['king', 'queen', 'man', 'woman', 'computer', 'science', 
                   'good', 'better', 'bad', 'worse', 'big', 'small', 'fast', 'slow']
    
    word_to_id = {word: i for i, word in enumerate(sample_words)}
    id_to_word = {i: word for i, word in enumerate(sample_words)}
    
    # Pad with random words
    for i in range(len(sample_words), vocab_size):
        word = f"word_{i}"
        word_to_id[word] = i
        id_to_word[i] = word
    
    # Create evaluator
    evaluator = Word2VecEvaluator(embeddings, word_to_id, word_to_id, id_to_word)
    
    # Test similarity
    print("\nTesting similarity...")
    similarity = evaluator.cosine_similarity('king', 'queen')
    print(f"Similarity between 'king' and 'queen': {similarity}")
    
    # Test finding similar words
    print("\nTesting similar words...")
    similar_words = evaluator.find_similar_words('king', top_k=5)
    print(f"Words similar to 'king': {similar_words}")
    
    # Test analogy
    print("\nTesting analogy...")
    analogy_result = evaluator.analogy_task('king', 'queen', 'man', top_k=3)
    print(f"king : queen :: man : {analogy_result}")
    
    # Test clustering
    print("\nTesting clustering...")
    cluster_result = evaluator.cluster_analysis(sample_words[:10], n_clusters=3)
    print(f"Clustering result: {cluster_result}")
    
    # Generate report
    print("\nGenerating evaluation report...")
    evaluator.generate_evaluation_report()
    
    print("\nEvaluation utilities testing completed successfully!") 