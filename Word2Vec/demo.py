#!/usr/bin/env python3
"""
Word2Vec Demo Script

This script demonstrates the complete Word2Vec implementation including:
- Data preparation and preprocessing
- Training both CBOW and Skip-gram models
- Hierarchical softmax and negative sampling
- Evaluation and visualization
- Similarity search and analogies

Author: Word2Vec Implementation Team
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

# Import our Word2Vec implementation
from word2vec_trainer import Word2VecTrainer, Word2VecConfig, train_word2vec
from evaluation import Word2VecEvaluator, create_sample_similarity_dataset, create_sample_analogy_dataset
from data_utils import create_sample_corpus
import pickle


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * len(title))


def demo_data_preparation() -> None:
    """Demonstrate data preparation and preprocessing."""
    print_section("DATA PREPARATION AND PREPROCESSING")
    
    # Create sample corpus if it doesn't exist
    corpus_file = "demo_corpus.txt"
    if not os.path.exists(corpus_file):
        print("Creating sample corpus...")
        create_sample_corpus(corpus_file)
    
    # Read and display some statistics
    from data_utils import DataProcessor, Vocabulary
    
    sentences = DataProcessor.read_corpus(corpus_file)
    print(f"✓ Loaded {len(sentences)} sentences from corpus")
    
    # Show first few sentences
    print("\nFirst 5 sentences:")
    for i, sentence in enumerate(sentences[:5]):
        print(f"  {i+1}. {' '.join(sentence)}")
    
    # Build vocabulary
    vocab = Vocabulary(min_count=1, sample=1e-3)
    vocab.build_vocabulary(sentences)
    print(f"\n✓ Built vocabulary with {vocab.vocab_size} words")
    
    # Show most frequent words
    most_frequent = sorted(vocab.word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nMost frequent words:")
    for word, freq in most_frequent:
        print(f"  {word}: {freq}")
    
    # Generate training pairs
    processor = DataProcessor(vocab)
    skipgram_pairs = processor.generate_training_pairs(sentences, window_size=5, model_type='skipgram')
    cbow_pairs = processor.generate_training_pairs(sentences, window_size=5, model_type='cbow')
    
    print(f"\n✓ Generated {len(skipgram_pairs)} Skip-gram training pairs")
    print(f"✓ Generated {len(cbow_pairs)} CBOW training pairs")
    
    # Show sample training pairs
    print("\nSample Skip-gram pairs (target -> context):")
    for i, (target, context) in enumerate(skipgram_pairs[:5]):
        target_word = vocab.get_word(target)
        context_word = vocab.get_word(context)
        print(f"  {target_word} -> {context_word}")


def demo_skipgram_training() -> None:
    """Demonstrate Skip-gram model training."""
    print_section("SKIP-GRAM MODEL TRAINING")
    
    # Configure Skip-gram with negative sampling
    config = Word2VecConfig()
    config.model_type = 'skipgram'
    config.training_method = 'negative_sampling'
    config.embedding_dim = 100
    config.num_epochs = 3
    config.batch_size = 64
    config.learning_rate = 0.025
    config.num_negative_samples = 5
    config.log_interval = 500
    config.save_interval = 2000
    config.eval_interval = 1000
    config.corpus_file = "demo_corpus.txt"
    config.model_file = "skipgram_model.pth"
    config.embeddings_file = "skipgram_embeddings.npy"
    
    print("Configuration:")
    print(f"  Model: {config.model_type}")
    print(f"  Training method: {config.training_method}")
    print(f"  Embedding dimension: {config.embedding_dim}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Negative samples: {config.num_negative_samples}")
    
    # Train the model
    print("\nStarting Skip-gram training...")
    start_time = time.time()
    
    trainer = Word2VecTrainer(config)
    trainer.train()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"✓ Skip-gram training completed in {training_time:.2f} seconds")
    print(f"✓ Model saved to {config.model_file}")
    print(f"✓ Embeddings saved to {config.embeddings_file}")
    
    return trainer


def demo_cbow_training() -> None:
    """Demonstrate CBOW model training."""
    print_section("CBOW MODEL TRAINING")
    
    # Configure CBOW with hierarchical softmax
    config = Word2VecConfig()
    config.model_type = 'cbow'
    config.training_method = 'hierarchical_softmax'
    config.embedding_dim = 100
    config.num_epochs = 3
    config.batch_size = 64
    config.learning_rate = 0.025
    config.log_interval = 500
    config.save_interval = 2000
    config.eval_interval = 1000
    config.corpus_file = "demo_corpus.txt"
    config.model_file = "cbow_model.pth"
    config.embeddings_file = "cbow_embeddings.npy"
    
    print("Configuration:")
    print(f"  Model: {config.model_type}")
    print(f"  Training method: {config.training_method}")
    print(f"  Embedding dimension: {config.embedding_dim}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Train the model
    print("\nStarting CBOW training...")
    start_time = time.time()
    
    trainer = Word2VecTrainer(config)
    trainer.train()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"✓ CBOW training completed in {training_time:.2f} seconds")
    print(f"✓ Model saved to {config.model_file}")
    print(f"✓ Embeddings saved to {config.embeddings_file}")
    
    return trainer


def demo_model_evaluation(trainer: Word2VecTrainer, model_name: str) -> None:
    """Demonstrate model evaluation."""
    print_section(f"MODEL EVALUATION - {model_name.upper()}")
    
    # Create sample evaluation datasets
    create_sample_similarity_dataset()
    create_sample_analogy_dataset()
    
    # Test word similarity
    print_subsection("Word Similarity")
    
    test_pairs = [
        ('word', 'words'),
        ('computer', 'machine'),
        ('learning', 'models'),
        ('neural', 'networks'),
        ('language', 'processing')
    ]
    
    print("Cosine similarities:")
    for word1, word2 in test_pairs:
        emb1 = trainer.get_word_embedding(word1)
        emb2 = trainer.get_word_embedding(word2)
        
        if emb1 is not None and emb2 is not None:
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            print(f"  {word1} - {word2}: {similarity:.4f}")
        else:
            print(f"  {word1} - {word2}: N/A (word not in vocabulary)")
    
    # Test finding similar words
    print_subsection("Similar Words")
    
    test_words = ['word', 'computer', 'learning', 'neural', 'language']
    
    for word in test_words:
        similar_words = trainer.find_similar_words(word, top_k=5)
        if similar_words:
            similar_list = [f"{w} ({s:.3f})" for w, s in similar_words]
            print(f"  {word}: {', '.join(similar_list)}")
        else:
            print(f"  {word}: N/A (word not in vocabulary)")
    
    # Test analogies
    print_subsection("Word Analogies")
    
    test_analogies = [
        ('king', 'queen', 'man'),
        ('good', 'better', 'bad'),
        ('big', 'bigger', 'small'),
        ('computer', 'computers', 'word'),
        ('neural', 'networks', 'machine')
    ]
    
    for word_a, word_b, word_c in test_analogies:
        # Get embeddings
        emb_a = trainer.get_word_embedding(word_a)
        emb_b = trainer.get_word_embedding(word_b)
        emb_c = trainer.get_word_embedding(word_c)
        
        if emb_a is not None and emb_b is not None and emb_c is not None:
            # Compute analogy vector
            analogy_vector = emb_b - emb_a + emb_c
            
            # Find most similar word
            embeddings = trainer.get_embeddings()
            similarities = np.dot(embeddings, analogy_vector) / (
                np.linalg.norm(embeddings, axis=1) * np.linalg.norm(analogy_vector)
            )
            
            # Get top candidate (excluding input words)
            top_idx = np.argsort(similarities)[::-1]
            exclude_ids = {trainer.vocabulary.get_word_id(w) for w in [word_a, word_b, word_c]}
            
            for idx in top_idx:
                if idx not in exclude_ids:
                    candidate = trainer.vocabulary.get_word(idx)
                    if candidate:
                        print(f"  {word_a} : {word_b} :: {word_c} : {candidate} ({similarities[idx]:.3f})")
                        break
        else:
            print(f"  {word_a} : {word_b} :: {word_c} : N/A (some words not in vocabulary)")


def demo_visualization(trainer: Word2VecTrainer, model_name: str) -> None:
    """Demonstrate embedding visualization."""
    print_section(f"EMBEDDING VISUALIZATION - {model_name.upper()}")
    
    try:
        # Get embeddings and vocabulary
        embeddings = trainer.get_embeddings()
        vocab = trainer.vocabulary
        
        # Create evaluator for visualization
        evaluator = Word2VecEvaluator(
            embeddings, vocab.word2id, vocab.word2id, vocab.id2word
        )
        
        # Select words for visualization
        visualization_words = []
        sample_words = ['word', 'words', 'computer', 'machine', 'learning', 'models', 
                       'neural', 'networks', 'language', 'processing', 'deep', 'data',
                       'algorithm', 'training', 'vector', 'representation']
        
        for word in sample_words:
            if word in vocab.word2id:
                visualization_words.append(word)
        
        if len(visualization_words) >= 10:
            print(f"Visualizing {len(visualization_words)} words...")
            
            # Create visualization
            save_path = f"{model_name.lower()}_embeddings_visualization.png"
            evaluator.visualize_embeddings(visualization_words, method='tsne', save_path=save_path)
            print(f"✓ Visualization saved to {save_path}")
        else:
            print("⚠ Not enough words available for visualization")
            
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")
        print("  (This might happen if required packages are not installed)")


def demo_comprehensive_evaluation(trainer: Word2VecTrainer, model_name: str) -> None:
    """Demonstrate comprehensive evaluation."""
    print_section(f"COMPREHENSIVE EVALUATION - {model_name.upper()}")
    
    # Create evaluator
    embeddings = trainer.get_embeddings()
    vocab = trainer.vocabulary
    
    evaluator = Word2VecEvaluator(
        embeddings, vocab.word2id, vocab.word2id, vocab.id2word
    )
    
    # Word similarity evaluation
    print_subsection("Word Similarity Evaluation")
    
    similarity_results = evaluator.evaluate_word_similarity('sample_similarity.txt')
    
    if 'error' not in similarity_results:
        print(f"  Spearman correlation: {similarity_results['spearman_correlation']:.4f}")
        print(f"  P-value: {similarity_results['p_value']:.4f}")
        print(f"  Coverage: {similarity_results['coverage']:.2%}")
        print(f"  Valid pairs: {similarity_results['num_pairs']}")
    else:
        print(f"  Error: {similarity_results['error']}")
    
    # Analogy evaluation
    print_subsection("Analogy Evaluation")
    
    analogy_results = evaluator.evaluate_analogies('sample_analogies.txt')
    
    if 'error' not in analogy_results:
        print(f"  Accuracy: {analogy_results['accuracy']:.2%}")
        print(f"  Correct: {analogy_results['correct']}")
        print(f"  Total: {analogy_results['total']}")
        print(f"  Coverage: {analogy_results['coverage']:.2%}")
    else:
        print(f"  Error: {analogy_results['error']}")
    
    # Clustering analysis
    print_subsection("Clustering Analysis")
    
    # Test clustering on available words
    test_words = ['word', 'words', 'computer', 'machine', 'learning', 'models', 
                  'neural', 'networks', 'language', 'processing']
    
    available_words = [w for w in test_words if w in vocab.word2id]
    
    if len(available_words) >= 5:
        cluster_results = evaluator.cluster_analysis(available_words, n_clusters=3)
        
        if 'error' not in cluster_results:
            print(f"  Clustered {len(cluster_results['valid_words'])} words into {cluster_results['n_clusters']} clusters")
            print(f"  Inertia: {cluster_results['inertia']:.4f}")
            
            for cluster_id, words in cluster_results['clusters'].items():
                print(f"    Cluster {cluster_id}: {words}")
        else:
            print(f"  Error: {cluster_results['error']}")
    else:
        print("  ⚠ Not enough words available for clustering")
    
    # Generate comprehensive report
    print_subsection("Generating Evaluation Report")
    
    report_file = f"{model_name.lower()}_evaluation_report.txt"
    evaluator.generate_evaluation_report(report_file)
    print(f"✓ Comprehensive evaluation report saved to {report_file}")


def demo_model_comparison() -> None:
    """Demonstrate model comparison."""
    print_section("MODEL COMPARISON")
    
    # Load both models if they exist
    models = {}
    
    for model_name, model_file in [('Skip-gram', 'skipgram_model.pth'), 
                                  ('CBOW', 'cbow_model.pth')]:
        if os.path.exists(model_file):
            try:
                config = Word2VecConfig()
                config.model_file = model_file
                trainer = Word2VecTrainer(config)
                trainer.load_model(model_file)
                models[model_name] = trainer
                print(f"✓ Loaded {model_name} model")
            except Exception as e:
                print(f"⚠ Failed to load {model_name} model: {e}")
    
    if len(models) < 2:
        print("⚠ Need at least 2 models for comparison")
        return
    
    # Compare word similarities
    print_subsection("Word Similarity Comparison")
    
    test_words = ['word', 'computer', 'learning', 'neural']
    
    for word in test_words:
        print(f"\nWords similar to '{word}':")
        
        for model_name, trainer in models.items():
            similar_words = trainer.find_similar_words(word, top_k=3)
            if similar_words:
                similar_list = [f"{w} ({s:.3f})" for w, s in similar_words]
                print(f"  {model_name}: {', '.join(similar_list)}")
            else:
                print(f"  {model_name}: N/A")
    
    # Compare analogy performance
    print_subsection("Analogy Performance Comparison")
    
    test_analogies = [
        ('king', 'queen', 'man'),
        ('good', 'better', 'bad'),
        ('big', 'bigger', 'small')
    ]
    
    for word_a, word_b, word_c in test_analogies:
        print(f"\n{word_a} : {word_b} :: {word_c} : ?")
        
        for model_name, trainer in models.items():
            # Simple analogy computation
            emb_a = trainer.get_word_embedding(word_a)
            emb_b = trainer.get_word_embedding(word_b)
            emb_c = trainer.get_word_embedding(word_c)
            
            if emb_a is not None and emb_b is not None and emb_c is not None:
                analogy_vector = emb_b - emb_a + emb_c
                embeddings = trainer.get_embeddings()
                similarities = np.dot(embeddings, analogy_vector) / (
                    np.linalg.norm(embeddings, axis=1) * np.linalg.norm(analogy_vector)
                )
                
                # Get top candidate
                top_idx = np.argsort(similarities)[::-1]
                exclude_ids = {trainer.vocabulary.get_word_id(w) for w in [word_a, word_b, word_c]}
                
                for idx in top_idx:
                    if idx not in exclude_ids:
                        candidate = trainer.vocabulary.get_word(idx)
                        if candidate:
                            print(f"  {model_name}: {candidate} ({similarities[idx]:.3f})")
                            break
            else:
                print(f"  {model_name}: N/A")


def main():
    """Main demo function."""
    print("🚀 Word2Vec Complete Implementation Demo")
    print("=" * 80)
    print("This demo showcases a full Word2Vec implementation including:")
    print("  • Data preparation and preprocessing")
    print("  • Skip-gram and CBOW model training")
    print("  • Hierarchical softmax and negative sampling")
    print("  • Comprehensive evaluation and visualization")
    print("  • Model comparison and analysis")
    print("=" * 80)
    
    try:
        # 1. Data preparation
        demo_data_preparation()
        
        # 2. Skip-gram training
        skipgram_trainer = demo_skipgram_training()
        
        # 3. CBOW training
        cbow_trainer = demo_cbow_training()
        
        # 4. Model evaluation
        demo_model_evaluation(skipgram_trainer, "Skip-gram")
        demo_model_evaluation(cbow_trainer, "CBOW")
        
        # 5. Visualization
        demo_visualization(skipgram_trainer, "Skip-gram")
        demo_visualization(cbow_trainer, "CBOW")
        
        # 6. Comprehensive evaluation
        demo_comprehensive_evaluation(skipgram_trainer, "Skip-gram")
        demo_comprehensive_evaluation(cbow_trainer, "CBOW")
        
        # 7. Model comparison
        demo_model_comparison()
        
        print_section("DEMO COMPLETED SUCCESSFULLY")
        print("✓ All components demonstrated successfully!")
        print("✓ Models trained and evaluated")
        print("✓ Visualizations created")
        print("✓ Evaluation reports generated")
        print("\nGenerated files:")
        print("  • demo_corpus.txt - Sample corpus")
        print("  • skipgram_model.pth - Skip-gram model")
        print("  • cbow_model.pth - CBOW model")
        print("  • skipgram_embeddings.npy - Skip-gram embeddings")
        print("  • cbow_embeddings.npy - CBOW embeddings")
        print("  • *_evaluation_report.txt - Evaluation reports")
        print("  • *_embeddings_visualization.png - Visualizations")
        print("  • sample_similarity.txt - Similarity dataset")
        print("  • sample_analogies.txt - Analogy dataset")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 