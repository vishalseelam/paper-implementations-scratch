#!/usr/bin/env python3
"""
Basic Usage Example for REALM/RAG Implementation
This script demonstrates the core functionality of the implementation
"""

import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import REALMRAGConfig, DEFAULT_CONFIG
from src.realm_rag import REALMRAGModel
from src.trainer import create_trainer
from src.evaluator import REALMRAGEvaluator
from src.data_utils import create_sample_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating basic usage"""
    
    logger.info("=== REALM/RAG Basic Usage Example ===")
    
    # Step 1: Configuration
    logger.info("Step 1: Setting up configuration")
    config = DEFAULT_CONFIG
    
    # Modify config for quick demo
    config.model.num_retrieved_docs = 3
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.experiment.use_wandb = False
    
    print(f"Model type: RAG")
    print(f"Retrieved docs: {config.model.num_retrieved_docs}")
    print(f"Training epochs: {config.training.num_epochs}")
    
    # Step 2: Create sample data
    logger.info("Step 2: Creating sample data")
    sample_data = create_sample_data(num_examples=20, num_docs=50)
    
    print(f"Training examples: {len(sample_data['datasets']['train'])}")
    print(f"Validation examples: {len(sample_data['datasets']['val'])}")
    print(f"Test examples: {len(sample_data['datasets']['test'])}")
    print(f"Knowledge base size: {len(sample_data['knowledge_base'])}")
    
    # Step 3: Initialize model
    logger.info("Step 3: Initializing REALM/RAG model")
    model = REALMRAGModel(config, model_type="rag")
    
    # Step 4: Prepare knowledge base
    logger.info("Step 4: Preparing knowledge base")
    model.prepare_knowledge_base(sample_data['knowledge_base'])
    
    print(f"Knowledge base prepared with {len(sample_data['knowledge_base'])} documents")
    
    # Step 5: Test inference
    logger.info("Step 5: Testing inference")
    test_questions = [
        "What is the capital of Country0?",
        "How does Process1 work?",
        "What is Concept2?"
    ]
    
    print("\n--- Inference Test ---")
    results = model.retrieve_and_generate(
        test_questions,
        k=3,
        return_retrieved_docs=True
    )
    
    for i, question in enumerate(test_questions):
        print(f"\nQuestion: {question}")
        print(f"Answer: {results['answers'][i]}")
        print(f"Retrieved docs: {len(results['retrieved_docs'][i])}")
        print(f"Top doc: {results['retrieved_docs'][i][0]['text'][:100]}...")
    
    # Step 6: Training (optional - commented out for speed)
    logger.info("Step 6: Training demonstration (skipped for speed)")
    """
    trainer = create_trainer(
        model=model,
        config=config,
        train_data=sample_data['datasets']['train'],
        val_data=sample_data['datasets']['val'],
        test_data=sample_data['datasets']['test']
    )
    
    # Train the model
    training_results = trainer.train()
    print(f"Training completed! Best metric: {training_results['best_metric']:.4f}")
    """
    
    # Step 7: Evaluation
    logger.info("Step 7: Evaluation demonstration")
    evaluator = REALMRAGEvaluator(model, config)
    
    from src.data_utils import REALMRAGDataset
    test_dataset = REALMRAGDataset(sample_data['datasets']['test'], config.model)
    
    eval_results = evaluator.evaluate_dataset(test_dataset, split="test")
    
    print("\n--- Evaluation Results ---")
    print(f"Exact Match: {eval_results['exact_match']:.4f}")
    print(f"F1 Score: {eval_results['f1_score']:.4f}")
    print(f"BLEU-4: {eval_results['bleu_4']:.4f}")
    print(f"ROUGE-L F1: {eval_results['rougeL_f1']:.4f}")
    print(f"Retrieval F1: {eval_results['retrieval_f1']:.4f}")
    
    # Step 8: Save and load model
    logger.info("Step 8: Save and load model demonstration")
    
    # Save model
    save_path = "outputs/demo_model"
    model.save_model(save_path)
    print(f"Model saved to: {save_path}")
    
    # Load model
    new_model = REALMRAGModel(config, model_type="rag")
    new_model.load_model(save_path)
    print(f"Model loaded from: {save_path}")
    
    # Test loaded model
    test_answer = new_model.retrieve_and_generate("What is the capital of Country0?")
    print(f"Test answer from loaded model: {test_answer[0]}")
    
    logger.info("=== Demo completed successfully! ===")


if __name__ == "__main__":
    main() 