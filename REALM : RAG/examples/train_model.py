#!/usr/bin/env python3
"""
Training Script for REALM/RAG Model
Complete training pipeline with evaluation and checkpointing
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import REALMRAGConfig, DEFAULT_CONFIG
from src.realm_rag import REALMRAGModel
from src.trainer import create_trainer
from src.evaluator import REALMRAGEvaluator
from src.data_utils import create_sample_data, prepare_data_for_training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train REALM/RAG model")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, help="Path to training data")
    parser.add_argument("--val_data", type=str, help="Path to validation data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--knowledge_base", type=str, help="Path to knowledge base")
    parser.add_argument("--use_sample_data", action="store_true", help="Use sample data for demo")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of sample examples")
    parser.add_argument("--num_docs", type=int, default=1000, help="Number of knowledge base documents")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="rag", choices=["rag", "realm"], help="Model type")
    parser.add_argument("--generator_model", type=str, default="facebook/bart-base", help="Generator model")
    parser.add_argument("--retriever_model", type=str, default="facebook/dpr-ctx_encoder-single-nq-base", help="Retriever model")
    parser.add_argument("--num_retrieved_docs", type=int, default=5, help="Number of retrieved documents")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    # Experiment arguments
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--run_name", type=str, help="Run name for experiment tracking")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="realm-rag", help="W&B project name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Evaluation arguments
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--model_path", type=str, help="Path to trained model for evaluation")
    
    return parser.parse_args()


def create_config(args) -> REALMRAGConfig:
    """Create configuration from arguments"""
    config = DEFAULT_CONFIG
    
    # Model configuration
    config.model.generator_model_name = args.generator_model
    config.model.retriever_model_name = args.retriever_model
    config.model.num_retrieved_docs = args.num_retrieved_docs
    
    # Training configuration
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.num_epochs = args.num_epochs
    config.training.warmup_steps = args.warmup_steps
    config.training.max_grad_norm = args.max_grad_norm
    config.training.weight_decay = args.weight_decay
    
    # Experiment configuration
    config.experiment.output_dir = args.output_dir
    config.experiment.run_name = args.run_name
    config.experiment.use_wandb = args.use_wandb
    config.experiment.wandb_project = args.wandb_project
    config.experiment.seed = args.seed
    
    return config


def load_data(args) -> Dict[str, Any]:
    """Load training data"""
    if args.use_sample_data:
        logger.info("Creating sample data")
        return create_sample_data(args.num_examples, args.num_docs)
    else:
        logger.info("Loading data from files")
        return prepare_data_for_training(
            config=DEFAULT_CONFIG.model,
            train_path=args.train_data,
            val_path=args.val_data,
            test_path=args.test_data,
            kb_path=args.knowledge_base
        )


def train_model(config: REALMRAGConfig, data: Dict[str, Any], model_type: str) -> REALMRAGModel:
    """Train the model"""
    logger.info(f"Initializing {model_type.upper()} model")
    
    # Create model
    model = REALMRAGModel(config, model_type=model_type)
    
    # Prepare knowledge base
    logger.info("Preparing knowledge base")
    model.prepare_knowledge_base(data['knowledge_base'])
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = create_trainer(
        model=model,
        config=config,
        train_data=data['datasets'].get('train'),
        val_data=data['datasets'].get('val'),
        test_data=data['datasets'].get('test')
    )
    
    # Train
    logger.info("Starting training")
    training_results = trainer.train()
    
    # Log results
    logger.info(f"Training completed!")
    logger.info(f"Best metric: {training_results['best_metric']:.4f}")
    logger.info(f"Total steps: {training_results['total_steps']}")
    
    return model


def evaluate_model(model: REALMRAGModel, config: REALMRAGConfig, data: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate the model"""
    logger.info("Starting evaluation")
    
    # Create evaluator
    evaluator = REALMRAGEvaluator(model, config)
    
    # Evaluate on test set
    from src.data_utils import REALMRAGDataset
    test_dataset = REALMRAGDataset(data['datasets']['test'], config.model)
    
    eval_results = evaluator.evaluate_dataset(test_dataset, split="test")
    
    # Log results
    logger.info("=== EVALUATION RESULTS ===")
    logger.info(f"Exact Match: {eval_results['exact_match']:.4f}")
    logger.info(f"F1 Score: {eval_results['f1_score']:.4f}")
    logger.info(f"BLEU-4: {eval_results['bleu_4']:.4f}")
    logger.info(f"ROUGE-L F1: {eval_results['rougeL_f1']:.4f}")
    logger.info(f"Retrieval F1: {eval_results['retrieval_f1']:.4f}")
    logger.info(f"Mean Reciprocal Rank: {eval_results['mean_reciprocal_rank']:.4f}")
    
    # Error analysis
    error_analysis = evaluator.error_analysis(split="test", top_k=5)
    logger.info(f"Error patterns: {error_analysis['error_patterns']}")
    
    # Create evaluation report
    report_path = evaluator.create_evaluation_report()
    logger.info(f"Evaluation report saved to: {report_path}")
    
    return eval_results


def main():
    """Main training function"""
    args = parse_args()
    
    logger.info("=== REALM/RAG Training Script ===")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create configuration
    config = create_config(args)
    
    # Load data
    data = load_data(args)
    
    logger.info(f"Data loaded:")
    logger.info(f"  Training examples: {len(data['datasets'].get('train', []))}")
    logger.info(f"  Validation examples: {len(data['datasets'].get('val', []))}")
    logger.info(f"  Test examples: {len(data['datasets'].get('test', []))}")
    logger.info(f"  Knowledge base size: {len(data['knowledge_base'])}")
    
    if args.eval_only:
        # Load and evaluate existing model
        if not args.model_path:
            raise ValueError("Model path required for evaluation")
        
        logger.info(f"Loading model from: {args.model_path}")
        model = REALMRAGModel(config, model_type=args.model_type)
        model.load_model(args.model_path)
        
        # Evaluate
        eval_results = evaluate_model(model, config, data)
        
    else:
        # Train model
        model = train_model(config, data, args.model_type)
        
        # Evaluate
        eval_results = evaluate_model(model, config, data)
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, "final_model")
        model.save_model(final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
    
    logger.info("=== Training/Evaluation completed! ===")


if __name__ == "__main__":
    main() 