"""
Training Pipeline for REALM/RAG Models
Supports joint optimization of retrieval and generation components
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import wandb
from accelerate import Accelerator

from .realm_rag import REALMRAGModel
from .config import REALMRAGConfig
from .data_utils import REALMRAGDataset, create_dataloader

logger = logging.getLogger(__name__)


class REALMRAGTrainer:
    """
    Trainer for REALM/RAG models with joint optimization
    
    Features:
    - Joint training of retrieval and generation
    - Gradient accumulation
    - Mixed precision training
    - Checkpointing and early stopping
    - Experiment tracking with W&B
    """
    
    def __init__(
        self,
        model: REALMRAGModel,
        config: REALMRAGConfig,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        accelerator: Optional[Accelerator] = None
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        # Initialize accelerator
        self.accelerator = accelerator or Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision="fp16" if config.training.fp16 else "bf16" if config.training.bf16 else "no"
        )
        
        # Move model to device
        self.model = self.accelerator.prepare(model)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf') if not config.training.greater_is_better else float('-inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.training_metrics = []
        self.validation_metrics = []
        
        # Initialize W&B if enabled
        if config.experiment.use_wandb:
            self._init_wandb()
        
        logger.info(f"Initialized trainer with {self.accelerator.num_processes} processes")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter-specific learning rates"""
        config = self.config.training
        
        # Separate parameters for retriever and generator
        retriever_params = []
        generator_params = []
        
        for name, param in self.model.named_parameters():
            if 'retriever' in name:
                retriever_params.append(param)
            elif 'generator' in name:
                generator_params.append(param)
            else:
                generator_params.append(param)  # Default to generator
        
        # Create parameter groups with different learning rates
        param_groups = []
        
        if retriever_params:
            param_groups.append({
                'params': retriever_params,
                'lr': config.learning_rate * 0.1,  # Lower LR for retriever
                'weight_decay': config.weight_decay
            })
        
        if generator_params:
            param_groups.append({
                'params': generator_params,
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay
            })
        
        # Create optimizer
        if config.optimizer.lower() == 'adamw':
            optimizer = AdamW(
                param_groups,
                lr=config.learning_rate,
                eps=config.adam_epsilon,
                betas=(config.adam_beta1, config.adam_beta2),
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler"""
        config = self.config.training
        
        if config.lr_scheduler_type == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=config.warmup_steps,
                num_training_steps=num_training_steps
            )
        elif config.lr_scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=config.learning_rate * 0.1
            )
        else:
            logger.warning(f"Unknown scheduler type: {config.lr_scheduler_type}")
            self.scheduler = None
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        config = self.config.experiment
        
        if self.accelerator.is_main_process:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.run_name,
                config=self.config.to_dict()
            )
            
            # Watch model
            wandb.watch(self.model, log="all", log_freq=100)
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop
        
        Returns:
            Dictionary containing training results
        """
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")
        
        config = self.config.training
        
        # Create data loader
        train_dataloader = create_dataloader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # Prepare dataloader with accelerator
        train_dataloader = self.accelerator.prepare(train_dataloader)
        
        # Calculate total training steps
        num_training_steps = len(train_dataloader) * config.num_epochs
        
        # Create scheduler
        self._create_scheduler(num_training_steps)
        if self.scheduler:
            self.scheduler = self.accelerator.prepare(self.scheduler)
        
        logger.info(f"Starting training for {config.num_epochs} epochs")
        logger.info(f"Total training steps: {num_training_steps}")
        
        # Training loop
        for epoch in range(config.num_epochs):
            self.epoch = epoch
            
            # Train one epoch
            train_metrics = self._train_epoch(train_dataloader)
            
            # Validation
            val_metrics = {}
            if self.val_dataset is not None:
                val_metrics = self._evaluate_epoch(self.val_dataset)
            
            # Log metrics
            self._log_metrics(train_metrics, val_metrics, epoch)
            
            # Save checkpoint
            if (epoch + 1) % (config.save_steps // len(train_dataloader)) == 0:
                self._save_checkpoint(epoch, val_metrics)
            
            # Early stopping
            if self._should_stop(val_metrics):
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation
        final_metrics = {}
        if self.test_dataset is not None:
            final_metrics = self._evaluate_epoch(self.test_dataset, split="test")
        
        # Save final model
        self._save_final_model()
        
        return {
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "final_metrics": final_metrics,
            "best_metric": self.best_metric,
            "total_steps": self.global_step
        }
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        config = self.config.training
        
        total_loss = 0.0
        total_generation_loss = 0.0
        total_retrieval_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Training Epoch {self.epoch}",
            disable=not self.accelerator.is_main_process
        )
        
        for batch in progress_bar:
            with self.accelerator.accumulate(self.model):
                # Forward pass
                batch_losses = self.model.train_step(batch)
                
                loss = batch_losses["total_loss"]
                total_loss += loss
                total_generation_loss += batch_losses["generation_loss"]
                total_retrieval_loss += batch_losses["retrieval_loss"]
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if config.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        config.max_grad_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'gen_loss': f"{batch_losses['generation_loss']:.4f}",
                    'ret_loss': f"{batch_losses['retrieval_loss']:.4f}"
                })
                
                # Log to W&B
                if (self.global_step % config.logging_steps == 0 and 
                    self.accelerator.is_main_process):
                    self._log_step_metrics(batch_losses)
        
        # Calculate average metrics
        avg_metrics = {
            "train_loss": total_loss / num_batches,
            "train_generation_loss": total_generation_loss / num_batches,
            "train_retrieval_loss": total_retrieval_loss / num_batches,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
        
        self.training_metrics.append(avg_metrics)
        return avg_metrics
    
    def _evaluate_epoch(self, dataset: Dataset, split: str = "validation") -> Dict[str, float]:
        """Evaluate one epoch"""
        self.model.eval()
        
        # Create data loader
        dataloader = create_dataloader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4
        )
        dataloader = self.accelerator.prepare(dataloader)
        
        total_metrics = {}
        all_predictions = []
        all_targets = []
        num_batches = 0
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Evaluating {split}",
            disable=not self.accelerator.is_main_process
        )
        
        with torch.no_grad():
            for batch in progress_bar:
                # Evaluation step
                eval_results = self.model.evaluate_step(batch)
                
                # Accumulate metrics
                for key, value in eval_results["metrics"].items():
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value
                
                # Collect predictions and targets
                all_predictions.extend(eval_results["predictions"])
                if "answers" in batch:
                    all_targets.extend(batch["answers"])
                
                num_batches += 1
        
        # Calculate average metrics
        avg_metrics = {
            f"{split}_{key}": value / num_batches
            for key, value in total_metrics.items()
        }
        
        # Additional metrics
        if all_targets:
            # Calculate corpus-level metrics
            corpus_metrics = self._calculate_corpus_metrics(all_predictions, all_targets)
            for key, value in corpus_metrics.items():
                avg_metrics[f"{split}_{key}"] = value
        
        if split == "validation":
            self.validation_metrics.append(avg_metrics)
        
        return avg_metrics
    
    def _calculate_corpus_metrics(
        self,
        predictions: List[str],
        targets: List[str]
    ) -> Dict[str, float]:
        """Calculate corpus-level metrics"""
        # Exact Match
        exact_matches = sum(
            1 for p, t in zip(predictions, targets)
            if p.strip().lower() == t.strip().lower()
        )
        em_score = exact_matches / len(predictions)
        
        # Token F1
        f1_scores = []
        for pred, target in zip(predictions, targets):
            pred_tokens = set(pred.strip().lower().split())
            target_tokens = set(target.strip().lower().split())
            
            if not pred_tokens and not target_tokens:
                f1_scores.append(1.0)
            elif not pred_tokens or not target_tokens:
                f1_scores.append(0.0)
            else:
                precision = len(pred_tokens & target_tokens) / len(pred_tokens)
                recall = len(pred_tokens & target_tokens) / len(target_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
        
        return {
            "exact_match": em_score,
            "f1_score": sum(f1_scores) / len(f1_scores),
            "num_examples": len(predictions)
        }
    
    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int
    ):
        """Log metrics to console and W&B"""
        # Console logging
        log_str = f"Epoch {epoch}: "
        log_str += f"train_loss={train_metrics['train_loss']:.4f}, "
        
        if val_metrics:
            main_val_metric = val_metrics.get('validation_f1_score', 
                                            val_metrics.get('validation_exact_match', 0))
            log_str += f"val_f1={main_val_metric:.4f}"
        
        logger.info(log_str)
        
        # W&B logging
        if self.config.experiment.use_wandb and self.accelerator.is_main_process:
            wandb_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            wandb.log(wandb_metrics)
    
    def _log_step_metrics(self, step_metrics: Dict[str, float]):
        """Log step-level metrics"""
        if self.config.experiment.use_wandb and self.accelerator.is_main_process:
            wandb_metrics = {
                **step_metrics,
                "step": self.global_step,
                "learning_rate": self.optimizer.param_groups[0]['lr']
            }
            wandb.log(wandb_metrics)
    
    def _save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint_dir = os.path.join(
            self.config.experiment.output_dir,
            f"checkpoint-epoch-{epoch}"
        )
        
        # Save model
        self.accelerator.save_state(checkpoint_dir)
        
        # Save additional info
        checkpoint_info = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "val_metrics": val_metrics,
            "config": self.config.to_dict()
        }
        
        with open(os.path.join(checkpoint_dir, "checkpoint_info.json"), "w") as f:
            json.dump(checkpoint_info, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def _save_final_model(self):
        """Save final model"""
        if not self.accelerator.is_main_process:
            return
        
        final_model_dir = os.path.join(
            self.config.experiment.output_dir,
            "final_model"
        )
        
        # Unwrap model for saving
        model = self.accelerator.unwrap_model(self.model)
        model.save_model(final_model_dir)
        
        logger.info(f"Saved final model to {final_model_dir}")
    
    def _should_stop(self, val_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early"""
        if not val_metrics:
            return False
        
        config = self.config.training
        main_metric = val_metrics.get(config.metric_for_best_model, 0)
        
        # Check if metric improved
        improved = False
        if config.greater_is_better:
            if main_metric > self.best_metric:
                self.best_metric = main_metric
                improved = True
        else:
            if main_metric < self.best_metric:
                self.best_metric = main_metric
                improved = True
        
        # Update patience counter
        if improved:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Early stopping (if patience is configured)
        patience_limit = getattr(config, 'patience', None)
        if patience_limit and self.patience_counter >= patience_limit:
            return True
        
        return False
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        
        # Load additional info
        info_path = os.path.join(checkpoint_path, "checkpoint_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                checkpoint_info = json.load(f)
            
            self.epoch = checkpoint_info.get("epoch", 0)
            self.global_step = checkpoint_info.get("global_step", 0)
            self.best_metric = checkpoint_info.get("best_metric", 
                                                  float('inf') if not self.config.training.greater_is_better else float('-inf'))
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def evaluate(self, dataset: Dataset, split: str = "test") -> Dict[str, float]:
        """Evaluate model on dataset"""
        return self._evaluate_epoch(dataset, split)
    
    def predict(self, questions: List[str], k: int = 5) -> List[str]:
        """Generate predictions for questions"""
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model.retrieve_and_generate(questions, k=k)
        
        return predictions


def create_trainer(
    model: REALMRAGModel,
    config: REALMRAGConfig,
    train_data: Optional[List[Dict]] = None,
    val_data: Optional[List[Dict]] = None,
    test_data: Optional[List[Dict]] = None
) -> REALMRAGTrainer:
    """
    Create trainer with datasets
    
    Args:
        model: REALM/RAG model
        config: Configuration
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        
    Returns:
        Configured trainer
    """
    # Create datasets
    train_dataset = None
    val_dataset = None
    test_dataset = None
    
    if train_data:
        train_dataset = REALMRAGDataset(train_data, config.model)
    
    if val_data:
        val_dataset = REALMRAGDataset(val_data, config.model)
    
    if test_data:
        test_dataset = REALMRAGDataset(test_data, config.model)
    
    # Create trainer
    trainer = REALMRAGTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )
    
    return trainer


if __name__ == "__main__":
    # Example usage
    from .config import DEFAULT_CONFIG
    from .realm_rag import REALMRAGModel, create_sample_qa_data
    
    # Create sample data
    qa_data = create_sample_qa_data(100)
    documents = [{"text": item["context"], "id": item["id"]} for item in qa_data]
    
    # Split data
    train_data = qa_data[:80]
    val_data = qa_data[80:90]
    test_data = qa_data[90:]
    
    # Create model
    model = REALMRAGModel(DEFAULT_CONFIG)
    
    # Prepare knowledge base
    model.prepare_knowledge_base(documents)
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        config=DEFAULT_CONFIG,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data
    )
    
    # Train model
    results = trainer.train()
    
    print("Training completed!")
    print(f"Best metric: {results['best_metric']:.4f}")
    print(f"Total steps: {results['total_steps']}") 