"""
Configuration for REALM/RAG implementation
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    # Retriever configuration
    retriever_model_name: str = "facebook/dpr-ctx_encoder-single-nq-base"
    question_encoder_model_name: str = "facebook/dpr-question_encoder-single-nq-base"
    
    # Generator configuration
    generator_model_name: str = "facebook/bart-base"
    
    # Architecture parameters
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    
    # Retrieval parameters
    num_retrieved_docs: int = 5
    max_doc_length: int = 512
    max_question_length: int = 64
    max_answer_length: int = 128
    
    # Generation parameters
    max_generation_length: int = 256
    min_generation_length: int = 8
    num_beams: int = 4
    temperature: float = 1.0
    top_p: float = 0.9
    repetition_penalty: float = 1.0


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Training parameters
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw"
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    
    # Scheduler
    lr_scheduler_type: str = "linear"
    
    # Evaluation
    eval_steps: int = 1000
    save_steps: int = 2000
    logging_steps: int = 100
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False
    
    # Checkpointing
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass
class DataConfig:
    """Configuration for data processing"""
    # Data paths
    train_data_path: str = "data/train.json"
    val_data_path: str = "data/val.json"
    test_data_path: str = "data/test.json"
    
    # Knowledge base
    knowledge_base_path: str = "data/knowledge_base.json"
    index_path: str = "data/faiss_index"
    
    # Preprocessing
    max_examples: Optional[int] = None
    seed: int = 42
    
    # Data format
    text_column: str = "text"
    question_column: str = "question"
    answer_column: str = "answer"
    context_column: str = "context"


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking"""
    # Experiment tracking
    use_wandb: bool = True
    wandb_project: str = "realm-rag"
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None
    
    # Output directories
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    logs_dir: str = "logs"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpu: int = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class REALMRAGConfig:
    """Main configuration combining all components"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        # Create output directories
        os.makedirs(self.experiment.output_dir, exist_ok=True)
        os.makedirs(self.experiment.cache_dir, exist_ok=True)
        os.makedirs(self.experiment.logs_dir, exist_ok=True)
        
        # Set random seeds
        torch.manual_seed(self.experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.experiment.seed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "experiment": self.experiment.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "REALMRAGConfig":
        """Create config from dictionary"""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            experiment=ExperimentConfig(**config_dict.get("experiment", {}))
        )


# Default configuration
DEFAULT_CONFIG = REALMRAGConfig() 