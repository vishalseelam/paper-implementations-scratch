"""
REALM/RAG Implementation
Retrieval-Augmented Language Model and Retrieval-Augmented Generation
Based on Guu et al., 2020 and Lewis et al., 2020
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"

from .retriever import DensePassageRetriever
from .generator import RAGGenerator
from .realm_rag import REALMRAGModel
from .trainer import REALMRAGTrainer
from .evaluator import REALMRAGEvaluator

__all__ = [
    "DensePassageRetriever",
    "RAGGenerator", 
    "REALMRAGModel",
    "REALMRAGTrainer",
    "REALMRAGEvaluator"
] 