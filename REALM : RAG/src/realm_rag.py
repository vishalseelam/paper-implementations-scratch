"""
REALM/RAG Model Implementation
Combines Dense Passage Retrieval with Generation for Knowledge-Intensive NLP Tasks
Based on Guu et al., 2020 and Lewis et al., 2020
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import asdict

from .retriever import DensePassageRetriever
from .generator import RAGGenerator, REALMGenerator
from .config import REALMRAGConfig, ModelConfig

logger = logging.getLogger(__name__)


class REALMRAGModel(nn.Module):
    """
    REALM/RAG Model combining retrieval and generation
    
    Architecture:
    1. Dense Passage Retrieval to find relevant documents
    2. RAG/REALM Generation using retrieved documents
    3. End-to-end training with joint optimization
    """
    
    def __init__(
        self, 
        config: REALMRAGConfig,
        device: str = "cuda",
        model_type: str = "rag"  # "rag" or "realm"
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.model_type = model_type.lower()
        
        # Initialize retriever
        self.retriever = DensePassageRetriever(
            config.model, 
            device=device
        )
        
        # Initialize generator
        if self.model_type == "realm":
            self.generator = REALMGenerator(
                config.model,
                device=device
            )
        else:
            self.generator = RAGGenerator(
                config.model,
                device=device
            )
        
        # Training parameters
        self.joint_training = True
        self.retrieval_weight = 1.0
        self.generation_weight = 1.0
        
        logger.info(f"Initialized {model_type.upper()} model with joint training")
    
    def retrieve_and_generate(
        self,
        questions: Union[str, List[str]],
        k: Optional[int] = None,
        return_retrieved_docs: bool = False,
        marginalize: bool = True
    ) -> Union[List[str], Dict[str, any]]:
        """
        End-to-end retrieval and generation
        
        Args:
            questions: Single question or list of questions
            k: Number of documents to retrieve
            return_retrieved_docs: Whether to return retrieved documents
            marginalize: Whether to marginalize over multiple documents
            
        Returns:
            Generated answers or dictionary with answers and metadata
        """
        # Handle single question
        if isinstance(questions, str):
            questions = [questions]
        
        # Retrieve documents
        retrieved_docs = self.retriever.retrieve(questions, k=k)
        
        # Extract contexts and scores
        contexts = []
        retrieval_scores = []
        
        for question_docs in retrieved_docs:
            question_contexts = []
            question_scores = []
            
            for doc in question_docs:
                # Combine title and text if available
                if "title" in doc and "text" in doc:
                    context = f"{doc['title']} {doc['text']}"
                elif "text" in doc:
                    context = doc["text"]
                else:
                    context = str(doc)
                
                question_contexts.append(context)
                question_scores.append(doc.get("score", 0.0))
            
            contexts.append(question_contexts)
            retrieval_scores.append(question_scores)
        
        # Generate answers
        if marginalize:
            results = self.generator.generate_with_scores(
                questions, 
                contexts, 
                retrieval_scores,
                marginalize=True
            )
            answers = [result["answer"] for result in results]
        else:
            answers = self.generator.generate_answer(questions, contexts)
        
        # Return results
        if return_retrieved_docs:
            return {
                "answers": answers,
                "retrieved_docs": retrieved_docs,
                "contexts": contexts,
                "retrieval_scores": retrieval_scores
            }
        else:
            return answers
    
    def forward(
        self,
        questions: List[str],
        answers: Optional[List[str]] = None,
        contexts: Optional[List[List[str]]] = None,
        k: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            questions: List of questions
            answers: List of ground truth answers (for training)
            contexts: Pre-retrieved contexts (optional)
            k: Number of documents to retrieve (if contexts not provided)
            
        Returns:
            Dictionary containing losses and logits
        """
        batch_size = len(questions)
        
        # Retrieve documents if not provided
        if contexts is None:
            retrieved_docs = self.retriever.retrieve(questions, k=k)
            contexts = []
            retrieval_scores = []
            
            for question_docs in retrieved_docs:
                question_contexts = []
                question_scores = []
                
                for doc in question_docs:
                    if "title" in doc and "text" in doc:
                        context = f"{doc['title']} {doc['text']}"
                    elif "text" in doc:
                        context = doc["text"]
                    else:
                        context = str(doc)
                    
                    question_contexts.append(context)
                    question_scores.append(doc.get("score", 0.0))
                
                contexts.append(question_contexts)
                retrieval_scores.append(question_scores)
        
        # Prepare inputs for generator
        all_input_texts = []
        all_labels = []
        
        for i, question in enumerate(questions):
            question_contexts = contexts[i]
            
            # Create input text with all retrieved documents
            combined_context = " ".join(question_contexts)
            input_text = f"question: {question} context: {combined_context}"
            all_input_texts.append(input_text)
            
            if answers is not None:
                all_labels.append(answers[i])
        
        # Tokenize inputs
        tokenizer = self.generator.tokenizer
        input_encodings = tokenizer(
            all_input_texts,
            truncation=True,
            padding=True,
            max_length=self.config.model.max_doc_length + self.config.model.max_question_length,
            return_tensors="pt"
        )
        
        # Tokenize labels if provided
        if answers is not None:
            label_encodings = tokenizer(
                all_labels,
                truncation=True,
                padding=True,
                max_length=self.config.model.max_answer_length,
                return_tensors="pt"
            )
            labels = label_encodings["input_ids"].to(self.device)
            decoder_attention_mask = label_encodings["attention_mask"].to(self.device)
        else:
            labels = None
            decoder_attention_mask = None
        
        # Move to device
        input_ids = input_encodings["input_ids"].to(self.device)
        attention_mask = input_encodings["attention_mask"].to(self.device)
        
        # Forward pass through generator
        if self.model_type == "realm" and hasattr(self.generator, 'forward_with_span_prediction'):
            outputs = self.generator.forward_with_span_prediction(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        else:
            outputs = self.generator.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask
            )
        
        return outputs
    
    def compute_retrieval_loss(
        self,
        questions: List[str],
        positive_docs: List[str],
        negative_docs: List[List[str]],
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute contrastive loss for retrieval training
        
        Args:
            questions: List of questions
            positive_docs: List of positive documents for each question
            negative_docs: List of negative documents for each question
            temperature: Temperature for softmax
            
        Returns:
            Retrieval loss
        """
        # Encode questions
        question_embeddings = self.retriever.encode_questions(questions)
        
        # Encode positive documents
        positive_embeddings = self.retriever.encode_documents(
            [{"text": doc} for doc in positive_docs]
        )
        
        # Encode negative documents
        all_negative_docs = []
        for neg_docs in negative_docs:
            all_negative_docs.extend([{"text": doc} for doc in neg_docs])
        
        negative_embeddings = self.retriever.encode_documents(all_negative_docs)
        
        # Compute similarities
        question_embeddings = torch.from_numpy(question_embeddings).to(self.device)
        positive_embeddings = torch.from_numpy(positive_embeddings).to(self.device)
        negative_embeddings = torch.from_numpy(negative_embeddings).to(self.device)
        
        # Positive similarities
        positive_similarities = torch.sum(
            question_embeddings * positive_embeddings, dim=1
        ) / temperature
        
        # Negative similarities
        negative_similarities = torch.matmul(
            question_embeddings, negative_embeddings.t()
        ) / temperature
        
        # Contrastive loss
        logits = torch.cat([
            positive_similarities.unsqueeze(1),
            negative_similarities
        ], dim=1)
        
        labels = torch.zeros(len(questions), dtype=torch.long).to(self.device)
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def train_step(
        self,
        batch: Dict[str, any],
        update_retriever: bool = True,
        update_generator: bool = True
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Training batch
            update_retriever: Whether to update retriever parameters
            update_generator: Whether to update generator parameters
            
        Returns:
            Dictionary of losses
        """
        questions = batch["questions"]
        answers = batch.get("answers", None)
        contexts = batch.get("contexts", None)
        
        # Forward pass
        outputs = self.forward(questions, answers, contexts)
        
        # Generation loss
        generation_loss = outputs.get("loss", torch.tensor(0.0))
        
        # Retrieval loss (if applicable)
        retrieval_loss = torch.tensor(0.0)
        if update_retriever and "positive_docs" in batch:
            retrieval_loss = self.compute_retrieval_loss(
                questions,
                batch["positive_docs"],
                batch.get("negative_docs", [[] for _ in questions])
            )
        
        # Total loss
        total_loss = (
            self.generation_weight * generation_loss +
            self.retrieval_weight * retrieval_loss
        )
        
        # Backward pass
        if update_generator or update_retriever:
            total_loss.backward()
        
        return {
            "total_loss": total_loss.item(),
            "generation_loss": generation_loss.item(),
            "retrieval_loss": retrieval_loss.item()
        }
    
    def evaluate_step(
        self,
        batch: Dict[str, any]
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Single evaluation step
        
        Args:
            batch: Evaluation batch
            
        Returns:
            Dictionary of metrics and predictions
        """
        questions = batch["questions"]
        answers = batch.get("answers", None)
        
        # Generate predictions
        with torch.no_grad():
            results = self.retrieve_and_generate(
                questions,
                return_retrieved_docs=True
            )
            
            predictions = results["answers"]
            retrieved_docs = results["retrieved_docs"]
        
        # Compute metrics if ground truth available
        metrics = {}
        if answers is not None:
            # Exact match
            exact_matches = [
                1 if pred.strip().lower() == ans.strip().lower() else 0
                for pred, ans in zip(predictions, answers)
            ]
            metrics["exact_match"] = sum(exact_matches) / len(exact_matches)
            
            # Token overlap (simplified F1)
            token_f1_scores = []
            for pred, ans in zip(predictions, answers):
                pred_tokens = set(pred.strip().lower().split())
                ans_tokens = set(ans.strip().lower().split())
                
                if len(pred_tokens) == 0 and len(ans_tokens) == 0:
                    token_f1_scores.append(1.0)
                elif len(pred_tokens) == 0 or len(ans_tokens) == 0:
                    token_f1_scores.append(0.0)
                else:
                    precision = len(pred_tokens & ans_tokens) / len(pred_tokens)
                    recall = len(pred_tokens & ans_tokens) / len(ans_tokens)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    token_f1_scores.append(f1)
            
            metrics["token_f1"] = sum(token_f1_scores) / len(token_f1_scores)
        
        return {
            "metrics": metrics,
            "predictions": predictions,
            "retrieved_docs": retrieved_docs
        }
    
    def save_model(self, save_dir: str):
        """Save the complete model"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save retriever
        retriever_dir = os.path.join(save_dir, "retriever")
        self.retriever.save_state(retriever_dir)
        
        # Save generator
        generator_dir = os.path.join(save_dir, "generator")
        self.generator.save_model(generator_dir)
        
        # Save model info
        model_info = {
            "model_type": self.model_type,
            "joint_training": self.joint_training,
            "retrieval_weight": self.retrieval_weight,
            "generation_weight": self.generation_weight
        }
        
        info_path = os.path.join(save_dir, "model_info.json")
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Saved {self.model_type.upper()} model to {save_dir}")
    
    def load_model(self, load_dir: str):
        """Load the complete model"""
        # Load configuration
        config_path = os.path.join(load_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            self.config = REALMRAGConfig.from_dict(config_dict)
        
        # Load retriever
        retriever_dir = os.path.join(load_dir, "retriever")
        if os.path.exists(retriever_dir):
            self.retriever.load_state(retriever_dir)
        
        # Load generator
        generator_dir = os.path.join(load_dir, "generator")
        if os.path.exists(generator_dir):
            self.generator.load_model(generator_dir)
        
        # Load model info
        info_path = os.path.join(load_dir, "model_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                model_info = json.load(f)
            
            self.model_type = model_info.get("model_type", "rag")
            self.joint_training = model_info.get("joint_training", True)
            self.retrieval_weight = model_info.get("retrieval_weight", 1.0)
            self.generation_weight = model_info.get("generation_weight", 1.0)
        
        logger.info(f"Loaded {self.model_type.upper()} model from {load_dir}")
    
    def prepare_knowledge_base(
        self,
        documents: List[Dict],
        batch_size: int = 16,
        save_path: Optional[str] = None
    ):
        """
        Prepare knowledge base for retrieval
        
        Args:
            documents: List of document dictionaries
            batch_size: Batch size for encoding
            save_path: Path to save encoded knowledge base
        """
        logger.info(f"Preparing knowledge base with {len(documents)} documents...")
        
        # Encode documents
        embeddings = self.retriever.encode_documents(
            documents,
            batch_size=batch_size
        )
        
        # Build index
        index = self.retriever.build_index(embeddings)
        
        # Save if path provided
        if save_path:
            self.retriever.save_state(save_path)
        
        logger.info("Knowledge base preparation complete")
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information"""
        return {
            "model_type": self.model_type,
            "config": self.config.to_dict(),
            "retriever_info": {
                "num_documents": len(self.retriever.documents) if self.retriever.documents else 0,
                "embedding_dim": self.retriever.doc_embeddings.shape[1] if self.retriever.doc_embeddings is not None else 0,
                "index_size": self.retriever.index.ntotal if self.retriever.index else 0
            },
            "generator_info": {
                "model_name": self.generator.model_name,
                "vocab_size": len(self.generator.tokenizer),
                "max_length": self.config.model.max_generation_length
            }
        }


def create_sample_qa_data(size: int = 100) -> List[Dict]:
    """
    Create sample QA data for testing
    
    Args:
        size: Number of QA pairs to create
        
    Returns:
        List of QA dictionaries
    """
    qa_pairs = []
    
    # Sample questions and answers
    samples = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "context": "France is a country in Western Europe. Its capital and largest city is Paris."
        },
        {
            "question": "How does photosynthesis work?",
            "answer": "Plants use sunlight, water, and carbon dioxide to make glucose and oxygen.",
            "context": "Photosynthesis is the process by which plants and other organisms use sunlight to synthesize foods from carbon dioxide and water."
        },
        {
            "question": "What is the speed of light?",
            "answer": "299,792,458 meters per second",
            "context": "The speed of light in vacuum is a universal physical constant denoted by c."
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare",
            "context": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career."
        },
        {
            "question": "What is the largest planet in our solar system?",
            "answer": "Jupiter",
            "context": "Jupiter is the fifth planet from the Sun and the largest in the Solar System."
        }
    ]
    
    for i in range(size):
        sample = samples[i % len(samples)]
        qa_pairs.append({
            "id": str(i),
            "question": sample["question"],
            "answer": sample["answer"],
            "context": sample["context"]
        })
    
    return qa_pairs


if __name__ == "__main__":
    # Example usage
    from .config import DEFAULT_CONFIG
    
    # Create sample data
    qa_data = create_sample_qa_data(10)
    documents = [{"text": item["context"], "id": item["id"]} for item in qa_data]
    
    # Initialize model
    model = REALMRAGModel(DEFAULT_CONFIG, model_type="rag")
    
    # Prepare knowledge base
    model.prepare_knowledge_base(documents)
    
    # Test retrieval and generation
    questions = [item["question"] for item in qa_data[:3]]
    results = model.retrieve_and_generate(questions, return_retrieved_docs=True)
    
    print("=== REALM/RAG Model Test ===")
    for i, question in enumerate(questions):
        print(f"\nQuestion: {question}")
        print(f"Answer: {results['answers'][i]}")
        print(f"Retrieved: {len(results['retrieved_docs'][i])} documents")
        print(f"Top doc: {results['retrieved_docs'][i][0]['text'][:100]}...") 