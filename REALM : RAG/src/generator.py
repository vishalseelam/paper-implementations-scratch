"""
RAG Generator implementation
Based on Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)
Combines retrieved documents with parametric knowledge for generation
"""

import logging
from typing import List, Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BartTokenizer,
    BartForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration,
    GenerationConfig
)
from torch.utils.data import Dataset, DataLoader
import numpy as np

from .config import ModelConfig

logger = logging.getLogger(__name__)


class RAGDataset(Dataset):
    """Dataset for RAG training and inference"""
    
    def __init__(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: Optional[List[str]] = None,
        tokenizer=None,
        max_input_length: int = 512,
        max_target_length: int = 128
    ):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        assert len(questions) == len(contexts), "Questions and contexts must have same length"
        if answers is not None:
            assert len(questions) == len(answers), "Questions and answers must have same length"
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        context_docs = self.contexts[idx]
        
        # Combine contexts into single string
        context = " ".join(context_docs)
        
        # Format input: question + context
        input_text = f"question: {question} context: {context}"
        
        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt"
        )
        
        result = {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "question": question,
            "context": context
        }
        
        # Add target if available (for training)
        if self.answers is not None:
            answer = self.answers[idx]
            target_encoding = self.tokenizer(
                answer,
                truncation=True,
                padding="max_length",
                max_length=self.max_target_length,
                return_tensors="pt"
            )
            result["labels"] = target_encoding["input_ids"].squeeze()
            result["decoder_attention_mask"] = target_encoding["attention_mask"].squeeze()
            result["answer"] = answer
        
        return result


class RAGGenerator(nn.Module):
    """
    RAG Generator that combines retrieved documents with parametric knowledge
    
    Architecture:
    1. Takes question + retrieved documents as input
    2. Encodes them using encoder
    3. Generates answer using decoder
    4. Can marginalize over multiple retrieved documents
    """
    
    def __init__(self, config: ModelConfig, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device
        
        # Initialize model and tokenizer
        self.model_name = config.generator_model_name
        
        if "bart" in self.model_name.lower():
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        elif "t5" in self.model_name.lower():
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        else:
            # Generic seq2seq model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Move to device
        self.model.to(device)
        
        # Generation config
        self.generation_config = GenerationConfig(
            max_length=config.max_generation_length,
            min_length=config.min_generation_length,
            num_beams=config.num_beams,
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=config.repetition_penalty,
            do_sample=True if config.temperature > 0 else False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None
        )
        
        logger.info(f"Initialized RAGGenerator with {self.model_name}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels [batch_size, target_len]
            decoder_attention_mask: Decoder attention mask [batch_size, target_len]
            
        Returns:
            Dictionary containing loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "last_hidden_state": outputs.encoder_last_hidden_state if hasattr(outputs, 'encoder_last_hidden_state') else None
        }
    
    def generate_answer(
        self,
        questions: Union[str, List[str]],
        contexts: Union[List[str], List[List[str]]],
        batch_size: int = 1
    ) -> List[str]:
        """
        Generate answers given questions and contexts
        
        Args:
            questions: Single question or list of questions
            contexts: Context documents for each question
            batch_size: Batch size for generation
            
        Returns:
            List of generated answers
        """
        # Handle single question
        if isinstance(questions, str):
            questions = [questions]
            contexts = [contexts] if isinstance(contexts[0], str) else contexts
        
        # Ensure contexts is list of lists
        if isinstance(contexts[0], str):
            contexts = [[ctx] for ctx in contexts]
        
        # Create dataset
        dataset = RAGDataset(
            questions=questions,
            contexts=contexts,
            tokenizer=self.tokenizer,
            max_input_length=self.config.max_doc_length + self.config.max_question_length,
            max_target_length=self.config.max_answer_length
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid pickling issues
            pin_memory=True
        )
        
        # Generate answers
        self.model.eval()
        answers = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Generate
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config
                )
                
                # Decode
                batch_answers = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                answers.extend(batch_answers)
        
        return answers
    
    def generate_with_scores(
        self,
        questions: Union[str, List[str]],
        contexts: Union[List[str], List[List[str]]],
        retrieval_scores: Optional[List[List[float]]] = None,
        marginalize: bool = True
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Generate answers with confidence scores
        
        Args:
            questions: Single question or list of questions
            contexts: Context documents for each question
            retrieval_scores: Retrieval scores for each document
            marginalize: Whether to marginalize over multiple documents
            
        Returns:
            List of dictionaries with answers and scores
        """
        # Handle single question
        if isinstance(questions, str):
            questions = [questions]
            contexts = [contexts] if isinstance(contexts[0], str) else contexts
        
        # Ensure contexts is list of lists
        if isinstance(contexts[0], str):
            contexts = [[ctx] for ctx in contexts]
        
        results = []
        
        for i, (question, context_docs) in enumerate(zip(questions, contexts)):
            if marginalize and len(context_docs) > 1:
                # Marginalize over multiple documents
                doc_results = []
                
                for j, ctx in enumerate(context_docs):
                    # Generate for each document separately
                    answer = self.generate_answer([question], [[ctx]])[0]
                    
                    # Calculate generation score (simplified)
                    score = 1.0  # Placeholder - could compute actual likelihood
                    
                    # Combine with retrieval score if available
                    if retrieval_scores and len(retrieval_scores[i]) > j:
                        score *= retrieval_scores[i][j]
                    
                    doc_results.append({
                        "answer": answer,
                        "score": score,
                        "context": ctx
                    })
                
                # Select best answer (could also ensemble)
                best_result = max(doc_results, key=lambda x: x["score"])
                results.append(best_result)
            
            else:
                # Single document or no marginalization
                answer = self.generate_answer([question], [context_docs])[0]
                score = 1.0
                
                results.append({
                    "answer": answer,
                    "score": score,
                    "context": " ".join(context_docs)
                })
        
        return results
    
    def compute_generation_scores(
        self,
        questions: List[str],
        contexts: List[List[str]],
        answers: List[str]
    ) -> List[float]:
        """
        Compute generation scores for question-context-answer triples
        
        Args:
            questions: List of questions
            contexts: List of context documents for each question
            answers: List of answers
            
        Returns:
            List of generation scores
        """
        # Create dataset
        dataset = RAGDataset(
            questions=questions,
            contexts=contexts,
            answers=answers,
            tokenizer=self.tokenizer,
            max_input_length=self.config.max_doc_length + self.config.max_question_length,
            max_target_length=self.config.max_answer_length
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Process one at a time for score computation
            shuffle=False,
            num_workers=0
        )
        
        # Compute scores
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Compute negative log likelihood
                loss = outputs.loss
                score = torch.exp(-loss).item()
                scores.append(score)
        
        return scores
    
    def save_model(self, save_path: str):
        """Save the generator model"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Saved generator model to {save_path}")
    
    def load_model(self, load_path: str):
        """Load the generator model"""
        if "bart" in self.model_name.lower():
            self.model = BartForConditionalGeneration.from_pretrained(load_path)
        elif "t5" in self.model_name.lower():
            self.model = T5ForConditionalGeneration.from_pretrained(load_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
        
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        logger.info(f"Loaded generator model from {load_path}")


class REALMGenerator(RAGGenerator):
    """
    REALM-specific generator with additional features
    
    Extends RAGGenerator with REALM-specific components:
    - Salient span prediction
    - Knowledge-augmented pre-training support
    """
    
    def __init__(self, config: ModelConfig, device: str = "cuda"):
        super().__init__(config, device)
        
        # Additional REALM components
        self.span_predictor = nn.Linear(
            self.model.config.hidden_size, 
            2  # start and end positions
        )
        self.span_predictor.to(device)
        
        logger.info("Initialized REALMGenerator with span prediction")
    
    def forward_with_span_prediction(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        span_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with span prediction for REALM
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for generation
            span_labels: Span labels for salient span prediction
            
        Returns:
            Dictionary with generation and span prediction losses
        """
        # Standard generation forward pass
        gen_outputs = super().forward(input_ids, attention_mask, labels)
        
        # Span prediction
        if hasattr(gen_outputs, 'encoder_last_hidden_state') and gen_outputs['encoder_last_hidden_state'] is not None:
            encoder_hidden_states = gen_outputs['encoder_last_hidden_state']
            span_logits = self.span_predictor(encoder_hidden_states)
            
            # Compute span loss if labels provided
            span_loss = None
            if span_labels is not None:
                start_labels, end_labels = span_labels[:, 0], span_labels[:, 1]
                start_logits, end_logits = span_logits[:, :, 0], span_logits[:, :, 1]
                
                start_loss = F.cross_entropy(start_logits, start_labels)
                end_loss = F.cross_entropy(end_logits, end_labels)
                span_loss = (start_loss + end_loss) / 2
            
            gen_outputs['span_logits'] = span_logits
            gen_outputs['span_loss'] = span_loss
        
        return gen_outputs
    
    def predict_salient_spans(
        self,
        questions: List[str],
        contexts: List[List[str]]
    ) -> List[List[Tuple[int, int]]]:
        """
        Predict salient spans in contexts for each question
        
        Args:
            questions: List of questions
            contexts: List of context documents
            
        Returns:
            List of predicted spans for each question
        """
        # Create dataset
        dataset = RAGDataset(
            questions=questions,
            contexts=contexts,
            tokenizer=self.tokenizer,
            max_input_length=self.config.max_doc_length + self.config.max_question_length
        )
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Predict spans
        self.model.eval()
        predicted_spans = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                outputs = self.forward_with_span_prediction(input_ids, attention_mask)
                
                if 'span_logits' in outputs:
                    span_logits = outputs['span_logits']
                    
                    # Get start and end predictions
                    start_logits = span_logits[:, :, 0]
                    end_logits = span_logits[:, :, 1]
                    
                    start_idx = torch.argmax(start_logits, dim=-1).item()
                    end_idx = torch.argmax(end_logits, dim=-1).item()
                    
                    predicted_spans.append([(start_idx, end_idx)])
                else:
                    predicted_spans.append([])
        
        return predicted_spans


if __name__ == "__main__":
    # Example usage
    from .config import DEFAULT_CONFIG
    
    # Initialize generator
    generator = RAGGenerator(DEFAULT_CONFIG.model)
    
    # Test generation
    questions = ["What is the capital of France?", "How does photosynthesis work?"]
    contexts = [
        ["France is a country in Europe. Paris is the capital of France."],
        ["Photosynthesis is the process by which plants make food from sunlight."]
    ]
    
    # Generate answers
    answers = generator.generate_answer(questions, contexts)
    
    for question, answer in zip(questions, answers):
        print(f"Q: {question}")
        print(f"A: {answer}")
        print() 