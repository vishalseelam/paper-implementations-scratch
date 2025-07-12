"""
Dense Passage Retriever implementation
Based on Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., 2020)
Used in both REALM and RAG architectures
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModel,
    DPRContextEncoder,
    DPRQuestionEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoderTokenizer
)
import faiss
from tqdm import tqdm

from .config import ModelConfig

logger = logging.getLogger(__name__)


class DocumentDataset(Dataset):
    """Dataset for document passages"""
    
    def __init__(self, documents: List[Dict], tokenizer, max_length: int = 512):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        
        # Combine title and text if both exist
        if "title" in doc and "text" in doc:
            text = f"{doc['title']} {doc['text']}"
        elif "text" in doc:
            text = doc["text"]
        else:
            text = str(doc)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "doc_id": idx,
            "text": text
        }


class QuestionDataset(Dataset):
    """Dataset for questions"""
    
    def __init__(self, questions: List[str], tokenizer, max_length: int = 64):
        self.questions = questions
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        
        encoding = self.tokenizer(
            question,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "question": question
        }


class DensePassageRetriever(nn.Module):
    """
    Dense Passage Retriever for REALM/RAG
    
    Encodes documents and questions into dense vectors and performs 
    similarity-based retrieval using FAISS index.
    """
    
    def __init__(self, config: ModelConfig, device: str = "cuda"):
        super().__init__()
        self.config = config
        self.device = device
        
        # Initialize encoders
        self.ctx_encoder = DPRContextEncoder.from_pretrained(
            config.retriever_model_name
        )
        self.question_encoder = DPRQuestionEncoder.from_pretrained(
            config.question_encoder_model_name
        )
        
        # Initialize tokenizers
        self.ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            config.retriever_model_name
        )
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            config.question_encoder_model_name
        )
        
        # Move to device
        self.ctx_encoder.to(device)
        self.question_encoder.to(device)
        
        # FAISS index
        self.index = None
        self.documents = None
        self.doc_embeddings = None
        
        logger.info(f"Initialized DensePassageRetriever with device: {device}")
    
    def encode_documents(
        self, 
        documents: List[Dict], 
        batch_size: int = 16,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Encode documents into dense vectors
        
        Args:
            documents: List of document dictionaries
            batch_size: Batch size for encoding
            save_path: Path to save encoded embeddings
            
        Returns:
            Document embeddings as numpy array
        """
        logger.info(f"Encoding {len(documents)} documents...")
        
        # Create dataset and dataloader
        dataset = DocumentDataset(
            documents, 
            self.ctx_tokenizer, 
            self.config.max_doc_length
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Encode documents
        self.ctx_encoder.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding documents"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Get embeddings
                outputs = self.ctx_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Extract pooled embeddings
                batch_embeddings = outputs.pooler_output
                embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings = np.concatenate(embeddings, axis=0)
        
        # Save embeddings if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, embeddings)
            logger.info(f"Saved document embeddings to {save_path}")
        
        self.documents = documents
        self.doc_embeddings = embeddings
        
        logger.info(f"Encoded documents shape: {embeddings.shape}")
        return embeddings
    
    def encode_questions(
        self, 
        questions: List[str], 
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode questions into dense vectors
        
        Args:
            questions: List of question strings
            batch_size: Batch size for encoding
            
        Returns:
            Question embeddings as numpy array
        """
        logger.info(f"Encoding {len(questions)} questions...")
        
        # Create dataset and dataloader
        dataset = QuestionDataset(
            questions, 
            self.question_tokenizer, 
            self.config.max_question_length
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Encode questions
        self.question_encoder.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding questions"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Get embeddings
                outputs = self.question_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Extract pooled embeddings
                batch_embeddings = outputs.pooler_output
                embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings = np.concatenate(embeddings, axis=0)
        
        logger.info(f"Encoded questions shape: {embeddings.shape}")
        return embeddings
    
    def build_index(
        self, 
        doc_embeddings: Optional[np.ndarray] = None,
        index_path: Optional[str] = None
    ) -> faiss.Index:
        """
        Build FAISS index for efficient retrieval
        
        Args:
            doc_embeddings: Document embeddings (uses self.doc_embeddings if None)
            index_path: Path to save the index
            
        Returns:
            FAISS index
        """
        if doc_embeddings is None:
            doc_embeddings = self.doc_embeddings
        
        if doc_embeddings is None:
            raise ValueError("No document embeddings available. Run encode_documents first.")
        
        logger.info(f"Building FAISS index for {len(doc_embeddings)} documents...")
        
        # Create index
        embedding_dim = doc_embeddings.shape[1]
        
        # Use IndexFlatIP for inner product similarity
        index = faiss.IndexFlatIP(embedding_dim)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(doc_embeddings)
        
        # Add embeddings to index
        index.add(doc_embeddings.astype(np.float32))
        
        # Save index if path provided
        if index_path:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(index, index_path)
            logger.info(f"Saved FAISS index to {index_path}")
        
        self.index = index
        logger.info(f"Built FAISS index with {index.ntotal} documents")
        return index
    
    def load_index(self, index_path: str) -> faiss.Index:
        """Load FAISS index from file"""
        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)
        return self.index
    
    def retrieve(
        self, 
        questions: Union[str, List[str]], 
        k: int = None
    ) -> List[List[Dict]]:
        """
        Retrieve top-k documents for given questions
        
        Args:
            questions: Single question or list of questions
            k: Number of documents to retrieve (uses config default if None)
            
        Returns:
            List of retrieved documents for each question
        """
        if k is None:
            k = self.config.num_retrieved_docs
        
        if self.index is None:
            raise ValueError("No FAISS index available. Run build_index first.")
        
        if self.documents is None:
            raise ValueError("No documents available. Run encode_documents first.")
        
        # Handle single question
        if isinstance(questions, str):
            questions = [questions]
        
        # Encode questions
        question_embeddings = self.encode_questions(questions)
        
        # Normalize question embeddings
        faiss.normalize_L2(question_embeddings)
        
        # Search
        scores, doc_indices = self.index.search(
            question_embeddings.astype(np.float32), k
        )
        
        # Prepare results
        results = []
        for i, question in enumerate(questions):
            question_results = []
            for j in range(k):
                doc_idx = doc_indices[i][j]
                score = scores[i][j]
                
                if doc_idx < len(self.documents):
                    doc = self.documents[doc_idx].copy()
                    doc["score"] = float(score)
                    doc["rank"] = j + 1
                    question_results.append(doc)
            
            results.append(question_results)
        
        return results
    
    def save_state(self, save_dir: str):
        """Save retriever state"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save documents
        if self.documents is not None:
            with open(os.path.join(save_dir, "documents.json"), "w") as f:
                json.dump(self.documents, f, indent=2)
        
        # Save embeddings
        if self.doc_embeddings is not None:
            np.save(os.path.join(save_dir, "doc_embeddings.npy"), self.doc_embeddings)
        
        # Save index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(save_dir, "faiss_index"))
        
        logger.info(f"Saved retriever state to {save_dir}")
    
    def load_state(self, save_dir: str):
        """Load retriever state"""
        # Load documents
        docs_path = os.path.join(save_dir, "documents.json")
        if os.path.exists(docs_path):
            with open(docs_path, "r") as f:
                self.documents = json.load(f)
        
        # Load embeddings
        emb_path = os.path.join(save_dir, "doc_embeddings.npy")
        if os.path.exists(emb_path):
            self.doc_embeddings = np.load(emb_path)
        
        # Load index
        index_path = os.path.join(save_dir, "faiss_index")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        logger.info(f"Loaded retriever state from {save_dir}")


def create_sample_knowledge_base(size: int = 1000) -> List[Dict]:
    """
    Create a sample knowledge base for testing
    
    Args:
        size: Number of documents to create
        
    Returns:
        List of sample documents
    """
    documents = []
    
    # Sample topics and facts
    topics = [
        "Science", "History", "Geography", "Literature", "Mathematics",
        "Technology", "Medicine", "Sports", "Art", "Music"
    ]
    
    for i in range(size):
        topic = topics[i % len(topics)]
        doc = {
            "id": str(i),
            "title": f"{topic} Document {i}",
            "text": f"This is a sample document about {topic.lower()}. "
                   f"It contains information that might be useful for "
                   f"answering questions related to {topic.lower()}. "
                   f"Document ID: {i}",
            "topic": topic
        }
        documents.append(doc)
    
    return documents


if __name__ == "__main__":
    # Example usage
    from .config import DEFAULT_CONFIG
    
    # Create sample knowledge base
    documents = create_sample_knowledge_base(100)
    
    # Initialize retriever
    retriever = DensePassageRetriever(DEFAULT_CONFIG.model)
    
    # Encode documents and build index
    embeddings = retriever.encode_documents(documents)
    index = retriever.build_index(embeddings)
    
    # Test retrieval
    questions = [
        "What is science?",
        "Tell me about history",
        "What is mathematics?"
    ]
    
    results = retriever.retrieve(questions, k=3)
    
    for i, (question, docs) in enumerate(zip(questions, results)):
        print(f"\nQuestion {i+1}: {question}")
        for doc in docs:
            print(f"  - {doc['title']} (score: {doc['score']:.4f})") 