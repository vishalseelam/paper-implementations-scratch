"""
Data utilities for REALM/RAG training
Includes dataset classes and data loading functions
"""

import json
import logging
from typing import List, Dict, Optional, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import Dataset as HFDataset, load_dataset

from .config import ModelConfig

logger = logging.getLogger(__name__)


class REALMRAGDataset(Dataset):
    """
    Dataset for REALM/RAG training and evaluation
    
    Handles:
    - Question-answer pairs
    - Context documents
    - Retrieval targets
    """
    
    def __init__(
        self,
        data: List[Dict],
        config: ModelConfig,
        tokenizer=None,
        include_contexts: bool = True,
        max_contexts: int = 5
    ):
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.include_contexts = include_contexts
        self.max_contexts = max_contexts
        
        # Validate data format
        self._validate_data()
        
        logger.info(f"Loaded dataset with {len(self.data)} examples")
    
    def _validate_data(self):
        """Validate data format"""
        if not self.data:
            raise ValueError("Empty dataset")
        
        required_fields = ["question"]
        optional_fields = ["answer", "context", "contexts", "positive_docs", "negative_docs"]
        
        for i, item in enumerate(self.data):
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Missing required field '{field}' in item {i}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Basic fields
        result = {
            "id": item.get("id", str(idx)),
            "question": item["question"],
        }
        
        # Add answer if available
        if "answer" in item:
            result["answer"] = item["answer"]
        
        # Add contexts
        if self.include_contexts:
            contexts = []
            
            # From single context field
            if "context" in item:
                contexts.append(item["context"])
            
            # From multiple contexts field
            if "contexts" in item:
                contexts.extend(item["contexts"][:self.max_contexts])
            
            # From positive documents
            if "positive_docs" in item:
                contexts.extend(item["positive_docs"][:self.max_contexts])
            
            # Ensure we have at least one context
            if not contexts:
                contexts = [""]  # Empty context
            
            result["contexts"] = contexts[:self.max_contexts]
        
        # Add retrieval training data
        if "positive_docs" in item:
            result["positive_docs"] = item["positive_docs"]
        
        if "negative_docs" in item:
            result["negative_docs"] = item["negative_docs"]
        
        return result
    
    def get_questions(self) -> List[str]:
        """Get all questions"""
        return [item["question"] for item in self.data]
    
    def get_answers(self) -> List[str]:
        """Get all answers"""
        return [item.get("answer", "") for item in self.data]
    
    def get_contexts(self) -> List[List[str]]:
        """Get all contexts"""
        contexts = []
        for item in self.data:
            item_contexts = []
            
            if "context" in item:
                item_contexts.append(item["context"])
            
            if "contexts" in item:
                item_contexts.extend(item["contexts"][:self.max_contexts])
            
            if not item_contexts:
                item_contexts = [""]
            
            contexts.append(item_contexts[:self.max_contexts])
        
        return contexts


class KnowledgeBaseDataset(Dataset):
    """
    Dataset for knowledge base documents
    Used for encoding and indexing
    """
    
    def __init__(
        self,
        documents: List[Dict],
        tokenizer=None,
        max_length: int = 512
    ):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Loaded knowledge base with {len(self.documents)} documents")
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        doc = self.documents[idx]
        
        # Extract text
        if "text" in doc:
            text = doc["text"]
        elif "content" in doc:
            text = doc["content"]
        else:
            text = str(doc)
        
        # Add title if available
        if "title" in doc:
            text = f"{doc['title']} {text}"
        
        return {
            "id": doc.get("id", str(idx)),
            "text": text,
            "title": doc.get("title", ""),
            "metadata": doc.get("metadata", {})
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for REALM/RAG data
    
    Args:
        batch: List of data samples
        
    Returns:
        Collated batch dictionary
    """
    # Collect all fields
    result = {}
    
    # Simple fields (lists)
    for key in ["id", "question", "answer"]:
        if key in batch[0]:
            result[key] = [item[key] for item in batch]
    
    # Context fields (list of lists)
    if "contexts" in batch[0]:
        result["contexts"] = [item["contexts"] for item in batch]
    
    # Retrieval training fields
    if "positive_docs" in batch[0]:
        result["positive_docs"] = [item["positive_docs"] for item in batch]
    
    if "negative_docs" in batch[0]:
        result["negative_docs"] = [item["negative_docs"] for item in batch]
    
    return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for REALM/RAG training
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )


def load_qa_dataset(
    data_path: str,
    split: str = "train",
    max_examples: Optional[int] = None
) -> List[Dict]:
    """
    Load QA dataset from file
    
    Args:
        data_path: Path to data file
        split: Data split (train/val/test)
        max_examples: Maximum number of examples to load
        
    Returns:
        List of QA examples
    """
    logger.info(f"Loading {split} data from {data_path}")
    
    # Load from JSON file
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
    
    # Load from HuggingFace dataset
    else:
        try:
            dataset = load_dataset(data_path, split=split)
            data = dataset.to_dict()
            
            # Convert to list of dictionaries
            keys = list(data.keys())
            data = [
                {key: data[key][i] for key in keys}
                for i in range(len(data[keys[0]]))
            ]
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    # Limit number of examples
    if max_examples and len(data) > max_examples:
        data = data[:max_examples]
    
    logger.info(f"Loaded {len(data)} {split} examples")
    return data


def load_knowledge_base(
    kb_path: str,
    max_docs: Optional[int] = None
) -> List[Dict]:
    """
    Load knowledge base from file
    
    Args:
        kb_path: Path to knowledge base file
        max_docs: Maximum number of documents to load
        
    Returns:
        List of documents
    """
    logger.info(f"Loading knowledge base from {kb_path}")
    
    # Load from JSON file
    if kb_path.endswith('.json'):
        with open(kb_path, 'r') as f:
            documents = json.load(f)
    
    # Load from text file (one document per line)
    elif kb_path.endswith('.txt'):
        documents = []
        with open(kb_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    documents.append({
                        "id": str(i),
                        "text": line
                    })
    
    else:
        raise ValueError(f"Unsupported file format: {kb_path}")
    
    # Limit number of documents
    if max_docs and len(documents) > max_docs:
        documents = documents[:max_docs]
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def create_retrieval_training_data(
    qa_data: List[Dict],
    knowledge_base: List[Dict],
    num_negatives: int = 5
) -> List[Dict]:
    """
    Create retrieval training data with positive and negative examples
    
    Args:
        qa_data: QA pairs with context information
        knowledge_base: Knowledge base documents
        num_negatives: Number of negative examples per question
        
    Returns:
        Enhanced QA data with retrieval training targets
    """
    logger.info("Creating retrieval training data...")
    
    # Create document lookup
    doc_lookup = {doc["id"]: doc for doc in knowledge_base}
    
    enhanced_data = []
    
    for item in qa_data:
        enhanced_item = item.copy()
        
        # Find positive documents
        positive_docs = []
        if "context" in item:
            # Find matching documents
            context_text = item["context"]
            for doc in knowledge_base:
                if context_text in doc.get("text", ""):
                    positive_docs.append(doc["text"])
                    break
        
        # Sample negative documents
        negative_docs = []
        available_docs = [doc for doc in knowledge_base if doc.get("text", "") not in positive_docs]
        
        if len(available_docs) >= num_negatives:
            sampled_docs = np.random.choice(available_docs, num_negatives, replace=False)
            negative_docs = [doc["text"] for doc in sampled_docs]
        else:
            negative_docs = [doc["text"] for doc in available_docs]
        
        enhanced_item["positive_docs"] = positive_docs
        enhanced_item["negative_docs"] = negative_docs
        
        enhanced_data.append(enhanced_item)
    
    logger.info(f"Created retrieval training data for {len(enhanced_data)} examples")
    return enhanced_data


def prepare_data_for_training(
    config: ModelConfig,
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    test_path: Optional[str] = None,
    kb_path: Optional[str] = None,
    train_data: Optional[List[Dict]] = None,
    val_data: Optional[List[Dict]] = None,
    test_data: Optional[List[Dict]] = None,
    knowledge_base: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Prepare all data for training
    
    Args:
        config: Model configuration
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        kb_path: Path to knowledge base
        train_data: Training data (if not loading from file)
        val_data: Validation data (if not loading from file)
        test_data: Test data (if not loading from file)
        knowledge_base: Knowledge base (if not loading from file)
        
    Returns:
        Dictionary with datasets and knowledge base
    """
    logger.info("Preparing data for training...")
    
    # Load datasets
    datasets = {}
    
    if train_data is not None:
        datasets["train"] = train_data
    elif train_path:
        datasets["train"] = load_qa_dataset(train_path, "train")
    
    if val_data is not None:
        datasets["val"] = val_data
    elif val_path:
        datasets["val"] = load_qa_dataset(val_path, "val")
    
    if test_data is not None:
        datasets["test"] = test_data
    elif test_path:
        datasets["test"] = load_qa_dataset(test_path, "test")
    
    # Load knowledge base
    if knowledge_base is not None:
        kb = knowledge_base
    elif kb_path:
        kb = load_knowledge_base(kb_path)
    else:
        kb = []
    
    # Create retrieval training data
    if datasets.get("train") and kb:
        datasets["train"] = create_retrieval_training_data(
            datasets["train"], kb, num_negatives=5
        )
    
    return {
        "datasets": datasets,
        "knowledge_base": kb
    }


def create_sample_data(
    num_examples: int = 100,
    num_docs: int = 1000
) -> Dict[str, Any]:
    """
    Create sample data for testing
    
    Args:
        num_examples: Number of QA examples
        num_docs: Number of knowledge base documents
        
    Returns:
        Sample data dictionary
    """
    logger.info(f"Creating sample data: {num_examples} QA pairs, {num_docs} documents")
    
    # Sample questions and answers
    templates = [
        ("What is the capital of {country}?", "The capital of {country} is {capital}.", "{country} is a country. Its capital is {capital}."),
        ("How does {process} work?", "{process} works by {mechanism}.", "{process} is a {type} process. It involves {mechanism}."),
        ("What is {concept}?", "{concept} is {definition}.", "{concept} refers to {definition}. It is important in {domain}."),
        ("Who invented {invention}?", "{invention} was invented by {inventor}.", "{inventor} was a {profession} who invented {invention}."),
        ("When was {event}?", "{event} happened in {year}.", "{event} was a significant event that occurred in {year}.")
    ]
    
    # Generate data
    qa_data = []
    kb_docs = []
    
    for i in range(num_examples):
        template = templates[i % len(templates)]
        question_template, answer_template, context_template = template
        
        # Fill in placeholders (simplified)
        if "capital" in question_template:
            country = f"Country{i}"
            capital = f"Capital{i}"
            question = question_template.format(country=country)
            answer = answer_template.format(country=country, capital=capital)
            context = context_template.format(country=country, capital=capital)
        else:
            # Generic filling
            question = question_template.format(
                process=f"Process{i}",
                concept=f"Concept{i}",
                invention=f"Invention{i}",
                event=f"Event{i}"
            )
            answer = answer_template.format(
                process=f"Process{i}",
                mechanism=f"Mechanism{i}",
                concept=f"Concept{i}",
                definition=f"Definition{i}",
                invention=f"Invention{i}",
                inventor=f"Inventor{i}",
                event=f"Event{i}",
                year=f"Year{i}"
            )
            context = context_template.format(
                process=f"Process{i}",
                type=f"Type{i}",
                mechanism=f"Mechanism{i}",
                concept=f"Concept{i}",
                definition=f"Definition{i}",
                domain=f"Domain{i}",
                inventor=f"Inventor{i}",
                profession=f"Profession{i}",
                invention=f"Invention{i}",
                event=f"Event{i}",
                year=f"Year{i}"
            )
        
        qa_data.append({
            "id": str(i),
            "question": question,
            "answer": answer,
            "context": context
        })
        
        # Create corresponding knowledge base document
        kb_docs.append({
            "id": str(i),
            "text": context,
            "title": f"Document {i}"
        })
    
    # Add additional random documents
    for i in range(num_examples, num_docs):
        kb_docs.append({
            "id": str(i),
            "text": f"This is document {i} containing random information about topic {i % 10}.",
            "title": f"Document {i}"
        })
    
    # Split QA data
    train_size = int(0.8 * num_examples)
    val_size = int(0.1 * num_examples)
    
    return {
        "datasets": {
            "train": qa_data[:train_size],
            "val": qa_data[train_size:train_size + val_size],
            "test": qa_data[train_size + val_size:]
        },
        "knowledge_base": kb_docs
    }


if __name__ == "__main__":
    # Test data utilities
    sample_data = create_sample_data(50, 100)
    
    print("Sample data created:")
    print(f"- Training examples: {len(sample_data['datasets']['train'])}")
    print(f"- Validation examples: {len(sample_data['datasets']['val'])}")
    print(f"- Test examples: {len(sample_data['datasets']['test'])}")
    print(f"- Knowledge base documents: {len(sample_data['knowledge_base'])}")
    
    # Create dataset
    from .config import DEFAULT_CONFIG
    
    train_dataset = REALMRAGDataset(
        sample_data["datasets"]["train"],
        DEFAULT_CONFIG.model
    )
    
    # Test dataloader
    dataloader = create_dataloader(train_dataset, batch_size=4)
    
    print(f"\nDataset created with {len(train_dataset)} examples")
    print(f"Dataloader created with batch size 4")
    
    # Test batch
    batch = next(iter(dataloader))
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Batch sizes: {[len(v) for v in batch.values()]}") 