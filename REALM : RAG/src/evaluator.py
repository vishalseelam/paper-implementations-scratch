"""
Evaluator for REALM/RAG Models
Comprehensive evaluation metrics and inference capabilities
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge_score import rouge_scorer
import pandas as pd

from .realm_rag import REALMRAGModel
from .config import REALMRAGConfig
from .data_utils import REALMRAGDataset, create_dataloader

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class REALMRAGEvaluator:
    """
    Comprehensive evaluator for REALM/RAG models
    
    Features:
    - Multiple evaluation metrics (EM, F1, BLEU, ROUGE)
    - Retrieval evaluation
    - Generation evaluation
    - Error analysis
    - Visualization
    """
    
    def __init__(
        self,
        model: REALMRAGModel,
        config: REALMRAGConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Initialize metrics
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Results storage
        self.results = {}
        self.detailed_results = []
        
        logger.info("Initialized REALM/RAG evaluator")
    
    def evaluate_dataset(
        self,
        dataset: REALMRAGDataset,
        split: str = "test",
        batch_size: int = 4,
        k: int = 5,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset
        
        Args:
            dataset: Dataset to evaluate on
            split: Dataset split name
            batch_size: Batch size for evaluation
            k: Number of documents to retrieve
            save_results: Whether to save detailed results
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating on {split} set with {len(dataset)} examples")
        
        # Create dataloader
        dataloader = create_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Collect all predictions and targets
        all_predictions = []
        all_targets = []
        all_questions = []
        all_retrieved_docs = []
        all_contexts = []
        
        # Evaluation loop
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                questions = batch["question"]
                answers = batch.get("answer", [""] * len(questions))
                
                # Generate predictions
                results = self.model.retrieve_and_generate(
                    questions,
                    k=k,
                    return_retrieved_docs=True
                )
                
                # Collect results
                all_predictions.extend(results["answers"])
                all_targets.extend(answers)
                all_questions.extend(questions)
                all_retrieved_docs.extend(results["retrieved_docs"])
                all_contexts.extend(results["contexts"])
        
        # Calculate metrics
        metrics = self._calculate_all_metrics(
            all_predictions,
            all_targets,
            all_questions,
            all_retrieved_docs,
            all_contexts
        )
        
        # Store results
        self.results[split] = metrics
        
        # Save detailed results
        if save_results:
            self._save_detailed_results(
                split,
                all_questions,
                all_predictions,
                all_targets,
                all_retrieved_docs,
                all_contexts
            )
        
        # Log results
        self._log_results(split, metrics)
        
        return metrics
    
    def _calculate_all_metrics(
        self,
        predictions: List[str],
        targets: List[str],
        questions: List[str],
        retrieved_docs: List[List[Dict]],
        contexts: List[List[str]]
    ) -> Dict[str, Any]:
        """Calculate all evaluation metrics"""
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self._calculate_basic_metrics(predictions, targets))
        
        # BLEU score
        metrics.update(self._calculate_bleu_score(predictions, targets))
        
        # ROUGE scores
        metrics.update(self._calculate_rouge_scores(predictions, targets))
        
        # Retrieval metrics
        metrics.update(self._calculate_retrieval_metrics(
            questions, targets, retrieved_docs
        ))
        
        # Length statistics
        metrics.update(self._calculate_length_statistics(predictions, targets))
        
        return metrics
    
    def _calculate_basic_metrics(
        self,
        predictions: List[str],
        targets: List[str]
    ) -> Dict[str, float]:
        """Calculate basic metrics (EM, F1)"""
        
        if not targets or all(t == "" for t in targets):
            return {
                "exact_match": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }
        
        # Exact Match
        exact_matches = []
        f1_scores = []
        precisions = []
        recalls = []
        
        for pred, target in zip(predictions, targets):
            # Normalize text
            pred_norm = pred.strip().lower()
            target_norm = target.strip().lower()
            
            # Exact match
            exact_matches.append(1 if pred_norm == target_norm else 0)
            
            # Token-level F1
            pred_tokens = set(pred_norm.split())
            target_tokens = set(target_norm.split())
            
            if not pred_tokens and not target_tokens:
                f1_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            elif not pred_tokens or not target_tokens:
                f1_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            else:
                common_tokens = pred_tokens & target_tokens
                precision = len(common_tokens) / len(pred_tokens)
                recall = len(common_tokens) / len(target_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                f1_scores.append(f1)
                precisions.append(precision)
                recalls.append(recall)
        
        return {
            "exact_match": np.mean(exact_matches),
            "f1_score": np.mean(f1_scores),
            "precision": np.mean(precisions),
            "recall": np.mean(recalls)
        }
    
    def _calculate_bleu_score(
        self,
        predictions: List[str],
        targets: List[str]
    ) -> Dict[str, float]:
        """Calculate BLEU score"""
        
        if not targets or all(t == "" for t in targets):
            return {
                "bleu_1": 0.0,
                "bleu_2": 0.0,
                "bleu_3": 0.0,
                "bleu_4": 0.0
            }
        
        # Prepare references and hypotheses
        references = []
        hypotheses = []
        
        for pred, target in zip(predictions, targets):
            # Tokenize
            pred_tokens = pred.strip().lower().split()
            target_tokens = target.strip().lower().split()
            
            references.append([target_tokens])
            hypotheses.append(pred_tokens)
        
        # Calculate BLEU scores
        bleu_scores = {}
        for n in range(1, 5):
            try:
                weights = [1.0/n] * n + [0.0] * (4-n)
                bleu_score = corpus_bleu(references, hypotheses, weights=weights)
                bleu_scores[f"bleu_{n}"] = bleu_score
            except:
                bleu_scores[f"bleu_{n}"] = 0.0
        
        return bleu_scores
    
    def _calculate_rouge_scores(
        self,
        predictions: List[str],
        targets: List[str]
    ) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        
        if not targets or all(t == "" for t in targets):
            return {
                "rouge1_precision": 0.0,
                "rouge1_recall": 0.0,
                "rouge1_f1": 0.0,
                "rouge2_precision": 0.0,
                "rouge2_recall": 0.0,
                "rouge2_f1": 0.0,
                "rougeL_precision": 0.0,
                "rougeL_recall": 0.0,
                "rougeL_f1": 0.0
            }
        
        # Calculate ROUGE scores
        rouge_scores = defaultdict(list)
        
        for pred, target in zip(predictions, targets):
            scores = self.rouge_scorer.score(target, pred)
            
            for rouge_type, score in scores.items():
                rouge_scores[f"{rouge_type}_precision"].append(score.precision)
                rouge_scores[f"{rouge_type}_recall"].append(score.recall)
                rouge_scores[f"{rouge_type}_f1"].append(score.fmeasure)
        
        # Average scores
        avg_rouge_scores = {
            key: np.mean(values) for key, values in rouge_scores.items()
        }
        
        return avg_rouge_scores
    
    def _calculate_retrieval_metrics(
        self,
        questions: List[str],
        targets: List[str],
        retrieved_docs: List[List[Dict]]
    ) -> Dict[str, float]:
        """Calculate retrieval metrics"""
        
        metrics = {
            "retrieval_precision": 0.0,
            "retrieval_recall": 0.0,
            "retrieval_f1": 0.0,
            "mean_reciprocal_rank": 0.0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0
        }
        
        if not targets or all(t == "" for t in targets):
            return metrics
        
        # Calculate retrieval metrics
        precisions = []
        recalls = []
        mrr_scores = []
        hit_1_scores = []
        hit_3_scores = []
        hit_5_scores = []
        
        for question, target, docs in zip(questions, targets, retrieved_docs):
            if not docs:
                precisions.append(0.0)
                recalls.append(0.0)
                mrr_scores.append(0.0)
                hit_1_scores.append(0.0)
                hit_3_scores.append(0.0)
                hit_5_scores.append(0.0)
                continue
            
            # Check if target answer appears in retrieved documents
            target_tokens = set(target.strip().lower().split())
            
            relevant_docs = []
            for i, doc in enumerate(docs):
                doc_text = doc.get("text", "").lower()
                doc_tokens = set(doc_text.split())
                
                # Check overlap with target
                overlap = len(target_tokens & doc_tokens)
                if overlap > 0:
                    relevant_docs.append(i)
            
            # Calculate metrics
            num_relevant = len(relevant_docs)
            num_retrieved = len(docs)
            
            if num_relevant > 0:
                precision = num_relevant / num_retrieved
                recall = 1.0  # Assume all relevant docs are in knowledge base
                
                # MRR
                first_relevant_rank = min(relevant_docs) + 1 if relevant_docs else float('inf')
                mrr = 1.0 / first_relevant_rank if first_relevant_rank != float('inf') else 0.0
                
                # Hit@K
                hit_1 = 1.0 if 0 in relevant_docs else 0.0
                hit_3 = 1.0 if any(i < 3 for i in relevant_docs) else 0.0
                hit_5 = 1.0 if any(i < 5 for i in relevant_docs) else 0.0
            else:
                precision = 0.0
                recall = 0.0
                mrr = 0.0
                hit_1 = 0.0
                hit_3 = 0.0
                hit_5 = 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            mrr_scores.append(mrr)
            hit_1_scores.append(hit_1)
            hit_3_scores.append(hit_3)
            hit_5_scores.append(hit_5)
        
        # Calculate averages
        metrics["retrieval_precision"] = np.mean(precisions)
        metrics["retrieval_recall"] = np.mean(recalls)
        
        # F1 score
        avg_precision = metrics["retrieval_precision"]
        avg_recall = metrics["retrieval_recall"]
        if avg_precision + avg_recall > 0:
            metrics["retrieval_f1"] = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        
        metrics["mean_reciprocal_rank"] = np.mean(mrr_scores)
        metrics["hit_at_1"] = np.mean(hit_1_scores)
        metrics["hit_at_3"] = np.mean(hit_3_scores)
        metrics["hit_at_5"] = np.mean(hit_5_scores)
        
        return metrics
    
    def _calculate_length_statistics(
        self,
        predictions: List[str],
        targets: List[str]
    ) -> Dict[str, float]:
        """Calculate length statistics"""
        
        pred_lengths = [len(pred.split()) for pred in predictions]
        target_lengths = [len(target.split()) for target in targets if target]
        
        return {
            "avg_prediction_length": np.mean(pred_lengths),
            "avg_target_length": np.mean(target_lengths) if target_lengths else 0.0,
            "length_ratio": np.mean(pred_lengths) / np.mean(target_lengths) if target_lengths else 0.0
        }
    
    def _save_detailed_results(
        self,
        split: str,
        questions: List[str],
        predictions: List[str],
        targets: List[str],
        retrieved_docs: List[List[Dict]],
        contexts: List[List[str]]
    ):
        """Save detailed results to file"""
        
        detailed_results = []
        
        for i, (question, pred, target, docs, ctx) in enumerate(zip(
            questions, predictions, targets, retrieved_docs, contexts
        )):
            result = {
                "id": i,
                "question": question,
                "prediction": pred,
                "target": target,
                "retrieved_docs": [
                    {
                        "text": doc.get("text", ""),
                        "score": doc.get("score", 0.0),
                        "rank": doc.get("rank", 0)
                    }
                    for doc in docs
                ],
                "contexts": ctx
            }
            
            detailed_results.append(result)
        
        # Save to file
        output_path = os.path.join(
            self.config.experiment.output_dir,
            f"{split}_detailed_results.json"
        )
        
        with open(output_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        logger.info(f"Saved detailed results to {output_path}")
    
    def _log_results(self, split: str, metrics: Dict[str, Any]):
        """Log evaluation results"""
        
        logger.info(f"=== {split.upper()} EVALUATION RESULTS ===")
        logger.info(f"Exact Match: {metrics['exact_match']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"BLEU-4: {metrics['bleu_4']:.4f}")
        logger.info(f"ROUGE-L F1: {metrics['rougeL_f1']:.4f}")
        logger.info(f"Retrieval F1: {metrics['retrieval_f1']:.4f}")
        logger.info(f"MRR: {metrics['mean_reciprocal_rank']:.4f}")
        logger.info("=" * 40)
    
    def error_analysis(
        self,
        split: str = "test",
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Perform error analysis
        
        Args:
            split: Dataset split to analyze
            top_k: Number of top errors to analyze
            
        Returns:
            Error analysis results
        """
        
        if split not in self.results:
            raise ValueError(f"No results found for split: {split}")
        
        # Load detailed results
        results_path = os.path.join(
            self.config.experiment.output_dir,
            f"{split}_detailed_results.json"
        )
        
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Detailed results not found: {results_path}")
        
        with open(results_path, 'r') as f:
            detailed_results = json.load(f)
        
        # Calculate individual errors
        errors = []
        
        for result in detailed_results:
            pred = result["prediction"].strip().lower()
            target = result["target"].strip().lower()
            
            # Calculate error score (1 - F1)
            pred_tokens = set(pred.split())
            target_tokens = set(target.split())
            
            if not pred_tokens and not target_tokens:
                f1 = 1.0
            elif not pred_tokens or not target_tokens:
                f1 = 0.0
            else:
                common_tokens = pred_tokens & target_tokens
                precision = len(common_tokens) / len(pred_tokens)
                recall = len(common_tokens) / len(target_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            error_score = 1 - f1
            
            errors.append({
                "id": result["id"],
                "question": result["question"],
                "prediction": result["prediction"],
                "target": result["target"],
                "error_score": error_score,
                "retrieved_docs": result["retrieved_docs"]
            })
        
        # Sort by error score
        errors.sort(key=lambda x: x["error_score"], reverse=True)
        
        # Analyze error patterns
        error_patterns = {
            "no_answer": 0,
            "wrong_answer": 0,
            "partial_answer": 0,
            "too_long": 0,
            "too_short": 0
        }
        
        for error in errors:
            pred = error["prediction"].strip()
            target = error["target"].strip()
            
            if not pred:
                error_patterns["no_answer"] += 1
            elif len(pred.split()) > 2 * len(target.split()):
                error_patterns["too_long"] += 1
            elif len(pred.split()) < 0.5 * len(target.split()):
                error_patterns["too_short"] += 1
            elif error["error_score"] > 0.5:
                error_patterns["wrong_answer"] += 1
            else:
                error_patterns["partial_answer"] += 1
        
        return {
            "top_errors": errors[:top_k],
            "error_patterns": error_patterns,
            "total_errors": len(errors)
        }
    
    def create_evaluation_report(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """
        Create comprehensive evaluation report
        
        Args:
            output_path: Path to save report
            
        Returns:
            Path to saved report
        """
        
        if output_path is None:
            output_path = os.path.join(
                self.config.experiment.output_dir,
                "evaluation_report.html"
            )
        
        # Create HTML report
        html_content = self._generate_html_report()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Evaluation report saved to {output_path}")
        return output_path
    
    def _generate_html_report(self) -> str:
        """Generate HTML evaluation report"""
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>REALM/RAG Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-table { border-collapse: collapse; width: 100%; }
                .metric-table th, .metric-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .metric-table th { background-color: #f2f2f2; }
                .section { margin-bottom: 30px; }
                .error { color: red; }
                .success { color: green; }
            </style>
        </head>
        <body>
            <h1>REALM/RAG Evaluation Report</h1>
        """
        
        # Add results for each split
        for split, metrics in self.results.items():
            html += f"""
            <div class="section">
                <h2>{split.upper()} Results</h2>
                <table class="metric-table">
                    <tr><th>Metric</th><th>Value</th></tr>
            """
            
            for metric, value in metrics.items():
                if isinstance(value, float):
                    html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
                else:
                    html += f"<tr><td>{metric}</td><td>{value}</td></tr>"
            
            html += """
                </table>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def compare_models(
        self,
        other_results: Dict[str, Any],
        split: str = "test"
    ) -> Dict[str, Any]:
        """
        Compare with other model results
        
        Args:
            other_results: Results from other model
            split: Dataset split to compare
            
        Returns:
            Comparison results
        """
        
        if split not in self.results:
            raise ValueError(f"No results found for split: {split}")
        
        current_results = self.results[split]
        
        comparison = {}
        
        for metric in current_results:
            if metric in other_results:
                current_val = current_results[metric]
                other_val = other_results[metric]
                
                if isinstance(current_val, (int, float)) and isinstance(other_val, (int, float)):
                    diff = current_val - other_val
                    percent_diff = (diff / other_val) * 100 if other_val != 0 else 0
                    
                    comparison[metric] = {
                        "current": current_val,
                        "other": other_val,
                        "difference": diff,
                        "percent_difference": percent_diff,
                        "better": diff > 0
                    }
        
        return comparison
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluation results"""
        
        summary = {
            "splits_evaluated": list(self.results.keys()),
            "best_metrics": {},
            "model_info": self.model.get_model_info()
        }
        
        # Find best metrics across splits
        all_metrics = set()
        for split_results in self.results.values():
            all_metrics.update(split_results.keys())
        
        for metric in all_metrics:
            best_value = None
            best_split = None
            
            for split, results in self.results.items():
                if metric in results:
                    value = results[metric]
                    if isinstance(value, (int, float)):
                        if best_value is None or value > best_value:
                            best_value = value
                            best_split = split
            
            if best_value is not None:
                summary["best_metrics"][metric] = {
                    "value": best_value,
                    "split": best_split
                }
        
        return summary


def evaluate_model(
    model: REALMRAGModel,
    config: REALMRAGConfig,
    test_data: List[Dict],
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model
    
    Args:
        model: REALM/RAG model to evaluate
        config: Configuration
        test_data: Test data
        save_results: Whether to save results
        
    Returns:
        Evaluation results
    """
    
    # Create evaluator
    evaluator = REALMRAGEvaluator(model, config)
    
    # Create test dataset
    test_dataset = REALMRAGDataset(test_data, config.model)
    
    # Evaluate
    results = evaluator.evaluate_dataset(
        test_dataset,
        split="test",
        save_results=save_results
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    from .config import DEFAULT_CONFIG
    from .realm_rag import REALMRAGModel, create_sample_qa_data
    
    # Create sample data
    qa_data = create_sample_qa_data(20)
    documents = [{"text": item["context"], "id": item["id"]} for item in qa_data]
    
    # Create model
    model = REALMRAGModel(DEFAULT_CONFIG)
    
    # Prepare knowledge base
    model.prepare_knowledge_base(documents)
    
    # Create evaluator
    evaluator = REALMRAGEvaluator(model, DEFAULT_CONFIG)
    
    # Create test dataset
    test_dataset = REALMRAGDataset(qa_data, DEFAULT_CONFIG.model)
    
    # Evaluate
    results = evaluator.evaluate_dataset(test_dataset, split="test")
    
    # Print results
    print("\n=== EVALUATION RESULTS ===")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
    
    # Create report
    report_path = evaluator.create_evaluation_report()
    print(f"\nDetailed report saved to: {report_path}")
    
    # Get summary
    summary = evaluator.get_summary()
    print(f"\nEvaluated {len(summary['splits_evaluated'])} splits")
    print(f"Best F1 score: {summary['best_metrics'].get('f1_score', {}).get('value', 0):.4f}") 