"""
Evaluation Module for RAG System
==================================
This module provides comprehensive evaluation metrics for assessing the 
performance of the RAG (Retrieval-Augmented Generation) chatbot.

Metrics implemented:
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- BLEU (Bilingual Evaluation Understudy)
- Semantic Similarity (using embeddings)
- Exact Match and F1 scores
- Retrieval metrics (Precision, Recall)
"""

import json
import os
from typing import List, Dict, Tuple
import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime


class RAGEvaluator:
    """
    Comprehensive evaluator for RAG systems.
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the evaluator with necessary models.
        
        Args:
            embedding_model_name: Name of the sentence transformer model for semantic similarity
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU()
        self.semantic_model = SentenceTransformer(embedding_model_name)
        
    def calculate_rouge(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
        
        Args:
            prediction: Generated answer
            reference: Ground truth answer
            
        Returns:
            Dictionary with ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rouge2_p': scores['rouge2'].precision,
            'rouge2_r': scores['rouge2'].recall,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeL_p': scores['rougeL'].precision,
            'rougeL_r': scores['rougeL'].recall,
        }
    
    def calculate_bleu(self, prediction: str, reference: str) -> float:
        """
        Calculate BLEU score.
        
        Args:
            prediction: Generated answer
            reference: Ground truth answer
            
        Returns:
            BLEU score
        """
        # BLEU expects list of references
        bleu_score = self.bleu.sentence_score(prediction, [reference])
        return bleu_score.score / 100.0  # Normalize to 0-1
    
    def calculate_semantic_similarity(self, prediction: str, reference: str) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        
        Args:
            prediction: Generated answer
            reference: Ground truth answer
            
        Returns:
            Cosine similarity score (0-1)
        """
        pred_embedding = self.semantic_model.encode([prediction])
        ref_embedding = self.semantic_model.encode([reference])
        similarity = cosine_similarity(pred_embedding, ref_embedding)[0][0]
        return float(similarity)
    
    def calculate_f1_score(self, prediction: str, reference: str) -> Tuple[float, float, float]:
        """
        Calculate token-level F1 score, precision, and recall.
        
        Args:
            prediction: Generated answer
            reference: Ground truth answer
            
        Returns:
            Tuple of (f1, precision, recall)
        """
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0, 0.0, 0.0
        
        common = pred_tokens.intersection(ref_tokens)
        
        precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = len(common) / len(ref_tokens) if len(ref_tokens) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1, precision, recall
    
    def exact_match(self, prediction: str, reference: str) -> bool:
        """
        Check if prediction exactly matches reference (case-insensitive, whitespace normalized).
        
        Args:
            prediction: Generated answer
            reference: Ground truth answer
            
        Returns:
            True if exact match, False otherwise
        """
        pred_normalized = ' '.join(prediction.lower().split())
        ref_normalized = ' '.join(reference.lower().split())
        return pred_normalized == ref_normalized
    
    def evaluate_single_qa(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Evaluate a single question-answer pair with all metrics.
        
        Args:
            prediction: Generated answer
            reference: Ground truth answer
            
        Returns:
            Dictionary with all evaluation metrics
        """
        rouge_scores = self.calculate_rouge(prediction, reference)
        bleu_score = self.calculate_bleu(prediction, reference)
        semantic_sim = self.calculate_semantic_similarity(prediction, reference)
        f1, precision, recall = self.calculate_f1_score(prediction, reference)
        exact = self.exact_match(prediction, reference)
        
        return {
            **rouge_scores,
            'bleu': bleu_score,
            'semantic_similarity': semantic_sim,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'exact_match': 1.0 if exact else 0.0
        }
    
    def evaluate_batch(self, predictions: List[str], references: List[str]) -> pd.DataFrame:
        """
        Evaluate multiple question-answer pairs.
        
        Args:
            predictions: List of generated answers
            references: List of ground truth answers
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            scores = self.evaluate_single_qa(pred, ref)
            scores['qa_id'] = i + 1
            results.append(scores)
        
        df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        cols = ['qa_id'] + [col for col in df.columns if col != 'qa_id']
        df = df[cols]
        
        return df
    
    def calculate_aggregate_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate aggregate statistics from evaluation results.
        
        Args:
            results_df: DataFrame from evaluate_batch
            
        Returns:
            Dictionary with mean and std of each metric
        """
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'qa_id']
        
        aggregates = {}
        for col in numeric_cols:
            aggregates[f'{col}_mean'] = results_df[col].mean()
            aggregates[f'{col}_std'] = results_df[col].std()
        
        return aggregates
    
    def evaluate_retrieval(self, 
                          retrieved_docs: List[str], 
                          relevant_docs: List[str]) -> Dict[str, float]:
        """
        Evaluate retrieval quality.
        
        Args:
            retrieved_docs: List of retrieved document IDs/contents
            relevant_docs: List of truly relevant document IDs/contents
            
        Returns:
            Dictionary with precision, recall, and F1
        """
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        if len(retrieved_set) == 0 or len(relevant_set) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        true_positives = len(retrieved_set.intersection(relevant_set))
        
        precision = true_positives / len(retrieved_set) if len(retrieved_set) > 0 else 0
        recall = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'retrieved_count': len(retrieved_set),
            'relevant_count': len(relevant_set),
            'true_positives': true_positives
        }
    
    def save_results(self, results_df: pd.DataFrame, aggregates: Dict, 
                    output_path: str = "evaluation_results.json"):
        """
        Save evaluation results to JSON file.
        
        Args:
            results_df: DataFrame with individual results
            aggregates: Dictionary with aggregate statistics
            output_path: Path to save results
        """
        output = {
            'timestamp': datetime.now().isoformat(),
            'aggregate_metrics': aggregates,
            'individual_results': results_df.to_dict('records')
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✅ Results saved to {output_path}")
    
    def print_summary(self, aggregates: Dict):
        """
        Print a formatted summary of evaluation results.
        
        Args:
            aggregates: Dictionary with aggregate statistics
        """
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        
        print("\n🎯 Key Metrics (Mean ± Std):")
        print("-" * 60)
        
        key_metrics = ['semantic_similarity', 'rouge1_f', 'rouge2_f', 'rougeL_f', 
                      'bleu', 'f1', 'exact_match']
        
        for metric in key_metrics:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in aggregates:
                mean = aggregates[mean_key]
                std = aggregates.get(std_key, 0)
                print(f"{metric:25s}: {mean:.4f} ± {std:.4f}")
        
        print("="*60 + "\n")


def load_benchmark_dataset(dataset_path: str = "benchmark_dataset.json") -> Tuple[List[str], List[str], List[Dict]]:
    """
    Load benchmark dataset from JSON file.
    
    Args:
        dataset_path: Path to the benchmark dataset JSON file
        
    Returns:
        Tuple of (questions, ground_truths, full_test_cases)
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    test_cases = data['test_cases']
    questions = [tc['question'] for tc in test_cases]
    ground_truths = [tc['ground_truth'] for tc in test_cases]
    
    return questions, ground_truths, test_cases


def run_evaluation_example():
    """
    Example usage of the evaluation module.
    """
    print("🚀 Running Evaluation Example...")
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Example predictions and references
    predictions = [
        "RAG combines retrieval and generation for better answers.",
        "The main components are loader, splitter, embeddings, vector store, retriever, and LLM.",
    ]
    
    references = [
        "A RAG system combines information retrieval with language generation to answer questions.",
        "The main components are document loader, text splitter, embedding model, vector store, retriever, and language model.",
    ]
    
    # Evaluate
    results_df = evaluator.evaluate_batch(predictions, references)
    aggregates = evaluator.calculate_aggregate_metrics(results_df)
    
    # Print results
    print("\n📋 Individual Results:")
    print(results_df.to_string())
    
    evaluator.print_summary(aggregates)
    
    # Save results
    evaluator.save_results(results_df, aggregates, "example_evaluation_results.json")


if __name__ == "__main__":
    run_evaluation_example()
