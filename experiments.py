"""
Experiments Module for RAG System
==================================
This module runs systematic experiments to compare different configurations
of the RAG system and identify optimal parameters.

Experiments include:
1. Chunk size comparison (500, 1000, 2000 tokens)
2. Chunk overlap comparison (0, 100, 200 tokens)
3. Embedding model comparison (different sentence transformers)
4. Retrieval parameter comparison (k=2, k=4, k=8)
5. Different LLM comparison (if multiple models available)
"""

import os
import json
import tempfile
import time
from typing import List, Dict, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from evaluation import RAGEvaluator, load_benchmark_dataset


class RAGExperimentRunner:
    """
    Run systematic experiments on RAG system configurations.
    """
    
    def __init__(self, pdf_path: str, benchmark_path: str = "benchmark_dataset.json"):
        """
        Initialize the experiment runner.
        
        Args:
            pdf_path: Path to the PDF document to use for experiments
            benchmark_path: Path to benchmark dataset JSON
        """
        self.pdf_path = pdf_path
        self.benchmark_path = benchmark_path
        self.evaluator = RAGEvaluator()
        
        # Load benchmark dataset
        self.questions, self.ground_truths, self.test_cases = load_benchmark_dataset(benchmark_path)
        
        print(f"✅ Loaded {len(self.questions)} test questions from benchmark dataset")
        
        # Cache for loaded models
        self.llm_cache = {}
    
    def load_llm(self, model_name: str = "google/flan-t5-base"):
        """
        Load LLM with caching to avoid reloading.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            LangChain-compatible LLM
        """
        if model_name in self.llm_cache:
            return self.llm_cache[model_name]
        
        print(f"📥 Loading model: {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        self.llm_cache[model_name] = llm
        
        return llm
    
    def create_vectorstore(self, 
                          chunk_size: int = 1000, 
                          chunk_overlap: int = 200,
                          embedding_model: str = "all-MiniLM-L6-v2") -> FAISS:
        """
        Create vector store with specified parameters.
        
        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_model: Name of embedding model
            
        Returns:
            FAISS vector store
        """
        # Load PDF
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Create vector store
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        return vectorstore
    
    def run_rag_inference(self, 
                         vectorstore: FAISS,
                         questions: List[str],
                         llm,
                         k: int = 4) -> Tuple[List[str], float]:
        """
        Run RAG inference on a list of questions.
        
        Args:
            vectorstore: Vector store for retrieval
            questions: List of questions
            llm: Language model
            k: Number of chunks to retrieve
            
        Returns:
            Tuple of (predictions, inference_time)
        """
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Use the given context to answer the question. "
            "If you don't know the answer based on the context, say you don't know. "
            "Keep the answer concise and relevant.\n\n"
            "Context: {context}"
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        predictions = []
        start_time = time.time()
        
        for question in questions:
            try:
                response = rag_chain.invoke({"input": question})
                answer = response["answer"]
                predictions.append(answer)
            except Exception as e:
                print(f"⚠️ Error on question: {question[:50]}... - {str(e)}")
                predictions.append(f"Error: {str(e)}")
        
        inference_time = time.time() - start_time
        
        return predictions, inference_time
    
    def experiment_chunk_size(self, sizes: List[int] = [500, 1000, 2000]) -> pd.DataFrame:
        """
        Experiment with different chunk sizes.
        
        Args:
            sizes: List of chunk sizes to test
            
        Returns:
            DataFrame with results
        """
        print("\n" + "="*60)
        print("🧪 EXPERIMENT 1: Chunk Size Comparison")
        print("="*60)
        
        results = []
        llm = self.load_llm("google/flan-t5-base")
        
        for size in sizes:
            print(f"\n📊 Testing chunk size: {size}")
            
            vectorstore = self.create_vectorstore(
                chunk_size=size,
                chunk_overlap=200,
                embedding_model="all-MiniLM-L6-v2"
            )
            
            predictions, inf_time = self.run_rag_inference(
                vectorstore, self.questions, llm, k=4
            )
            
            # Evaluate
            eval_df = self.evaluator.evaluate_batch(predictions, self.ground_truths)
            aggregates = self.evaluator.calculate_aggregate_metrics(eval_df)
            
            results.append({
                'chunk_size': size,
                'chunk_overlap': 200,
                'rouge1_f': aggregates['rouge1_f_mean'],
                'rouge2_f': aggregates['rouge2_f_mean'],
                'rougeL_f': aggregates['rougeL_f_mean'],
                'bleu': aggregates['bleu_mean'],
                'semantic_similarity': aggregates['semantic_similarity_mean'],
                'f1': aggregates['f1_mean'],
                'inference_time': inf_time
            })
        
        return pd.DataFrame(results)
    
    def experiment_chunk_overlap(self, overlaps: List[int] = [0, 100, 200]) -> pd.DataFrame:
        """
        Experiment with different chunk overlaps.
        
        Args:
            overlaps: List of overlap sizes to test
            
        Returns:
            DataFrame with results
        """
        print("\n" + "="*60)
        print("🧪 EXPERIMENT 2: Chunk Overlap Comparison")
        print("="*60)
        
        results = []
        llm = self.load_llm("google/flan-t5-base")
        
        for overlap in overlaps:
            print(f"\n📊 Testing chunk overlap: {overlap}")
            
            vectorstore = self.create_vectorstore(
                chunk_size=1000,
                chunk_overlap=overlap,
                embedding_model="all-MiniLM-L6-v2"
            )
            
            predictions, inf_time = self.run_rag_inference(
                vectorstore, self.questions, llm, k=4
            )
            
            # Evaluate
            eval_df = self.evaluator.evaluate_batch(predictions, self.ground_truths)
            aggregates = self.evaluator.calculate_aggregate_metrics(eval_df)
            
            results.append({
                'chunk_size': 1000,
                'chunk_overlap': overlap,
                'rouge1_f': aggregates['rouge1_f_mean'],
                'rouge2_f': aggregates['rouge2_f_mean'],
                'rougeL_f': aggregates['rougeL_f_mean'],
                'bleu': aggregates['bleu_mean'],
                'semantic_similarity': aggregates['semantic_similarity_mean'],
                'f1': aggregates['f1_mean'],
                'inference_time': inf_time
            })
        
        return pd.DataFrame(results)
    
    def experiment_retrieval_k(self, k_values: List[int] = [2, 4, 8]) -> pd.DataFrame:
        """
        Experiment with different k values for retrieval.
        
        Args:
            k_values: List of k values to test
            
        Returns:
            DataFrame with results
        """
        print("\n" + "="*60)
        print("🧪 EXPERIMENT 3: Retrieval K Comparison")
        print("="*60)
        
        results = []
        llm = self.load_llm("google/flan-t5-base")
        
        # Create vectorstore once
        vectorstore = self.create_vectorstore(
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="all-MiniLM-L6-v2"
        )
        
        for k in k_values:
            print(f"\n📊 Testing k={k}")
            
            predictions, inf_time = self.run_rag_inference(
                vectorstore, self.questions, llm, k=k
            )
            
            # Evaluate
            eval_df = self.evaluator.evaluate_batch(predictions, self.ground_truths)
            aggregates = self.evaluator.calculate_aggregate_metrics(eval_df)
            
            results.append({
                'k': k,
                'rouge1_f': aggregates['rouge1_f_mean'],
                'rouge2_f': aggregates['rouge2_f_mean'],
                'rougeL_f': aggregates['rougeL_f_mean'],
                'bleu': aggregates['bleu_mean'],
                'semantic_similarity': aggregates['semantic_similarity_mean'],
                'f1': aggregates['f1_mean'],
                'inference_time': inf_time
            })
        
        return pd.DataFrame(results)
    
    def experiment_embedding_models(self, 
                                   models: List[str] = [
                                       "all-MiniLM-L6-v2",
                                       "all-mpnet-base-v2"
                                   ]) -> pd.DataFrame:
        """
        Experiment with different embedding models.
        
        Args:
            models: List of embedding model names
            
        Returns:
            DataFrame with results
        """
        print("\n" + "="*60)
        print("🧪 EXPERIMENT 4: Embedding Model Comparison")
        print("="*60)
        
        results = []
        llm = self.load_llm("google/flan-t5-base")
        
        for model_name in models:
            print(f"\n📊 Testing embedding model: {model_name}")
            
            vectorstore = self.create_vectorstore(
                chunk_size=1000,
                chunk_overlap=200,
                embedding_model=model_name
            )
            
            predictions, inf_time = self.run_rag_inference(
                vectorstore, self.questions, llm, k=4
            )
            
            # Evaluate
            eval_df = self.evaluator.evaluate_batch(predictions, self.ground_truths)
            aggregates = self.evaluator.calculate_aggregate_metrics(eval_df)
            
            results.append({
                'embedding_model': model_name,
                'rouge1_f': aggregates['rouge1_f_mean'],
                'rouge2_f': aggregates['rouge2_f_mean'],
                'rougeL_f': aggregates['rougeL_f_mean'],
                'bleu': aggregates['bleu_mean'],
                'semantic_similarity': aggregates['semantic_similarity_mean'],
                'f1': aggregates['f1_mean'],
                'inference_time': inf_time
            })
        
        return pd.DataFrame(results)
    
    def visualize_results(self, results_dict: Dict[str, pd.DataFrame], output_dir: str = "experiment_results"):
        """
        Create visualizations of experiment results.
        
        Args:
            results_dict: Dictionary mapping experiment names to result DataFrames
            output_dir: Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        sns.set_style("whitegrid")
        metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'bleu', 'semantic_similarity', 'f1']
        
        for exp_name, df in results_dict.items():
            print(f"\n📊 Creating visualizations for: {exp_name}")
            
            # Determine x-axis column
            x_col = None
            for col in df.columns:
                if col not in metrics + ['inference_time']:
                    x_col = col
                    break
            
            if x_col is None:
                continue
            
            # Create subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'{exp_name} - Metric Comparison', fontsize=16, fontweight='bold')
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx // 3, idx % 3]
                
                if metric in df.columns:
                    ax.plot(df[x_col], df[metric], marker='o', linewidth=2, markersize=8)
                    ax.set_xlabel(x_col, fontsize=10)
                    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
                    ax.set_title(metric.upper(), fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # Rotate x-labels if needed
                    if df[x_col].dtype == 'object':
                        ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(output_dir, f"{exp_name.replace(' ', '_').lower()}_metrics.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"  ✅ Saved plot: {plot_path}")
            plt.close()
    
    def run_all_experiments(self) -> Dict[str, pd.DataFrame]:
        """
        Run all experiments and return results.
        
        Returns:
            Dictionary mapping experiment names to result DataFrames
        """
        print("\n" + "="*60)
        print("🚀 RUNNING ALL EXPERIMENTS")
        print("="*60)
        
        results = {}
        
        # Experiment 1: Chunk size
        results['Chunk Size Comparison'] = self.experiment_chunk_size([500, 1000, 2000])
        
        # Experiment 2: Chunk overlap
        results['Chunk Overlap Comparison'] = self.experiment_chunk_overlap([0, 100, 200])
        
        # Experiment 3: Retrieval k
        results['Retrieval K Comparison'] = self.experiment_retrieval_k([2, 4, 8])
        
        # Experiment 4: Embedding models
        results['Embedding Model Comparison'] = self.experiment_embedding_models([
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2"
        ])
        
        return results
    
    def save_all_results(self, results_dict: Dict[str, pd.DataFrame], output_dir: str = "experiment_results"):
        """
        Save all experiment results to CSV and JSON files.
        
        Args:
            results_dict: Dictionary mapping experiment names to result DataFrames
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for exp_name, df in results_dict.items():
            # Save CSV
            csv_path = os.path.join(output_dir, f"{exp_name.replace(' ', '_').lower()}_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            print(f"✅ Saved CSV: {csv_path}")
            
            # Save JSON
            json_path = os.path.join(output_dir, f"{exp_name.replace(' ', '_').lower()}_{timestamp}.json")
            df.to_json(json_path, orient='records', indent=2)
            print(f"✅ Saved JSON: {json_path}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'experiments': list(results_dict.keys()),
            'total_questions': len(self.questions),
            'results': {name: df.to_dict('records') for name, df in results_dict.items()}
        }
        
        summary_path = os.path.join(output_dir, f"experiment_summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Saved summary: {summary_path}")


def main():
    """
    Main function to run experiments.
    """
    import sys
    
    # Check if PDF path is provided
    if len(sys.argv) < 2:
        print("⚠️ Usage: python experiments.py <path_to_pdf>")
        print("Example: python experiments.py documents/sample_document.pdf")
        return
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        return
    
    # Initialize experiment runner
    runner = RAGExperimentRunner(pdf_path)
    
    # Run all experiments
    results = runner.run_all_experiments()
    
    # Save results
    runner.save_all_results(results)
    
    # Create visualizations
    runner.visualize_results(results)
    
    print("\n" + "="*60)
    print("✅ ALL EXPERIMENTS COMPLETED!")
    print("="*60)
    print("\nResults saved in 'experiment_results/' directory")
    print("Check CSV files for detailed metrics")
    print("Check PNG files for visualizations")


if __name__ == "__main__":
    main()
