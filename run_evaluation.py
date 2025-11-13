"""
Run Complete Evaluation
=======================
This script runs a complete evaluation of the RAG system using the benchmark dataset.
It generates predictions for all test questions and evaluates them against ground truth.
"""

import os
import sys
import json
from typing import List, Dict
import pandas as pd
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


def load_model(model_name: str = "google/flan-t5-base"):
    """Load the LLM."""
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
    return llm


def create_rag_system(pdf_path: str, 
                     chunk_size: int = 1000,
                     chunk_overlap: int = 200,
                     embedding_model: str = "all-MiniLM-L6-v2",
                     k: int = 4):
    """Create the RAG system with specified parameters."""
    
    print(f"📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"✂️  Splitting text (chunk_size={chunk_size}, overlap={chunk_overlap})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    print(f"   Created {len(texts)} chunks")
    
    print(f"🔢 Creating embeddings with {embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    print(f"💾 Building FAISS vector store")
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    return vectorstore, k


def generate_predictions(vectorstore, llm, questions: List[str], k: int = 4) -> List[str]:
    """Generate predictions for all questions."""
    
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
    print(f"\n🤖 Generating predictions for {len(questions)} questions...")
    
    for i, question in enumerate(questions, 1):
        print(f"   Question {i}/{len(questions)}: {question[:60]}...")
        try:
            response = rag_chain.invoke({"input": question})
            answer = response["answer"]
            predictions.append(answer)
            print(f"   ✅ Answer: {answer[:80]}...")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            predictions.append(f"Error: {str(e)}")
    
    return predictions


def save_predictions(questions: List[str], 
                    predictions: List[str], 
                    ground_truths: List[str],
                    output_path: str = "predictions.json"):
    """Save predictions with questions and ground truths."""
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(questions),
        "predictions": [
            {
                "id": i + 1,
                "question": q,
                "prediction": p,
                "ground_truth": g
            }
            for i, (q, p, g) in enumerate(zip(questions, predictions, ground_truths))
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Predictions saved to: {output_path}")


def main():
    """Main evaluation pipeline."""
    
    print("="*60)
    print("🎯 RAG SYSTEM EVALUATION")
    print("="*60)
    
    # Check if PDF path is provided
    if len(sys.argv) < 2:
        print("\n⚠️  Usage: python run_evaluation.py <path_to_pdf>")
        print("Example: python run_evaluation.py documents/sample_document.pdf")
        return
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        return
    
    # Configuration
    config = {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "embedding_model": "all-MiniLM-L6-v2",
        "k": 4,
        "llm_model": "google/flan-t5-base"
    }
    
    print("\n📋 Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # Load benchmark dataset
    print("\n📊 Loading benchmark dataset...")
    questions, ground_truths, test_cases = load_benchmark_dataset("benchmark_dataset.json")
    print(f"   Loaded {len(questions)} test questions")
    
    # Load LLM
    llm = load_model(config["llm_model"])
    
    # Create RAG system
    print("\n🔧 Setting up RAG system...")
    vectorstore, k = create_rag_system(
        pdf_path,
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        embedding_model=config["embedding_model"],
        k=config["k"]
    )
    
    # Generate predictions
    predictions = generate_predictions(vectorstore, llm, questions, k)
    
    # Save predictions
    save_predictions(questions, predictions, ground_truths, "predictions.json")
    
    # Evaluate
    print("\n📊 Evaluating predictions...")
    evaluator = RAGEvaluator()
    results_df = evaluator.evaluate_batch(predictions, ground_truths)
    aggregates = evaluator.calculate_aggregate_metrics(results_df)
    
    # Print summary
    evaluator.print_summary(aggregates)
    
    # Save detailed results
    print("\n💾 Saving evaluation results...")
    results_df.to_csv("evaluation_results.csv", index=False)
    print("   ✅ Saved: evaluation_results.csv")
    
    evaluator.save_results(results_df, aggregates, "evaluation_results.json")
    
    # Print top and bottom performers
    print("\n🏆 Best Performing Questions (by semantic similarity):")
    top_5 = results_df.nlargest(5, 'semantic_similarity')[['qa_id', 'semantic_similarity', 'rouge1_f', 'f1']]
    print(top_5.to_string(index=False))
    
    print("\n⚠️  Worst Performing Questions (by semantic similarity):")
    bottom_5 = results_df.nsmallest(5, 'semantic_similarity')[['qa_id', 'semantic_similarity', 'rouge1_f', 'f1']]
    print(bottom_5.to_string(index=False))
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - predictions.json (all Q&A pairs)")
    print("  - evaluation_results.csv (detailed metrics)")
    print("  - evaluation_results.json (summary)")


if __name__ == "__main__":
    main()
