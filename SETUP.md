# Setup & Usage Guide

Complete guide for setting up and running the RAG chatbot project.

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- 8GB+ RAM (for running local LLMs)
- GPU recommended but not required

### Setup Steps

```bash
# 1. Navigate to project directory
cd rag

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### 1. Interactive Chatbot

```bash
streamlit run app.py
```

- Opens in browser at `http://localhost:8501`
- Upload a PDF using the sidebar
- Ask questions in the chat interface
- View source chunks used for answers

**Expected startup time**: 1-2 minutes (loading LLM)

### 2. Run Evaluation on Benchmark Dataset

```bash
python run_evaluation.py documents/your_document.pdf
```

**What it does**:

- Tests all 15 benchmark questions
- Generates predictions
- Calculates ROUGE, BLEU, F1, semantic similarity
- Saves results to CSV and JSON

**Outputs**:

- `predictions.json` - All questions with predictions and ground truth
- `evaluation_results.csv` - Detailed metrics per question
- `evaluation_results.json` - Aggregate statistics

**Expected runtime**: 5-10 minutes

### 3. Run Full Experiments

```bash
python experiments.py documents/your_document.pdf
```

**What it does**:

- **Experiment 1**: Chunk size comparison (500, 1000, 2000)
- **Experiment 2**: Chunk overlap comparison (0, 100, 200)
- **Experiment 3**: Retrieval k comparison (2, 4, 8)
- **Experiment 4**: Embedding model comparison (MiniLM vs MPNet)

**Outputs** (saved in `experiment_results/`):

- CSV files with detailed metrics
- JSON files with structured results
- PNG plots showing metric comparisons

**Expected runtime**: 30-60 minutes

### 4. Test Evaluation Module

```bash
python evaluation.py
```

Runs a simple example evaluation to verify the evaluation framework works.

---

## 📁 Project Structure

```
rag/
├── app.py                      # Streamlit chatbot application
├── evaluation.py               # Evaluation metrics (ROUGE, BLEU, F1, semantic)
├── experiments.py              # Systematic experiments runner
├── run_evaluation.py           # End-to-end evaluation script
├── benchmark_dataset.json      # 15 test Q&A pairs
├── requirements.txt            # Python dependencies
├── README.md                   # Full project documentation
├── SETUP.md                    # This file
├── .gitignore                  # Git ignore rules
├── documents/                  # PDF documents for testing
└── experiment_results/         # Generated experiment outputs (CSV, JSON, PNG)
```

---

## 🔧 Configuration Options

### Change LLM Model

Edit `app.py` line 24:

```python
model_name = "google/flan-t5-large"  # Change this
```

**Options**:

- `google/flan-t5-small` - Fastest, least accurate
- `google/flan-t5-base` - Balanced (recommended for experiments)
- `google/flan-t5-large` - Slower, more accurate (recommended for app)
- `google/flan-t5-xl` - Best quality, requires GPU

### Modify RAG Parameters

Edit configuration in `run_evaluation.py` or `experiments.py`:

```python
config = {
    "chunk_size": 1000,           # Size of text chunks
    "chunk_overlap": 200,         # Overlap between chunks
    "embedding_model": "all-MiniLM-L6-v2",  # Embedding model
    "k": 4,                       # Number of chunks to retrieve
    "llm_model": "google/flan-t5-base"
}
```

### Add Custom Questions

Edit `benchmark_dataset.json` to add your own test questions:

```json
{
  "id": 16,
  "question": "Your question here?",
  "ground_truth": "Expected answer here.",
  "difficulty": "medium",
  "topic": "your_topic"
}
```

---

## 📊 Understanding Outputs

### Evaluation Metrics Explained

| Metric                  | Range  | What it Measures         | Higher is Better |
| ----------------------- | ------ | ------------------------ | ---------------- |
| **ROUGE-1**             | 0-1    | Word overlap             | ✅               |
| **ROUGE-2**             | 0-1    | Phrase overlap (2-grams) | ✅               |
| **ROUGE-L**             | 0-1    | Longest common sequence  | ✅               |
| **BLEU**                | 0-1    | N-gram precision         | ✅               |
| **Semantic Similarity** | 0-1    | Meaning similarity       | ✅               |
| **F1 Score**            | 0-1    | Token precision/recall   | ✅               |
| **Exact Match**         | 0 or 1 | Perfect string match     | ✅               |

### Reading Results

**evaluation_results.csv**:

```
qa_id, rouge1_f, rouge2_f, bleu, semantic_similarity, f1, ...
1,     0.6234,   0.4521,   0.38, 0.7821,              0.65, ...
```

**evaluation_results.json**:

```json
{
  "aggregate_metrics": {
    "semantic_similarity_mean": 0.7234,
    "rouge1_f_mean": 0.6123,
    ...
  }
}
```

---

## 🐛 Troubleshooting

### Error: "Module not found"

```bash
pip install -r requirements.txt
```

### Error: Out of memory

Use a smaller model:

```python
model_name = "google/flan-t5-base"  # Instead of large
```

### Error: PDF upload fails in Streamlit

- Ensure PDF is text-based (not scanned image)
- Try a different PDF
- Check file size (large PDFs may cause issues)

### Experiments taking too long

- Use `flan-t5-base` instead of `large`
- Reduce number of test questions
- Run experiments individually instead of all at once

### ImportError for evaluation libraries

```bash
pip install rouge-score sacrebleu evaluate matplotlib seaborn
```

---

## 📈 Expected Performance

### Runtime Benchmarks

- **First-time setup**: 5-10 minutes (downloading models)
- **Chatbot startup**: 1-2 minutes (loading LLM)
- **Per question answer**: 2-5 seconds
- **Evaluation (15 questions)**: 5-10 minutes
- **Full experiments (60 runs)**: 30-60 minutes

### Resource Usage

- **RAM**: 4-8 GB (flan-t5-base/large)
- **Disk**: ~5 GB (models + dependencies)
- **GPU**: Optional but speeds up inference 3-5x

---

## 🎯 Quick Command Reference

```bash
# Run chatbot
streamlit run app.py

# Run evaluation
python run_evaluation.py documents/your_document.pdf

# Run experiments
python experiments.py documents/your_document.pdf

# Test evaluation module
python evaluation.py

# Install dependencies
pip install -r requirements.txt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

---

## 📝 Adding Your Documents

Place PDF files in the `documents/` directory:

```bash
documents/
├── sample_paper.pdf
├── textbook_chapter.pdf
└── research_article.pdf
```

**Best document types for benchmark dataset**:

- Machine Learning papers
- NLP research articles
- RAG system documentation
- LLM technical guides

---

## 🔄 Workflow for Complete Evaluation

### Step-by-step Process

1. **Add your PDF**:

   ```bash
   cp /path/to/your/paper.pdf documents/
   ```

2. **Run evaluation**:

   ```bash
   python run_evaluation.py documents/paper.pdf
   ```

3. **Review results**:

   ```bash
   cat evaluation_results.json
   open evaluation_results.csv
   ```

4. **Run experiments** (optional):

   ```bash
   python experiments.py documents/paper.pdf
   ```

5. **View visualizations**:

   ```bash
   open experiment_results/*.png
   ```

6. **Update README** with your actual results

---

## 🎓 For Academic Submission

### Files to Submit

✅ Core code:

- `app.py`
- `evaluation.py`
- `experiments.py`
- `run_evaluation.py`

✅ Data & config:

- `benchmark_dataset.json`
- `requirements.txt`

✅ Documentation:

- `README.md` (with filled results)

✅ Results (after running):

- `evaluation_results.csv`
- `evaluation_results.json`
- `experiment_results/` directory

❌ Don't submit:

- `venv/` directory
- `__pycache__/` directories
- Large PDF files (unless specifically requested)

### Before Submission Checklist

- [ ] Run evaluation and fill results in README.md
- [ ] Run experiments and include output files
- [ ] Test chatbot application works
- [ ] Verify all dependencies in requirements.txt
- [ ] Remove any absolute file paths
- [ ] Update author name in README.md
- [ ] Check all code runs without errors

---

## 🆘 Getting Help

1. **Check error messages** - They usually point to the issue
2. **Verify dependencies** - Run `pip install -r requirements.txt`
3. **Check file paths** - Ensure PDFs are in `documents/` directory
4. **Review README.md** - Contains detailed documentation
5. **Test with smaller model** - Use `flan-t5-base` if `large` fails

---

**Last Updated**: November 13, 2025  
**Project**: CS501 NLP - RAG Chatbot  
**Quick Support**: Check README.md for full documentation
