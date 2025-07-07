# FinMORAL Framework Implementation

**Financial Multi-modal Reasoning with Answer Selection**

This repository implements the FinMORAL framework for advanced table question answering with multi-modal reasoning and intelligent answer selection.

## 🎯 Overview

FinMORAL is a comprehensive framework that combines:
- **Multi-modal reasoning** across tables, passages, numbers, and schema
- **Three specialized modules**: SQL execution, NumSolver arithmetic, and CoT reasoning
- **Intelligent answer selection** using Mix-SC voting and DistilBERT reranking
- **Comprehensive evaluation** with EM, TwAccuracy, and ablation studies

## 📊 Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FinMORAL Framework                       │
├─────────────────────────────────────────────────────────────┤
│  Step 1: Dataset Preparation (WTQ + FTQ)                   │
│  ├── Extract q, T, P, N, S components                      │
│  └── Create FinMORAL-compatible format                     │
├─────────────────────────────────────────────────────────────┤
│  Step 2: Candidate Generation (3 Modules)                  │
│  ├── SQL Module: Generate & execute SQL queries            │
│  ├── NumSolver: Symbolic arithmetic reasoning              │
│  └── CoT Reasoning: Step-by-step with self-consistency     │
├─────────────────────────────────────────────────────────────┤
│  Step 3: Final Answer Selection (2 Strategies)             │
│  ├── Mix-SC Voting: Consistency + heuristic scoring        │
│  └── DistilBERT Reranker: Pairwise preference learning     │
├─────────────────────────────────────────────────────────────┤
│  Step 4: Evaluation (EM + TwAccuracy + Ablations)          │
│  ├── Baseline comparison (TabLaP, GPT-4o, TAPEX, etc.)     │
│  ├── Ablation studies (w/o modules)                        │
│  ├── Cross-domain generalization                           │
│  └── Modality drop analysis                                │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Download datasets:**
   - **WTQ (WikiTableQuestions):** [Download here](https://github.com/ppasupat/WikiTableQuestions)
   - **FTQ (Filtered FeTaQA):**
     - FTQ is a filtered version of the [FeTaQA dataset](https://github.com/Yale-LILY/FeTaQA).
     - **FeTaQA** is a large-scale Free-form Table Question Answering dataset with 10K Wikipedia-based table, question, and free-form answer pairs. See the official repo: [https://github.com/Yale-LILY/FeTaQA](https://github.com/Yale-LILY/FeTaQA)
     - **License:** CC-BY-SA-4.0 ([details](https://github.com/Yale-LILY/FeTaQA/blob/main/LICENSE))
     - **Citation:**
       ```
       @article{Nan2021FeTaQAFT,
         title={FeTaQA: Free-form Table Question Answering},
         author={Nan, Linyong and Hsieh, Chiachun and Mao, Ziming and Lin, Xi Victoria and Verma, Neha and Zhang, Rui and Kryściński, Wojciech and Schoelkopf, Hailey and Kong, Riley and Tang, Xiangru and Mutuma, Mutethia and Rosand, Ben and Trindade, Isabel and Bandaru, Renusree and Cunningham, Jacob and Xiong, Caiming and Radev, Dragomir},
         journal={Transactions of the Association for Computational Linguistics},
         year={2022},
         volume={10},
         pages={35-49}
       }
       ```

3. **Set up API keys** (optional, for real LLM calls):
```bash
export GROQ_API_KEY="your_groq_key"
export TOGETHER_API_KEY="your_together_key"
export GOOGLE_API_KEY="your_google_key"
```

### Run Complete Pipeline

```bash
python run_complete_pipeline.py
```

This will execute all four steps on the **entire WTQ and FTQ datasets by default**:
1. **Data Preparation** → `step1_processed_data.jsonl`
2. **Candidate Generation** → `step2_candidates.jsonl`
3. **Final Selection** → `step3_final_results.jsonl`
4. **Evaluation** → `evaluation_report.json`

### Run Individual Steps

```bash
# Step 1: Dataset Preparation
python step1_data_preparation.py

# Step 2: Candidate Generation
python step2_candidate_generation.py

# Step 3: Final Answer Selection
python step3_ranking_finetune.py

# Step 4: Evaluation
python step4_evaluation.py
```

### ⚡ Debugging or Sampling
- By default, **all scripts process the full dataset**.
- To process only a sample (for debugging), set the `max_examples` parameter in the main function of each script (e.g., `max_examples=100`).
- The pipeline is research-grade and ready for full-dataset experiments.


## 🔧 Framework Components

### Step 1: Dataset Preparation

**Input**: WTQ and FTQ datasets  
**Output**: FinMORAL-compatible format with components:
- `q`: Question
- `T`: Table (flattened with `|` separators)
- `P`: Passage context
- `N`: Number list
- `S`: Schema metadata

**Key Features**:
- Multi-modal data extraction
- Schema alignment
- Content filtering (GPT-4 + human verification)

### Step 2: Candidate Generation

**Three Specialized Modules**:

1. **SQL Module**
   - Pointer-generator transformer (4-layer, 512-dim)
   - Generates SQL queries from (q, T, S)
   - Executes queries on table data

2. **NumSolver**
   - Symbolic arithmetic parser
   - Tree-based expression evaluation
   - Handles complex mathematical operations

3. **CoT Reasoning**
   - GPT-4.5 with self-consistency (k=5)
   - Step-by-step reasoning
   - Majority voting for final answer

### Step 3: Final Answer Selection

**Two-Strategy Approach**:

1. **Mix-SC Voting**
   - Consistency score: C(ai) = majority agreement
   - Heuristic score: H(ai) = execution success + reasoning quality
   - Combined score: C(ai) + λ × H(ai)

2. **DistilBERT Reranker**
   - Pairwise preference learning
   - Input format: [q; C; ai; aj]
   - Output: σij = frank([q; C; ai; aj])

### Step 4: Evaluation

**Comprehensive Metrics**:

- **EM (Exact Match)**: Standard accuracy metric
- **TwAccuracy**: Trustworthy answer selection
- **Ablation Studies**: Drop CoT, SQL, NumSolver, Reranker
- **Cross-Domain**: WTQ ↔ FTQ generalization
- **Modality Drop**: Remove T, P, N, S components


## 🔬 Advanced Features

### Cross-Domain Generalization

Test model performance when trained on one dataset and evaluated on another:
- **WTQ → FTQ**: SQL/arithmetic → Financial reasoning
- **FTQ → WTQ**: Financial → General table QA

### Modality Drop Analysis

Evaluate importance of each modality:
- **w/o Table (T)**: Remove table information
- **w/o Passage (P)**: Remove context passages
- **w/o Numbers (N)**: Remove numerical data
- **w/o Schema (S)**: Remove schema metadata

### Self-Consistency Voting

CoT reasoning with k=5 samples:
1. Generate 5 reasoning paths
2. Extract final answers
3. Apply majority voting
4. Select most common answer

## 🛠️ Configuration

### API Configuration

Edit the API keys in each step file:
```python
# In step2_candidate_generation.py
self.groq_api_key = "YOUR_GROQ_API_KEY_HERE"
self.together_api_key = "YOUR_TOGETHER_API_KEY_HERE"
self.google_api_key = "YOUR_GOOGLE_API_KEY_HERE"
```

### Model Configuration

Adjust model parameters:
```python
# CoT self-consistency
self.cot_samples = 5  # Number of samples for voting

# Mix-SC voting
self.lambda_weight = 0.5  # Weight for heuristic score

# DistilBERT reranker
max_length = 256  # Input sequence length
batch_size = 8    # Training batch size
epochs = 3        # Training epochs
```
## 🆘 Troubleshooting

### Common Issues

1. **Missing datasets**: Ensure WTQdata and fetaQAdata directories exist
2. **API errors**: Check API keys and rate limits
3. **Memory issues**: Reduce batch_size or max_examples
4. **CUDA errors**: Set device to CPU if GPU unavailable

### Getting Help

- Check the logs in each step file
- Verify dataset format matches expected structure
- Ensure all dependencies are installed correctly
- Review API key configuration

---

**Performance Notice:**

> ⚠️ **Performance (speed, accuracy, and resource usage) may vary depending on your device (CPU/GPU, RAM) and the specific LLM/model/API you use. Results may differ across hardware and model choices.**

---

**FinMORAL Framework** - Advancing table question answering through multi-modal reasoning and intelligent answer selection.
