# 🧠 HybridRouter — Hybrid Token-Efficient Routing Agent
### AMD Developer Hackathon ACT III 2026 — Track 1

> **Goal:** Build an AI agent that completes a fixed set of tasks autonomously, deciding in real time which Fireworks AI model is the cheapest one that can still answer accurately. Minimize total Fireworks tokens without falling below the accuracy threshold.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Core Philosophy](#core-philosophy)
3. [Architecture Overview](#architecture-overview)
4. [Phase 1 — Data Builder Pipeline](#phase-1--data-builder-pipeline)
5. [Phase 2 — Router Training](#phase-2--router-training)
6. [Phase 3 — Inference Wrapper](#phase-3--inference-wrapper)
7. [Feature Extraction System](#feature-extraction-system)
8. [Response Comparison & Labeling](#response-comparison--labeling)
9. [Model Tiers & Routing Strategy](#model-tiers--routing-strategy)
10. [Scoring & Accuracy Evaluators](#scoring--accuracy-evaluators)
11. [Tech Stack](#tech-stack)
12. [Project Structure](#project-structure)
13. [Setup & Installation](#setup--installation)
14. [Running the Pipeline](#running-the-pipeline)
15. [Evaluation & Submission](#evaluation--submission)
16. [Team](#team)

---

## Project Overview

**HybridRouter** is an intelligent model routing system built for the AMD Developer Hackathon ACT III 2026, Track 1: *Hybrid Token-Efficient Routing Agent*.

The system answers the question: **"For any given prompt, what is the cheapest model that can still answer it accurately?"**

Instead of always routing to the most powerful (and expensive) model, HybridRouter:
- Extracts a rich feature vector from every incoming prompt
- Uses a pre-trained lightweight ML router to predict the optimal model tier
- Routes to a **local model first** (zero Fireworks tokens) when confidence is high
- Falls back to the cheapest sufficient Fireworks model when local confidence is low
- Only escalates to expensive models when the task genuinely demands it

**Scoring is based on:**
- ✅ Output accuracy across the standardized task set
- ✅ Total Fireworks tokens consumed (lower = better)
- 🏠 Local model answers = zero Fireworks tokens (best possible outcome)

---

## Core Philosophy

```
Local inference first → cheapest Fireworks model → expensive Fireworks model
       ↑                         ↑                          ↑
  Zero tokens             Low token cost              High token cost
  (best score)            (acceptable)               (only if needed)
```

The router is a **probability-based scoring system**:
1. Extract features from the prompt (fast, cheap, or free)
2. Predict a **local model confidence score** — "How likely is the local model to answer this correctly?"
3. If confidence exceeds threshold → answer locally (free)
4. If not → pick the cheapest Fireworks model tier that meets the accuracy bar

The ML router is pre-trained, frozen at submission time, and adds **<5ms** of overhead per query.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INFERENCE WRAPPER                        │
│                    (Submission Entrypoint)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │ Incoming Prompt
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION LAYER                     │
│                                                                 │
│  ┌─────────────────────────────┐  ┌───────────────────────────┐ │
│  │  Hardcoded Feature          │  │  LLM-based Feature        │ │
│  │  Extractor (Pure Python)    │  │  Extractor (≤0.6B local   │ │
│  │                             │  │  model, zero Fireworks    │ │
│  │  - prompt_count             │  │  tokens)                  │ │
│  │  - has_code_block           │  │                           │ │
│  │  - has_math_symbols         │  │  - reasoning_depth (1–5)  │ │
│  │  - question_type            │  │  - domain classification  │ │
│  │  - language detection       │  │  - ambiguity_score        │ │
│  │  - num_sentences            │  │  - requires_factual_recall│ │
│  │  - avg_word_length          │  │  - task_type              │ │
│  │  - contains_url             │  │                           │ │
│  │  - complexity_heuristic     │  │  Model: Qwen2.5-0.5B or   │ │
│  └─────────────┬───────────────┘  │  SmolLM-360M              │ │
│                │                  └─────────────┬─────────────┘ │
│                └──────────────┬─────────────────┘               │
└───────────────────────────────┼─────────────────────────────────┘
                                │ Feature Vector
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML ROUTING MODEL                             │
│              (XGBoost / LightGBM — pre-trained)                 │
│                                                                 │
│   Input: Feature Vector                                         │
│   Output: local_confidence_score + recommended_model_tier       │
│   Latency: <5ms on CPU                                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
     confidence ≥ T_high   T_low ≤ conf < T_high  conf < T_low
              │                 │                  │
              ▼                 ▼                  ▼
      ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐
      │ LOCAL MODEL  │  │ CHEAP FIREWORKS  │  │ MID/EXPENSIVE    │
      │ (Ollama)     │  │ MODEL            │  │ FIREWORKS MODEL  │
      │ 0 tokens     │  │ (llama-8b tier)  │  │ (llama-70b or    │
      └──────────────┘  └──────────────────┘  │  deepseek-r1)    │
                                              └──────────────────┘
```

---

## Phase 1 — Data Builder Pipeline

The Data Builder is responsible for generating the **labeled training dataset** that the ML router learns from. This is the most critical phase of the project.

### Pipeline Flow

```
Prompt Collection (500–1000 prompts)
            │
            ▼
   Hardcoded Feature Extractor
            │
            ▼
   LLM-based Feature Extractor (≤0.6B local model)
            │
            ├──────────────────────────┐
            ▼                          ▼
   Run on Local Model          Run on Remote Fireworks Model
   (store response)            (store response + token count)
            │                          │
            └──────────┬───────────────┘
                       ▼
         Response Comparison Engine
         (Embedding similarity +
          LLM-as-a-judge +
          Task-specific evaluators)
                       │
                       ▼
         Generate Label:
         local_confidence_score,
         cheapest_correct_fireworks_model
                       │
                       ▼
         Save to CSV / SQLite dataset
```

### Prompt Collection Strategy

> ⚠️ **TBD — To be finalized with the team.**

Candidate sources:
- Open benchmarks: MMLU, HumanEval, GSM8K, OpenHermes, TruthfulQA
- Manually curated prompts across task types
- Synthetically generated prompts covering edge cases

Target distribution across task types:
- General Q&A / Factual recall
- Mathematical reasoning
- Code generation & debugging
- Scientific reasoning
- Creative writing
- Multi-step instruction following

### Benchmark Sweep

For each prompt in the collection, we run it through **all available Fireworks model tiers** and record:
- Whether the response meets the accuracy threshold
- The token count consumed
- The response content

The **ground truth label** for each prompt is the **cheapest model that still answered correctly**. This sweep generates the labeled dataset without any manual annotation.

```
prompt_001  →  [local: ✅] → Label: local (0 tokens)
prompt_002  →  [local: ❌] [llama-8b: ✅ 180 tokens] → Label: llama-8b
prompt_003  →  [local: ❌] [llama-8b: ❌] [llama-70b: ✅ 590 tokens] → Label: llama-70b
```

---

## Phase 2 — Router Training

Using the labeled CSV from Phase 1, we train a lightweight ML classifier.

### Model Choice

**XGBoost or LightGBM** — chosen because:
- Sub-millisecond inference on CPU
- Handles tabular/feature-based data natively
- Highly interpretable (feature importance)
- No GPU required at inference time
- Small serialized artifact size (<5MB)

### Training Objective

The router is trained as a **multi-class classifier**:
- Class 0: Route to local model
- Class 1: Route to cheap Fireworks tier (e.g., llama-8b)
- Class 2: Route to mid Fireworks tier (e.g., llama-70b)
- Class 3: Route to expensive Fireworks tier (e.g., deepseek-r1)

Optionally, also trained as a **binary classifier** first:
- Class 0: Local model is sufficient
- Class 1: Fireworks model needed

### Confidence Thresholds

The router outputs a probability score, not just a class label. Thresholds are tuned to maximize the scoring objective (minimize tokens, maintain accuracy):

```python
if local_confidence >= THRESHOLD_HIGH:    # e.g., 0.85
    route_to_local()
elif local_confidence >= THRESHOLD_MID:   # e.g., 0.60
    route_to_cheap_fireworks()
else:
    route_to_mid_or_expensive_fireworks()
```

---

## Phase 3 — Inference Wrapper

The submission entrypoint. Wraps the entire routing logic and exposes a clean interface for the standardized evaluation environment.

### Flow

```python
def route_and_answer(prompt: str) -> str:
    # Step 1: Extract features (fast, free)
    features = extract_features(prompt)
    
    # Step 2: Predict routing decision
    confidence, tier = router.predict(features)
    
    # Step 3: Route and answer
    if tier == "local":
        return local_model.generate(prompt)        # 0 Fireworks tokens
    elif tier == "cheap":
        return fireworks_api.call(prompt, model=CHEAP_MODEL)
    elif tier == "mid":
        return fireworks_api.call(prompt, model=MID_MODEL)
    else:
        return fireworks_api.call(prompt, model=EXPENSIVE_MODEL)
```

---

## Feature Extraction System

### Hardcoded Features (Pure Python — Zero Cost)

| Feature | Type | Description |
|--------|------|-------------|
| `prompt_token_count` | int | Approximate token count of the prompt |
| `has_code_block` | bool | Presence of ``` or code-like syntax |
| `has_math_symbols` | bool | Presence of `=`, `∑`, `∫`, LaTeX markers |
| `question_type` | categorical | factual / instructional / creative / analytical |
| `language` | categorical | English / other |
| `num_sentences` | int | Sentence count |
| `avg_word_length` | float | Average characters per word |
| `contains_url` | bool | Presence of URLs |
| `complexity_heuristic` | float | Rule-based complexity score |
| `num_questions_marks` | int | Count of `?` — proxy for sub-question count |
| `has_numbered_list` | bool | Multi-step task signal |
| `starts_with_imperative` | bool | Command-style prompts |

### LLM-based Features (≤0.6B Local Model — Zero Fireworks Tokens)

Model: **Qwen2.5-0.5B-Instruct** or **SmolLM-360M** (runs on CPU, near-zero cost)

| Feature | Type | Description |
|--------|------|-------------|
| `reasoning_depth_required` | int (1–5) | How many reasoning steps needed |
| `domain` | categorical | science / code / math / general / legal / creative |
| `ambiguity_score` | float (0–1) | How ambiguous/underspecified the prompt is |
| `requires_factual_recall` | bool | Needs specific memorized facts |
| `task_type` | categorical | generation / classification / extraction / QA |
| `context_dependency` | bool | Requires external context to answer |

---

## Response Comparison & Labeling

This is the engine that generates ground truth labels from the benchmark sweep. Three complementary methods are used:

### 1. Embedding Similarity
- Generate sentence embeddings for local and remote responses using a local embedding model
- Compute cosine similarity
- If similarity > threshold → responses are equivalent → local model passes

```python
similarity = cosine_similarity(
    embed(local_response),
    embed(remote_response)
)
label = "local_sufficient" if similarity > 0.85 else "remote_needed"
```

### 2. LLM-as-a-Judge
- A small Fireworks model (used sparingly, only during data building — not at inference time) judges whether both responses are equivalent in quality
- Structured prompt asks the judge to score both responses and compare
- Used when embedding similarity alone is inconclusive

```
Judge Prompt:
"Given this question: [PROMPT]
Response A: [LOCAL_RESPONSE]
Response B: [REFERENCE_RESPONSE]
Are these responses equivalent in accuracy and completeness? Answer YES or NO with a brief reason."
```

### 3. Task-Specific Evaluators
Applied when the task type is deterministic:

| Task Type | Evaluator Method |
|-----------|-----------------|
| **Math / Arithmetic** | Extract final numerical answer, compare directly |
| **Coding** | Run generated code against test cases, check pass rate |
| **Science (MCQ)** | Extract selected option, compare to known correct answer |
| **Factual Q&A** | Keyword presence + embedding similarity |
| **Reasoning chains** | Check final conclusion, not intermediate steps |

---

## Model Tiers & Routing Strategy

### Fireworks AI Model Tiers

| Tier | Model (example) | Typical Use Case | Token Cost |
|------|----------------|-----------------|------------|
| 🏠 **Local** | Llama 3.2 3B / Qwen2.5 3B (Ollama) | Simple Q&A, basic factual | **0 tokens** |
| 🟢 **Cheap** | `llama-v3p1-8b-instruct` | Moderate Q&A, classification | Low |
| 🟡 **Mid** | `llama-v3p1-70b-instruct` | Reasoning, multi-step tasks | Medium |
| 🔴 **Expensive** | `deepseek-r1` or equivalent | Hard math, complex code, research | High |

> **Note:** Exact model names and costs will be updated once the hackathon's available model list is confirmed.

### Routing Decision Logic

```
Confidence ≥ 0.85  →  Local (free)
Confidence ≥ 0.60  →  Cheap Fireworks model
Confidence ≥ 0.40  →  Mid Fireworks model
Confidence < 0.40  →  Expensive Fireworks model
```

Thresholds are tunable and will be optimized on the validation set to find the best accuracy/token tradeoff.

---

## Scoring & Accuracy Evaluators

### Hackathon Scoring Formula

```
Score = Accuracy_Weight × Accuracy - Token_Weight × Normalized_Token_Count
```

The goal is to **maximize accuracy while minimizing Fireworks token usage**.

### Our Accuracy Evaluation (Internal)

| Task Category | Evaluator |
|--------------|-----------|
| Math | Final answer extraction + exact match |
| Code | Test case execution |
| Science MCQ | Option extraction + comparison |
| Factual Q&A | Embedding similarity + keyword match |
| General | LLM-as-a-judge (small model) |

---

## Tech Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| Feature extraction (hardcoded) | Python 3.11+ | Fast, no dependencies |
| Feature extraction (LLM) | Qwen2.5-0.5B via `transformers` or Ollama | Sub-0.6B, runs anywhere |
| Local inference | Ollama + Llama 3.2 3B / Qwen2.5 3B | Zero Fireworks tokens |
| ML Router | XGBoost / LightGBM | Sub-5ms, no GPU needed |
| Embedding similarity | `sentence-transformers` (local) | Free, fast |
| Remote inference | Fireworks AI Python SDK | Hackathon requirement |
| Data storage | CSV + SQLite | Simple, portable |
| Evaluation | Custom Python evaluators | Task-specific accuracy |

---

## Project Structure

```
HybridRouter/
├── README.md
│
├── data_builder/                  # Phase 1: Dataset generation
│   ├── prompt_collection/
│   │   ├── prompts.jsonl          # Raw prompt dataset
│   │   └── collect_prompts.py     # Prompt sourcing script
│   ├── feature_extractor/
│   │   ├── hardcoded_features.py  # Pure Python feature extraction
│   │   └── llm_features.py        # Sub-0.6B LLM feature extraction
│   ├── inference/
│   │   ├── local_inference.py     # Ollama local model runner
│   │   └── fireworks_inference.py # Fireworks API sweep runner
│   ├── comparator/
│   │   ├── embedding_similarity.py
│   │   ├── llm_judge.py
│   │   └── task_evaluators.py     # Math, code, science evaluators
│   ├── labeler.py                 # Combines comparisons → ground truth labels
│   └── build_dataset.py           # Main orchestration script
│
├── router/                        # Phase 2: Router training
│   ├── train.py                   # Train XGBoost/LightGBM router
│   ├── evaluate.py                # Router evaluation metrics
│   ├── tune_thresholds.py         # Optimize confidence thresholds
│   └── artifacts/
│       ├── router_model.pkl       # Serialized trained router
│       └── feature_schema.json    # Feature column spec
│
├── inference_wrapper/             # Phase 3: Submission entrypoint
│   ├── router_wrapper.py          # Main routing logic
│   ├── feature_pipeline.py        # Unified feature extraction
│   └── model_clients.py           # Local + Fireworks model clients
│
├── evaluator/                     # Accuracy evaluation tools
│   ├── math_eval.py
│   ├── code_eval.py
│   ├── science_eval.py
│   └── general_eval.py
│
├── configs/
│   ├── models.yaml                # Model tier definitions
│   ├── thresholds.yaml            # Routing confidence thresholds
│   └── fireworks.yaml             # Fireworks API config
│
├── notebooks/
│   ├── data_analysis.ipynb        # EDA on prompt dataset
│   ├── router_analysis.ipynb      # Router performance analysis
│   └── threshold_tuning.ipynb     # Threshold optimization plots
│
├── tests/
│   ├── test_feature_extractor.py
│   ├── test_comparator.py
│   └── test_router.py
│
├── requirements.txt
├── .env.example
└── Dockerfile                     # For standardized scoring environment
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed (for local inference)
- Fireworks AI API key

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd HybridRouter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull local models via Ollama
ollama pull llama3.2:3b        # Main local inference model
ollama pull qwen2.5:0.5b       # Feature extraction LLM

# Copy and configure environment
cp .env.example .env
# Add your FIREWORKS_API_KEY to .env
```

### Environment Variables

```env
FIREWORKS_API_KEY=your_api_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
LOCAL_MODEL=llama3.2:3b
FEATURE_LLM=qwen2.5:0.5b
ROUTER_THRESHOLD_HIGH=0.85
ROUTER_THRESHOLD_MID=0.60
```

---

## Running the Pipeline

### Phase 1: Build the Dataset

```bash
# Run the full data builder pipeline
python data_builder/build_dataset.py \
  --prompts data_builder/prompt_collection/prompts.jsonl \
  --output data_builder/dataset.csv \
  --sweep-models cheap,mid,expensive
```

### Phase 2: Train the Router

```bash
# Train the ML router on the generated dataset
python router/train.py \
  --dataset data_builder/dataset.csv \
  --output router/artifacts/router_model.pkl

# Evaluate and tune thresholds
python router/tune_thresholds.py \
  --model router/artifacts/router_model.pkl \
  --dataset data_builder/dataset.csv
```

### Phase 3: Run the Inference Wrapper

```bash
# Test a single prompt
python inference_wrapper/router_wrapper.py \
  --prompt "Solve: 2x + 5 = 15"

# Run on the full standardized eval set
python inference_wrapper/router_wrapper.py \
  --eval-set standardized_tasks.jsonl \
  --output results.jsonl
```

---

## Evaluation & Submission

### Local Evaluation (Before Submission)

Run the full eval pipeline locally to estimate your accuracy/token score:

```bash
python evaluator/run_eval.py \
  --results results.jsonl \
  --tasks standardized_tasks.jsonl \
  --report eval_report.json
```

### Submission Checklist

- [ ] Router model artifact is bundled (`router/artifacts/router_model.pkl`)
- [ ] Local models are pulled and available via Ollama
- [ ] All Fireworks API calls go through `FIREWORKS_BASE_URL`
- [ ] Local inference uses zero Fireworks tokens
- [ ] Accuracy meets the minimum threshold on local eval
- [ ] Dockerfile builds and runs cleanly

---

## Team

| Member | Role |
|--------|------|
| Devansh | Architecture, ML Router, Data Builder Pipeline |
| [Partner Name] | TBD — Prompt Collection Strategy, Evaluators |

**Hackathon:** AMD Developer Hackathon ACT III 2026
**Track:** Track 1 — Hybrid Token-Efficient Routing Agent
**Scoring:** Token count (Fireworks only) + Output accuracy

---

> *"The best token is the one you never spend."*
