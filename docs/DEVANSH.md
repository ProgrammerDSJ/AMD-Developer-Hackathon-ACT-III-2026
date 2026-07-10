# 🧠 DEVANSH.md — Devansh Jhawar's Task Breakdown
### HybridRouter · AMD Developer Hackathon ACT II 2026

> You own the **ML routing brain**. Your job is to build the system that learns which model is cheapest for any prompt — and prove it with data. Badal handles the runtime. You handle the intelligence behind every routing decision.

---

## 📋 Ownership Summary

| Area | What You Build |
|------|---------------|
| Hardcoded Feature Extractor | Pure Python rules-based prompt analysis |
| Data Builder Pipeline | Prompt collection → benchmark sweep → labeled dataset |
| Response Comparison Engine | Embedding similarity + LLM-as-judge + task evaluators |
| ML Router Training | XGBoost/LightGBM classifier, artifacts |
| Threshold Tuning | Optimize confidence cutoffs for best token/accuracy tradeoff |
| Accuracy Evaluators | Math, code, science, and general task evaluators |
| Feature Schema | Define and publish `feature_schema.json` for Badal |

---

## Task 1 — Hardcoded Feature Extractor

**File:** `feature_extractor/hardcoded_features.py`

The fast, free half of the feature extraction layer. Pure Python — no model calls, no latency.

### Features to extract

| Feature | Type | How |
|--------|------|-----|
| `prompt_token_count` | int | `len(prompt.split()) * 1.3` approx |
| `has_code_block` | bool | Check for backticks or `def `/ `class ` patterns |
| `has_math_symbols` | bool | Detect `=`, `∑`, `∫`, LaTeX `\frac`, `^` |
| `question_type` | str | `factual` / `instructional` / `creative` / `analytical` |
| `language` | str | `english` / `other` (simple heuristic) |
| `num_sentences` | int | Count `.`, `!`, `?` delimiters |
| `avg_word_length` | float | Mean characters per word |
| `contains_url` | bool | Regex for `http://` or `https://` |
| `complexity_heuristic` | float | Composite score from above signals |
| `num_question_marks` | int | Count of `?` — proxy for sub-question count |
| `has_numbered_list` | bool | Check for `1.`, `2.` or `- ` patterns |
| `starts_with_imperative` | bool | Starts with a verb: `Write`, `Explain`, `Solve`, etc. |

### Implementation skeleton

```python
import re

def extract_hardcoded_features(prompt: str) -> dict:
    tokens = prompt.split()
    sentences = re.split(r'[.!?]', prompt)
    
    return {
        "prompt_token_count":     int(len(tokens) * 1.3),
        "has_code_block":         bool(re.search(r'```|def |class ', prompt)),
        "has_math_symbols":       bool(re.search(r'[=∑∫\\^]|\\frac', prompt)),
        "question_type":          _classify_question_type(prompt),
        "language":               "english",  # extend if needed
        "num_sentences":          len([s for s in sentences if s.strip()]),
        "avg_word_length":        sum(len(w) for w in tokens) / max(len(tokens), 1),
        "contains_url":           bool(re.search(r'https?://', prompt)),
        "complexity_heuristic":   _compute_complexity(prompt),
        "num_question_marks":     prompt.count('?'),
        "has_numbered_list":      bool(re.search(r'\d+\.|^- ', prompt, re.MULTILINE)),
        "starts_with_imperative": _starts_with_verb(prompt),
    }
```

### Requirements

- [ ] Zero imports beyond Python stdlib
- [ ] Latency target: **< 5ms per call**
- [ ] All features are flat scalars (int, float, bool, or encoded category int)
- [ ] Expose: `extract_hardcoded_features(prompt: str) -> dict`
- [ ] Write unit tests in `tests/test_hardcoded_features.py`

---

## Task 2 — Data Builder Pipeline

**Directory:** `data_builder/`

This is your most critical task. Without labeled data, the ML router has nothing to learn from.

### Full pipeline flow

```
Prompt Collection (500–1000 prompts)
        │
        ▼
Extract hardcoded + LLM features for each prompt
        │
        ├────────────────────────────┐
        ▼                            ▼
Run prompt on Local Model    Run prompt on all Fireworks tiers
(store response)             (store response + token count per tier)
        │                            │
        └──────────────┬─────────────┘
                       ▼
        Response Comparison Engine
        (embedding similarity +
         LLM-as-judge +
         task-specific evaluators)
                       │
                       ▼
        Generate ground truth label:
        cheapest_tier_that_answered_correctly
                       │
                       ▼
        Save row to CSV / SQLite
```

### Prompt collection strategy (TBD with partner)

Target 500–1000 prompts distributed across:
- General Q&A / Factual recall
- Mathematical reasoning (from GSM8K)
- Code generation (from HumanEval)
- Scientific reasoning (from MMLU)
- Creative writing
- Multi-step instruction following

### Benchmark sweep

For each prompt, run it through all tiers and record:

```python
{
    "prompt_id": "uuid",
    "prompt": "...",
    "features": { ... },                    # hardcoded + llm features
    "local_response": "...",
    "local_correct": True/False,
    "tier1_response": "...",
    "tier1_tokens": 180,
    "tier1_correct": True/False,
    "tier2_response": "...",
    "tier2_tokens": 620,
    "tier2_correct": True/False,
    "tier3_response": "...",
    "tier3_tokens": 1100,
    "tier3_correct": True/False,
    "label": "local"                        # cheapest correct tier
}
```

Ground truth label assignment:

```python
def assign_label(row):
    if row["local_correct"]:   return "local"
    if row["tier1_correct"]:   return "tier1"
    if row["tier2_correct"]:   return "tier2"
    if row["tier3_correct"]:   return "tier3"
    return "tier3"  # fallback — try the best we have
```

### Main script

**File:** `data_builder/build_dataset.py`

```bash
python data_builder/build_dataset.py \
  --prompts data_builder/prompt_collection/prompts.jsonl \
  --output  data_builder/dataset.csv
```

### Requirements

- [ ] Deduplication — no duplicate prompts
- [ ] Resume support — skip prompts already in dataset if rerun
- [ ] Progress bar (tqdm)
- [ ] Rate limit handling for Fireworks API calls
- [ ] Save incrementally — don't lose data if it crashes halfway

---

## Task 3 — Response Comparison Engine

**Directory:** `data_builder/comparator/`

Determines whether the local model's response is equivalent to the remote (reference) response. Three complementary methods:

### Method 1 — Embedding Similarity

**File:** `data_builder/comparator/embedding_similarity.py`

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # local, fast, free

def is_equivalent_by_embedding(response_a: str, response_b: str, threshold: float = 0.85) -> bool:
    emb_a = model.encode(response_a)
    emb_b = model.encode(response_b)
    similarity = cosine_similarity([emb_a], [emb_b])[0][0]
    return similarity >= threshold
```

- Used as first-pass check (fast, free)
- If similarity > 0.85 → equivalent
- If 0.60 < similarity < 0.85 → escalate to LLM-as-judge

### Method 2 — LLM-as-Judge

**File:** `data_builder/comparator/llm_judge.py`

- Use a **small Fireworks model** (used only during data building — not at inference time)
- Structured prompt asking it to compare two responses

```python
JUDGE_PROMPT = """
Question: {prompt}

Response A: {response_a}
Response B: {response_b}

Are these two responses equivalent in accuracy and completeness?
Answer with JSON: {{"equivalent": true/false, "reason": "..."}}
"""
```

- Triggered only when embedding similarity is inconclusive
- Keeps Fireworks token cost low during data building

### Method 3 — Task-Specific Evaluators

**File:** `data_builder/comparator/task_evaluators.py`

| Task Type | Evaluator |
|-----------|-----------|
| **Math** | Extract final numerical answer with regex, compare directly |
| **Code** | Execute generated code against test cases, check pass rate |
| **Science MCQ** | Extract selected option letter (A/B/C/D), compare to answer key |
| **Factual Q&A** | Keyword presence check + embedding similarity |

```python
def evaluate_math(response: str, reference: str) -> bool:
    # Extract final number from both and compare
    ...

def evaluate_code(response: str, test_cases: list) -> bool:
    # Sandbox execute and check pass rate >= 0.8
    ...

def evaluate_mcq(response: str, correct_option: str) -> bool:
    # Find A/B/C/D in response and compare
    ...
```

### Requirements

- [ ] Each method returns `bool` (equivalent or not)
- [ ] Methods are composable — `comparator.py` orchestrates them
- [ ] Code execution is sandboxed (use `subprocess` with timeout, never `exec()` directly)
- [ ] Timeout code execution at 5 seconds

---

## Task 4 — Feature Schema Definition

**File:** `router/artifacts/feature_schema.json`

**This is the first thing you must produce** — before Badal can write his feature pipeline, he needs to know the exact column names and order.

```json
{
  "columns": [
    "prompt_token_count",
    "has_code_block",
    "has_math_symbols",
    "question_type_encoded",
    "num_sentences",
    "avg_word_length",
    "contains_url",
    "complexity_heuristic",
    "num_question_marks",
    "has_numbered_list",
    "starts_with_imperative",
    "reasoning_depth_required",
    "domain_encoded",
    "ambiguity_score",
    "requires_factual_recall",
    "task_type_encoded",
    "context_dependency"
  ],
  "categorical_encodings": {
    "question_type": {"factual": 0, "instructional": 1, "creative": 2, "analytical": 3},
    "domain": {"general": 0, "math": 1, "code": 2, "science": 3, "legal": 4, "creative": 5},
    "task_type": {"QA": 0, "generation": 1, "classification": 2, "extraction": 3}
  },
  "label_mapping": {
    "local": 0,
    "tier1": 1,
    "tier2": 2,
    "tier3": 3
  }
}
```

### Requirements

- [ ] All categorical features must be integer-encoded (ML models need numbers)
- [ ] Share this file with Badal **before he starts Task 3**
- [ ] Don't change column names after sharing — treat it as a contract

---

## Task 5 — ML Router Training

**File:** `router/train.py`

Train the classifier that powers all routing decisions.

### Model choice

**XGBoost or LightGBM** — sub-5ms inference, no GPU, small artifact.

### Training setup

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data_builder/dataset.csv")
X = df[feature_columns]
y = df["label_encoded"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="mlogloss"
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20)
```

### Outputs

- `router/artifacts/router_model.pkl` — serialized model
- `router/artifacts/feature_schema.json` — feature column spec
- Training report: accuracy per class, confusion matrix, feature importances

### Requirements

- [ ] Stratified train/val split
- [ ] Early stopping to prevent overfitting
- [ ] Save model with `joblib.dump`
- [ ] Log per-class accuracy and confusion matrix
- [ ] Log feature importance — know which features drive routing decisions
- [ ] Expose: `python router/train.py --dataset data_builder/dataset.csv`

---

## Task 6 — Threshold Tuning

**File:** `router/tune_thresholds.py`

The router outputs a probability distribution across 4 classes. Thresholds determine when to use cheaper vs. more expensive tiers.

### Goal

Find `T_local`, `T_tier1`, `T_tier2` that maximize this objective:

```
Objective = Accuracy - α × Normalized_Token_Cost
```

Where `α` balances accuracy vs. token savings.

### Grid search approach

```python
for t_local in [0.70, 0.75, 0.80, 0.85, 0.90]:
    for t_tier1 in [0.55, 0.60, 0.65, 0.70]:
        simulate_routing(val_set, t_local, t_tier1)
        compute_accuracy_and_token_cost()
        log_result()

# Pick thresholds with best objective score
```

### Requirements

- [ ] Run on validation set (never test set)
- [ ] Output a plot: accuracy vs. token cost tradeoff curve
- [ ] Write best thresholds to `configs/thresholds.yaml`
- [ ] Expose: `python router/tune_thresholds.py --model router/artifacts/router_model.pkl`

---

## Task 7 — Accuracy Evaluators

**Directory:** `evaluator/`

Used both in data building (to generate labels) and in final submission evaluation.

| File | Evaluator |
|------|-----------|
| `evaluator/math_eval.py` | Extract final numerical answer, compare |
| `evaluator/code_eval.py` | Execute against test cases, check pass rate |
| `evaluator/science_eval.py` | MCQ option extraction + comparison |
| `evaluator/general_eval.py` | Embedding similarity + keyword match |
| `evaluator/run_eval.py` | Master eval script across all task types |

### Master eval script

```bash
python evaluator/run_eval.py \
  --results results.jsonl \
  --tasks   standardized_tasks.jsonl \
  --report  eval_report.json
```

Outputs per-task-type accuracy and an overall score.

---

## 📁 Your Files

```
HybridRouter/
├── feature_extractor/
│   └── hardcoded_features.py          ← Task 1
│
├── data_builder/
│   ├── prompt_collection/
│   │   └── prompts.jsonl              ← Task 2
│   ├── comparator/
│   │   ├── embedding_similarity.py    ← Task 3
│   │   ├── llm_judge.py               ← Task 3
│   │   └── task_evaluators.py         ← Task 3
│   ├── labeler.py                     ← Task 2
│   └── build_dataset.py               ← Task 2
│
├── router/
│   ├── train.py                       ← Task 5
│   ├── evaluate.py                    ← Task 5
│   ├── tune_thresholds.py             ← Task 6
│   └── artifacts/
│       ├── router_model.pkl           ← Task 5 output
│       └── feature_schema.json        ← Task 4 output
│
└── evaluator/
    ├── math_eval.py                   ← Task 7
    ├── code_eval.py                   ← Task 7
    ├── science_eval.py                ← Task 7
    ├── general_eval.py                ← Task 7
    └── run_eval.py                    ← Task 7
```

---

## ✅ Full Task Checklist

- [ ] **Task 1:** `hardcoded_features.py` — 12 features, < 5ms, unit tested
- [ ] **Task 2:** `build_dataset.py` — full benchmark sweep, labeled CSV with 500+ rows
- [ ] **Task 3:** Comparator — embedding similarity + LLM judge + task evaluators
- [ ] **Task 4:** `feature_schema.json` — **publish this to Badal first**
- [ ] **Task 5:** `train.py` — trained router, confusion matrix, feature importances logged
- [ ] **Task 6:** `tune_thresholds.py` — optimized thresholds written to `configs/thresholds.yaml`
- [ ] **Task 7:** All evaluators working, `run_eval.py` produces a clean accuracy report

---

## 🤝 Key Integration Points with Badal

| What Badal needs from you | When |
|--------------------------|------|
| `feature_schema.json` — column names + order | **First priority** — before Badal starts Task 3 |
| `hardcoded_features.py` — working module | Before Badal starts Task 3 |
| `router_model.pkl` — trained model artifact | Before Badal starts Task 4 |
| Confirmed Fireworks model names | Early — Badal needs these for clients |

> **Produce `feature_schema.json` first.** It is the contract between your ML pipeline and Badal's runtime. Without it, he can't write the feature pipeline or the router wrapper.

---

> *"Garbage in, garbage out. Build your dataset right, and the router trains itself."*
