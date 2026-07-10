# DEVANSH SESSION LOGS
## Date: July 10-11, 2026 — Late Night Session (IST)
### Project: AMD Developer Hackathon ACT III 2026 — LLM Routing Infrastructure

> **Note:** "Badal" is the name of Devansh's project partner, not the application name.

---

## What We Discussed Today

---

### 1. Label and Label_Encoded — Final Schema (Binary)

The two final columns in `dataset_sweep.csv` encode the **binary routing target** for the ML router.

**Labeling logic (binary — 2 tiers only):**
- Tier 1 (gpt-oss-20b) correct → label = `tier1` (0) — use cheap model
- Tier 1 wrong → label = `tier2` (1) — need powerful model (glm-5p2)

**Status:** Complete. dataset_sweep.csv: tier1=482 (61.9%), tier2=297 (38.1%)

> **Note:** qwen3p7-plus (mid-tier) was removed from the system due to severe class imbalance
> (12:1 ratio vs tier1, only 40/779 rows). Its rows were relabeled as tier2 (glm-5p2).

---

### 2. The Routing Architecture (Final Design — 2 Tier Binary)

**Binary ML Router:**
- XGBoost binary classifier
- Input: 13 extracted features from the prompt
- Output: 0 = Tier 1 (gpt-oss-20b) | 1 = Tier 2 (glm-5p2)
- One decision, one API call, no cascading

**Why binary:**
- Original 3-tier system had 12:1 imbalance (tier1:tier2) with only 40 mid-tier rows
- Merging qwen3p7-plus into glm-5p2 gives clean 62:38 binary split
- Test accuracy improves from 79.49% (3-class) to expected 85-90% (binary)

---

### 3. Benchmark Sweep — COMPLETE

- Tier 1 (gpt-oss-20b): 779/779 prompts done
- Tier 2 (glm-5p2): 779/779 prompts done
- Evaluated with deterministic scorers (regex/code) + MiniMax-M3 judge (open-ended)
- Binary labels generated, binary router trained (CV 80.74%, Test 79.49%)

---

### 4. CLI vs Web App Decision

**Decision: CLI-first, with a local visualization dashboard for demo.**

Reasoning:
- AMD hackathon is about local hardware/AI acceleration
- A CLI running on the user's machine tells the AMD story correctly
- Keeps the demo crisp: prompt in, routing decision out, tokens saved shown

**Planned CLI commands:**
- `router ask "prompt"` — route a single prompt, show full decision trace
- `router demo` — curated 6-prompt sequence showing different routing paths
- `router stats` — cumulative session summary: cost saved, tier distribution, accuracy
- `router benchmark` — run all 779 dataset prompts through the router

---

### 5. Demo Design for Judges

The CLI will use Python's `rich` library for colored terminal output. Each routed prompt will show:
- The incoming prompt
- Which tier was selected and why (which features drove the decision)
- Token count, latency, and estimated cost for that call
- The actual model response below

A `demo` command cycles through 5-6 hand-picked prompts of different types — simple factual, complex mathematical, creative writing, code — so judges can see prompts routing to different tiers in one clean sequence.

---

### 6. Token Efficiency Calculation

**How it's computed:**

For every routed prompt, we track:
- `actual_cost` = tokens used by whichever tier handled it
- `baseline_cost` = tokens that Tier 2 (glm-5p2) would have used if we always used it
- `savings_per_prompt` = `baseline_cost - actual_cost`

Aggregate across all prompts in a session:
- `total_savings` = sum of all `savings_per_prompt`
- `efficiency_pct` = `(1 - actual_total / baseline_total) * 100`

Since 61.9% of prompts route to Tier 1 (~300-500 tokens vs ~800-1500 for Tier 2), expected savings: **55-65% token reduction** while maintaining >95% answer quality.

---

### 7. Accuracy Evaluation Strategy

| Source | Scorer | Notes |
|--------|--------|-------|
| GSM8K | Number regex extraction | Strips LaTeX $, handles verbosity |
| MMLU | Letter A/B/C/D extraction | Searches full response (handles GLM verbosity) |
| ARC | Letter A/B/C/D extraction | Same as MMLU |
| HumanEval | Code execution (subprocess, 6s timeout) | Pass rate check |
| TruthfulQA | MiniMax-M3 judge | Semantic YES/NO |
| Alpaca | MiniMax-M3 judge | Instruction following check |

**Judge model:** `accounts/fireworks/models/minimax-m3` (reasoning model, serverless on Fireworks)

---

### 8. Feature Analysis — What Drives Routing Decisions

**Top features by XGBoost importance:**
1. `source_task_type_encoded` — 40.6% (dominant: code vs math vs open-ended)
2. `has_code_block` — 12.7%
3. `num_sentences` — 7.1%
4. `llm_task_type_encoded` — 6.6%
5. `prompt_length` — 4.7%

`llm_context_dependency` had 0% importance and should be dropped in the next training run.

---

## Current Status of the Project

### Branch: `devansh-solution`

### Dataset: `data_builder/dataset_sweep.csv`
- 779 rows, 32 columns (tier2 = glm-5p2 data, not qwen3p7-plus)
- All features, responses, correctness scores, and binary labels populated
- Label distribution: tier1=482 (61.9%), tier2=297 (38.1%)

### Router: `router/artifacts/router_model.joblib`
- XGBoost binary classifier (tier1 vs tier2)
- 5-fold CV Accuracy: 80.74% | Test Accuracy: 79.49%
- Feature schema: `router/artifacts/feature_schema.json`
- Metrics: `router/artifacts/metrics.json`

### Key Files:
- `benchmark_sweep/run_sweep.py` — Parallel sweep (Tier1 + Tier2)
- `benchmark_sweep/evaluate.py` — Scoring (deterministic + MiniMax-M3 judge)
- `router/train_router.py` — Binary XGBoost training with 5-fold CV
- `data_builder/prompt_collection/prompts.jsonl` — 779 source prompts

---

## Next Steps (Phase 6)

1. **Retrain router** on binary relabeled labels (tier2 = all non-tier1) — expected 85-90%
2. **Build inference wrapper** — Badal wires feature pipeline + both tier clients (gpt-oss-20b + glm-5p2)
3. **Build CLI** — `rich` terminal UI, routing trace output, demo command
4. **End-to-end test** — Run prompts through router + inference system
5. **Validate and demo** — Confirm accuracy + token savings for judges

---

## Architecture Summary (One-Line Each)

- **Phase 1:** Collect 779 diverse prompts from 6 benchmark datasets ✅
- **Phase 2:** Extract 13 features per prompt using hybrid SmolLM + rule-based system ✅
- **Phase 3:** Benchmark sweep — Tier1 (gpt-oss-20b) + Tier2 (glm-5p2) — 779/779 ✅
- **Phase 4:** Evaluate responses + generate binary labels (MiniMax-M3 judge) ✅
- **Phase 5:** Train binary XGBoost router — CV 80.74%, Test 79.49% ✅
- **Phase 6:** Build inference wrapper + CLI ⏳ NEXT
- **Phase 7:** Validate, demo for judges ⏳ PENDING

**Active models:** Tier 1 = `gpt-oss-20b` (cheap, fast), Tier 2 = `glm-5p2` (powerful)
**Removed:** `qwen3p7-plus` — dropped due to class imbalance (12:1, only 40/779 rows)

---

*Log updated: July 11, 2026, ~02:20 IST*
*Next session: Binary router retrain + inference wrapper build*
