# DEVANSH SESSION LOGS
## Date: July 10, 2026 — Late Night Session (IST)
### Project: AMD Developer Hackathon ACT III 2026 — LLM Routing Infrastructure

> **Note:** "Badal" is the name of Devansh's project partner, not the application name.

---

## What We Discussed Today

---

### 1. Why `label` and `label_encoded` Are Empty

The two final columns in `dataset.csv` are intentionally blank. They are the **ground-truth training target** for the ML router and cannot be filled without running the Benchmark Sweep first.

**Dependency chain:**
- Call Fireworks API Tier 1, 2, and 3 for all 779 prompts
- Evaluate each response against the reference answer
- Fill `tier1_correct`, `tier2_correct`, `tier3_correct`
- Derive `label` = cheapest tier that gave a correct answer
- Derive `label_encoded` = 0 (Tier 1), 1 (Tier 2), 2 (Tier 3)

**Labeling logic:**
- Tier 1 correct → label = `tier1` (0)
- Tier 1 wrong, Tier 2 correct → label = `tier2` (1)
- Tier 1 wrong, Tier 2 wrong → label = `tier3` (2) — either Tier 3 correct or fallback

---

### 2. The Two-Stage Routing Architecture (Final Design)

We confirmed the final routing system design:

**Stage 1 — Local Gate (Runtime)**
- Runs on device at inference time
- Asks: "Can my local model handle this prompt?"
- Uses a calibrated `local_model_threshold` specific to the user's installed local model
- If confidence > threshold → run locally (free, zero latency, AMD GPU utilization)
- If confidence < threshold → escalate to Stage 2

**Stage 2 — ML Router (Fireworks Tier Classifier)**
- XGBoost or LightGBM classifier
- Input: 14 extracted features from the prompt
- Output: Tier 1 / Tier 2 / Tier 3
- Completely hardware-agnostic — labels come from Fireworks API ground truth
- This is what gets trained in Phase 3

**Why this design is hardware-agnostic:**
- The Fireworks tier labels don't care what local model the user has
- The same trained router works for any user, any device, any local model
- Only the Local Gate threshold changes per device (calibrated at first run)

---

### 3. Benchmark Sweep — Phase 3

**Confirmed next step.** The sweep will:
- Call Fireworks Tier 1 (cheap/fast), Tier 2 (mid), Tier 3 (expensive) for all 779 prompts
- Store responses in `tier1_response`, `tier2_response`, `tier3_response`
- Store token counts in `tier1_tokens`, `tier2_tokens`, `tier3_tokens`
- Evaluate correctness and store in `tier1_correct`, `tier2_correct`, `tier3_correct`
- Fill `label` and `label_encoded` using cheapest-correct-tier logic

After the sweep, train the XGBoost/LightGBM classifier on the complete dataset.

---

### 4. CLI vs Web App Decision

**Decision: CLI-first, with a local visualization dashboard for demo.**

Reasoning:
- AMD hackathon is about local hardware/AI acceleration
- The demo story is: "Your AMD GPU runs the local model, the router decides when to use it vs the cloud"
- A cloud-hosted web app kills the local hardware narrative entirely
- A CLI running on the user's machine uses their GPU and tells the AMD story correctly

**Planned CLI commands:**
- `badal calibrate` — first-run calibration against local Ollama model
- `badal ask "prompt"` — route a single prompt, show full decision trace
- `badal demo` — curated 6-prompt sequence showing different routing paths
- `badal stats` — cumulative session summary: cost saved, tier distribution, accuracy
- `badal benchmark` — run all 779 dataset prompts through the router, report aggregate metrics

**Note:** "badal" here is used as the CLI command name — Badal is actually Devansh's partner's name. The CLI command name can be changed to something else (e.g., `router`, `llmroute`, etc.).

---

### 5. Demo Design for Judges

The CLI will use Python's `rich` library for colored terminal output. Each routed prompt will show:
- The incoming prompt
- Whether the Local Gate passed or failed (local vs cloud)
- Which tier was selected and why (which features drove the decision)
- Token count, latency, and estimated cost for that call
- The actual model response below

A `demo` command cycles through 5-6 hand-picked prompts of different types — simple factual, complex mathematical, creative writing, code — so judges can see prompts routing to different tiers in one clean sequence. This runs in about 2 minutes and requires no input from the judge.

---

### 6. Token Efficiency Calculation

**How it's computed:**

For every routed prompt, we track:
- `actual_cost` = tokens used × cost per token for whichever tier handled it
- `baseline_cost` = tokens used × Tier 3 cost per token (what it would have cost without routing)
- `savings_per_prompt` = `baseline_cost - actual_cost`

Aggregate across all prompts in a session or benchmark run:
- `total_savings` = sum of all `savings_per_prompt`
- `efficiency_pct` = `(1 - actual_total_cost / baseline_total_cost) * 100`

If 60% of prompts route to Tier 1 (which is ~10x cheaper than Tier 3), you might see 80-85% efficiency.

**Accuracy preservation** is also tracked — the percentage of prompts where the cheaper routed tier still gave a correct answer vs what Tier 3 would have given. This is the quality-cost tradeoff metric.

The `stats` command will print: prompts handled, tier distribution breakdown, total tokens used, total actual cost, baseline cost (if everything went to Tier 3), savings amount, savings percentage, and local model hit rate.

---

### 7. Ollama Auto-Detection and Auto-Calibration

**Auto-detection flow:**
1. On startup, ping `http://localhost:11434/api/tags`
2. If it responds, Ollama is running
3. Parse the JSON response to get list of all pulled models
4. Select the largest model (by parameter count) as the active local model
5. Allow user to override via config if preferred

**Auto-calibration flow:**
1. Bundled file of ~100 calibration prompts with known correct answers (held-out from training set)
2. Send each to the detected local model via Ollama REST API
3. Evaluate responses against reference answers (exact match or ROUGE)
4. Compute accuracy score (e.g., 47%)
5. Map accuracy → threshold using a pre-built calibration curve (shipped with the package)
6. Write result to `~/.router/config.json`: model name, accuracy, threshold
7. On future runs: check if current Ollama model matches config → load threshold instantly → if model changed → re-calibrate

**Calibration curve:** Built offline during Phase 4 using the `local_correct` column from the dataset. It's a sigmoid or lookup table mapping local model accuracy to an optimal routing threshold. Shipped as a small JSON file.

**Self-correcting property:** A stronger local model (e.g., Kimi K2) gets high calibration accuracy → low threshold → routes locally for most prompts. A weak model (SmolLM 360m) gets low accuracy → high threshold → escalates to Fireworks almost always. No manual tuning required by any user.

---

### 8. Difficulty and Reasoning Level — Feature Analysis

**`difficulty` column:**
- Source: pulled directly from benchmark metadata (gsm8k = "medium", mmlu = "hard", alpaca = "easy")
- Completely deterministic and model-independent
- Not computed by us

**`complexity_heuristic`:**
- Rule-based calculation from `hardcoded_features.py`
- Looks at prompt length, math symbols, code blocks, multi-part structure
- Completely model-independent

**`llm_reasoning_depth`:**
- Extracted by SmolLM 360m during Phase 2 hybrid extraction
- Scale of 1-5: how many reasoning steps does this prompt structurally require
- SmolLM reads the prompt and estimates the depth of multi-step reasoning needed
- This IS somewhat relative to SmolLM's capability, but represents a property of the prompt more than the model

**Key architectural point:** `llm_reasoning_depth` is used as an input feature to train the Fireworks tier router — not the local gate. The Fireworks tier labels are ground truth from actual API calls, so any slight noise in SmolLM's ratings just means this feature has slightly lower predictive signal. XGBoost's regularization naturally handles noisy features by downweighting them.

---

### 9. The Kimi K2 / Strong Local Model Problem — Resolved

**The concern:** If a judge runs the system with Kimi K2 as their local model, does the training data become invalid because it was generated assuming a weak local model?

**Answer: No. Here's why.**

The Fireworks tier labels (the training target) are 100% independent of any local model. They represent: "which is the cheapest Fireworks API tier that answers this prompt correctly?" A judge having Kimi K2 doesn't change that ground truth at all.

The `llm_reasoning_depth` features were extracted by SmolLM but represent structural properties of the prompts. The ML model learns the relationship between those features and Fireworks tier outcomes — that relationship doesn't change based on local model strength.

The local gate threshold is self-calibrating at runtime. Kimi K2 → 91% calibration accuracy → threshold 0.35 → routes locally for ~85% of prompts. SmolLM → 43% accuracy → threshold 0.88 → escalates to Fireworks almost always. Both behaviors are correct.

**The one real limitation documented:** The `llm_reasoning_depth` feature is SmolLM-relative. Using a different feature extractor would shift the scores and require retraining. This is a known, documented tradeoff — internally consistent across all 779 rows, so the training is valid.

---

## Current Status of the Project

### Branch: `devansh-solution` (not yet merged to main)

### Dataset: `data_builder/dataset.csv`
- 779 rows, 34 columns
- All hardcoded and LLM-extracted features populated
- `label`, `label_encoded`, `tier*_response`, `tier*_tokens`, `tier*_correct` columns are empty — awaiting benchmark sweep

### Key Files:
- `feature_extractor/llm_features.py` — Hybrid extraction (SmolLM + rule-based)
- `data_builder/fill_llm_features.py` — Batch dataset enrichment script with checkpointing
- `data_builder/process_dataset.py` — Dataset schema definition
- `feature_extractor/hardcoded_features.py` — Rule-based feature extraction
- `data_builder/prompt_collection/prompts.jsonl` — 779 source prompts

---

## Immediate Next Steps (Phase 3)

1. **Build benchmark sweep script** — Call Fireworks Tier 1/2/3 for all 779 prompts, fill response/token/correct columns
2. **Run the sweep** — Requires Fireworks API key in `.env`
3. **Generate labels** — Run labeling script to fill `label` and `label_encoded`
4. **Train the router** — XGBoost/LightGBM on the labeled dataset
5. **Evaluate router performance** — Accuracy, precision/recall per tier, cost savings on test split
6. **Build calibration curve** — Use `local_correct` column to build the local gate threshold mapping
7. **Build CLI framework** — `rich` terminal UI, routing trace output, demo command
8. **Commit and merge** — After validation, merge `devansh-solution` into main

---

## Architecture Summary (One-Line Each)

- **Phase 1:** Collect 779 diverse prompts from 6 benchmark datasets ✅
- **Phase 2:** Extract 14 features per prompt using hybrid SmolLM + rule-based system ✅
- **Phase 3:** Benchmark sweep across Fireworks tiers to generate ground-truth labels ⏳ NEXT
- **Phase 4:** Train XGBoost/LightGBM router on labeled dataset ⏳ PENDING
- **Phase 5:** Build CLI with rich terminal output, local gate, Ollama auto-detection ⏳ PENDING
- **Phase 6:** Validate, document, demo for judges ⏳ PENDING

---

*Log written at end of session — July 10, 2026, ~23:45 IST*
*Next session: Start Fireworks benchmark sweep implementation*
