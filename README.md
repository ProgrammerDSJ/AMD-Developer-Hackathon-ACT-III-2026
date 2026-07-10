# 🧠 HybridRouter — Hybrid Token-Efficient Routing Agent
### AMD Developer Hackathon ACT II 2026 — Track 1

[![Docker Compatible](https://img.shields.io/badge/Docker-Monolithic%20Standalone-blue?style=flat&logo=docker)](#docker-containerization-monolithic-setup)
[![Python](https://img.shields.io/badge/Python-3.11%2B-green?style=flat&logo=python)](#local-setup-without-docker)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20Models-orange?style=flat)](#local-setup-without-docker)
[![Fireworks AI](https://img.shields.io/badge/Fireworks%20AI-Remote%20Tiers-red?style=flat)](#environment-variables)

> **Objective:** Build an intelligent, token-efficient routing system that answers the question: *"For any given prompt, what is the cheapest model tier (Local vs. Remote Tier 1 vs. Remote Tier 2) that can still answer it accurately?"*

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Routing Architecture](#routing-architecture)
3. [Key Features](#key-features)
4. [Docker Containerization (Monolithic Setup)](#docker-containerization-monolithic-setup)
5. [Local Setup (Without Docker)](#local-setup-without-docker)
6. [Interactive CLI Usage](#interactive-cli-usage)
7. [Automated Testing & Grader API (`--json`)](#automated-testing--grader-api---json)
8. [Calibration System](#calibration-system)
9. [Project Structure](#project-structure)
10. [Team](#team)

---

## Project Overview

**HybridRouter** is designed for the **AMD Developer Hackathon ACT II 2026**. It implements a 3-tier routing hierarchy:

1. **Local Model (Ollama):** Running CPU-efficient models locally. Cost = **0 Fireworks Tokens**.
2. **Tier 1 Remote (gpt-oss-20b):** Moderate complexity queries. Cost = **Low Token Cost**.
3. **Tier 2 Remote (glm-5p2):** Heavy reasoning/open-ended queries. Cost = **Full Token Cost**.

By intelligently pre-filtering simple queries locally and dynamically selecting the cheapest remote model for complex queries, HybridRouter delivers **60% to 70% Fireworks token savings** compared to an "always-route-to-Tier-2" baseline while preserving overall answer quality.

---

## Routing Architecture

```
                       [ User Prompt ]
                              |
                              v
                +----------------------------+
                |  Step 0: Feature Extractor |
                |  (Extracts 15 features in  |
                |   <5ms on CPU)             |
                +-------------+--------------+
                              |
                              v
                +----------------------------+
                |  Step 1: Simplicity Gate 0 |
                +-------------+--------------+
                              |
               +--------------+--------------+
               | is_simple                   | is_complex
               v                             v
      [ Local Model ]               +----------------------------+
      (0 Fireworks Tokens)          |    Step 2: ML Router       |
                                    |    (XGBoost classifier)    |
                                    +-------------+--------------+
                                                  |
                                   +--------------+--------------+
                                   | label = 0    | label = 1
                                   v              v
                            [ Tier 1 Remote ]  [ Tier 2 Remote ]
                            (gpt-oss-20b)      (glm-5p2)
```

---

## Key Features

* **Feature-First Pipeline:** Feature extraction runs exactly once at the entry point of the pipeline. Those same features are used by both the Simplicity Gate (Gate 0) and the XGBoost ML Router, preventing redundant processing.
* **Gate 0 (Simplicity Pre-Filter):** A rule-based gate targeting trivial inputs. Greetings, conversational phrases, simple arithmetic, and ultra-short queries route directly to the local model, bypassing Fireworks AI entirely.
* **Calibration-Driven Routing:** Gate 0 thresholds are *model-aware*. It queries local Ollama tags, checks calibration history, and automatically scales routing aggression:
  * Local models with high calibration accuracy (e.g., $\ge 85\%$) get relaxed thresholds, allowing more queries to route locally.
  * Weaker local models get strict thresholds, reserving local inference for near-perfect hits.
* **Category Safety Blocks:** Checks benchmark-specific accuracies (MMLU, ARC, GSM8K, TruthfulQA). If calibration shows a local model is weak at a category (e.g. 0% score on math), Gate 0 blocks local routing for that category, automatically upgrading the query to remote.
* **XGBoost ML Router:** A fast, tabular classifier trained on a stratification of 6 benchmark datasets. Adds $<5\text{ms}$ of overhead and runs easily on CPU.

---

## Docker Containerization (Monolithic Setup)

For the hackathon submission, we package everything in a single, standalone **Monolithic Docker Container**. 
* **Ollama is built-in:** The daemon runs inside the container.
* **Models are pre-downloaded:** `qwen2.5:0.5b`, `smollm2:135m`, and `smollm2:360m` are pulled during `docker build` and baked into the image.
* **Pre-Calibrated:** A pre-calibrated snapshot configuration is copied into the image, so it is active immediately without waiting for calibration.

### 1. Build the Docker Image
Execute this command in the project root:
```bash
docker build -t hybridrouter .
```

### 2. Run in Interactive Mode
Launches the full interactive CLI. Ensure you pass your Fireworks API key:
```bash
docker run -it -e FIREWORKS_API_KEY=your_fireworks_api_key_here hybridrouter
```

### 3. Run in Non-Interactive JSON Mode (For Automated Graders)
Perfect for testing scripts. Feed the prompt as arguments and get a clean JSON response back:
```bash
docker run --rm -e FIREWORKS_API_KEY=your_fireworks_api_key_here hybridrouter --json "What is 2+2?"
```

---

## Local Setup (Without Docker)

If you prefer to run the codebase directly on your host machine, follow these steps:

### 1. Prerequisites
* Python 3.11+
* [Ollama](https://ollama.com/) installed and running on the host.
* Pull the local models on your host Ollama:
  ```bash
  ollama pull qwen2.5:0.5b
  ollama pull smollm2:135m
  ollama pull smollm2:360m
  ```

### 2. Installation
```bash
# Clone the repository
git clone <repo-url>
cd AMD-Developer-Hackathon-ACT-II-2026

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# OR
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory:
```env
FIREWORKS_API_KEY=your_fireworks_api_key_here
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1
TIER1_MODEL=accounts/fireworks/models/gpt-oss-20b
TIER2_MODEL=accounts/fireworks/models/glm-5p2
```

---

## Interactive CLI Usage

To start the interactive command-line interface on your host:
```bash
python cli/main.py
```

### Startup Wizard
On start, the CLI checks Ollama. If any installed models are not yet calibrated, it offers to run the 100-prompt calibration sweep.
After checking calibration status, it displays a selection table:
```
----------------------------------- Startup -----------------------------------
+-----------------------------------------------------------------------------+
|                        |   |                            |    Cal |          |
| Local Model            |   | Status                     |    Acc | Threshold|
|------------------------+---+----------------------------+--------+----------|
| qwen2.5:0.5b           | 2 | Calibrated                 |  28.0% |     0.95 |
| smollm2:135m           | 1 | Calibrated                 |  21.0% |     0.95 |
| smollm2:360m           | 1 | Calibrated                 |  28.0% |     0.95 |
+-----------------------------------------------------------------------------+

  Select local model for this session:
  [1] qwen2.5:0.5b  cal=28%  threshold=0.95  ~5% routed locally
  ...
  Choose (1):
```

### Interactive Console Commands
Once inside the shell (`>>>`), you can type prompts or run administrative commands:
* `stats` — Prints the session's overall token efficiency, cost summary, and model distribution.
* `switch` — Prompts you to change the active local model.
* `recalibrate` — Runs the calibration suite on any model to refresh its threshold statistics.
* `exit` / `quit` — Saves session metrics and exits the program.

### Curated Demo Loop
To test the pipeline across a set of 6 pre-selected prompts covering math, MCQ, code, science, and reasoning:
```bash
python cli/main.py --demo
```

---

## Automated Testing & Grader API (`--json`)

If you are writing a grader or evaluation harness, use the `--json` flag. 

### CLI Syntax
```bash
python cli/main.py --json "Your prompt here"
```

### Output Format
The program will suppress all ASCII styling and interactive prompts. It outputs a **single, valid JSON string** to `stdout` containing key-value metrics:
```json
{
  "dest": "local",
  "tokens": 0,
  "saved": 858,
  "response": "AI stands for artificial intelligence...",
  "latency_ms": 4418.6
}
```

* `dest`: The routing decision: `"local"`, `"tier1"`, or `"tier2"`.
* `tokens`: Remote Fireworks AI tokens consumed (always `0` for local!).
* `saved`: Estimated tokens saved vs. always using the Tier 2 remote model.
* `response`: The textual answer generated by the model.
* `latency_ms`: Total execution time in milliseconds.

---

## Calibration System

Calibration calculates how well a local model performs on your specific dataset.
* Prompts are loaded from `calibration/calibration_prompts.jsonl`.
* The system evaluates responses using a **hardened code executor**, deterministic regex parse tables, and strict rule matchers.
* It computes the overall accuracy score and categorizes accuracy across MMLU, ARC, GSM8K, HumanEval, and TruthfulQA.
* The calibration stats are persisted in `~/.hybridrouter/config.json`.

To run recalibration for a specific model directly from the command line:
```bash
python cli/main.py --recalibrate "qwen2.5:0.5b"
```

---

## Project Structure

```
AMD-Developer-Hackathon-ACT-II-2026/
│
├── cli/
│   └── main.py                     # Main CLI entrypoint & JSON API
│
├── inference_wrapper/
│   ├── simplicity_gate.py          # Gate 0 pre-filter & model safety logic
│   ├── feature_extractor.py        # Tabular prompt feature extraction
│   ├── router_core.py              # XGBoost ML Router classifier
│   ├── local_client.py             # Ollama integration
│   └── fireworks_client.py         # Fireworks AI client (Tier 1 & Tier 2)
│
├── calibration/
│   ├── run_calibration.py          # Calibration evaluator runner
│   ├── extract_calibration_set.py   # Extracts prompts from benchmarks
│   ├── calibration_prompts.jsonl   # 100-prompt evaluation dataset
│   └── config_preset.json          # Pre-packaged calibration snapshot
│
├── router/
│   ├── train_router.py             # Script to fit/save XGBoost model
│   └── artifacts/
│       ├── router_model.joblib     # Serialized XGBoost model
│       └── feature_schema.json     # Feature column list
│
├── Dockerfile                      # Monolithic image definition
├── docker-compose.yml              # Sandbox setup compose
├── entrypoint.sh                   # Startup wrapper script
├── requirements.txt                # Python libraries
└── .env.example                    # Template file for API keys
```

---

## Team

* **Devansh Jhawar** — ML Routing Model, Scoring & Calibration, Code Executors
* **Badal Patel** — Tiny Feature Agent, Containerization, API Integrations, Runtime Orchestration

*Developed for the AMD Developer Hackathon ACT II 2026.*
