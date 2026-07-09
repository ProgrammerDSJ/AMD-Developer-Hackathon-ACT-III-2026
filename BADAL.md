# ⚙️ BADAL.md — Badal Patel's Task Breakdown
### HybridRouter · AMD Developer Hackathon ACT III 2026

> You own the **tiny feature agent** and the **entire runtime**. Devansh handles the ML brain. You build everything that makes it run.

---

## 📋 Ownership Summary

| Area | What You Build |
|------|---------------|
| Sub-0.6B Feature Agent | Tiny local LLM extracting nuanced prompt features |
| Model Clients | Ollama (local) + 3 Fireworks remote tier clients |
| Feature Pipeline | Unified wiring of hardcoded + LLM features |
| Runtime Orchestration | End-to-end pipeline from prompt to final answer |
| App Interface | What judges interact with |
| Containerization | Dockerfile for scoring environment |

---

## Task 1 — Sub-0.6B LLM Feature Extraction Agent

**File:** `feature_extractor/llm_features.py`

Your agent extracts semantic features that hardcoded rules can't capture.

### Output format

```python
{
    "reasoning_depth_required": 3,    # int 1-5
    "domain": "math",                 # science/code/math/general/legal/creative
    "ambiguity_score": 0.2,           # float 0-1
    "requires_factual_recall": True,  # bool
    "task_type": "generation",        # generation/classification/extraction/QA
    "context_dependency": False       # bool
}
```

### Model to use

**`Qwen2.5-0.5B-Instruct`** or **`SmolLM2-360M-Instruct`** or some other model that you're aware of and might work for our use case.
- Runs fully locally via `transformers` or Ollama
- Zero Fireworks tokens
- Runs on CPU — no GPU needed
- Load once at startup, reuse per prompt

### Implementation approach

Use a structured JSON-fill prompt — never open-ended questions:

```python
PROMPT = """Analyze this prompt and fill in the JSON. Be concise.

Prompt: "{user_prompt}"

Fill in:
{{
  "reasoning_depth_required": <1-5>,
  "domain": "<science|code|math|general|legal|creative>",
  "ambiguity_score": <0.0-1.0>,
  "requires_factual_recall": <true|false>,
  "task_type": "<generation|classification|extraction|QA>",
  "context_dependency": <true|false>
}}

JSON only:
"""
```

### Requirements

- [ ] Load model once at init — not per-call
- [ ] Set `max_new_tokens=80`
- [ ] Add JSON parse fallback to default values if malformed
- [ ] Latency target: **< 300ms on CPU**
- [ ] Expose: `extract_llm_features(prompt: str) -> dict`

---

## Task 2 — Model Clients (Local + 3 Remote Tiers)

**File:** `inference_wrapper/model_clients.py`

All 4 clients implement the same interface so the router can call them interchangeably.

### Base interface

```python
class BaseModelClient:
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        raise NotImplementedError
    def is_available(self) -> bool:
        raise NotImplementedError
```

### Client 1 — Local (Ollama, 0 Fireworks tokens)

- **Model:** `llama3.2:3b` or `qwen2.5:3b`
- **How:** HTTP to `http://localhost:11434/api/generate`
- **Fallback:** if Ollama is offline, `is_available()` returns `False`

### Client 2 — Tier 1 Low (Fireworks)

- **Model:** `accounts/fireworks/models/llama-v3p1-8b-instruct`
- **Use case:** Simple Q&A, classification

### Client 3 — Tier 2 Mid (Fireworks)

- **Model:** `accounts/fireworks/models/llama-v3p1-70b-instruct`
- **Use case:** Multi-step reasoning, moderate complexity

### Client 4 — Tier 3 High (Fireworks)

- **Model:** `accounts/fireworks/models/deepseek-r1` (confirm from hackathon list)
- **Use case:** Hard math, complex code, research-level reasoning

### Fireworks base (Tiers 1–3 share this)

```python
from openai import OpenAI

class FireworksClient(BaseModelClient):
    def __init__(self, model: str):
        self.client = OpenAI(
            api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ["FIREWORKS_BASE_URL"]
        )
        self.model = model

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content
```

### Requirements

- [ ] All 4 clients implement `BaseModelClient`
- [ ] API keys loaded from env vars — never hardcoded
- [ ] Log token usage after every Fireworks call
- [ ] Expose factory: `get_client(tier: str) -> BaseModelClient`

---

## Task 3 — Unified Feature Pipeline

**File:** `inference_wrapper/feature_pipeline.py`

Merges Devansh's hardcoded features and your LLM features into one vector for the router.

```python
def extract_all_features(prompt: str) -> dict:
    hardcoded = extract_hardcoded_features(prompt)  # Devansh's module
    llm_based  = extract_llm_features(prompt)        # Your module
    return {**hardcoded, **llm_based}
```

### Requirements

- [ ] Returns a flat dict — no nested objects
- [ ] Column order must match `router/artifacts/feature_schema.json` (Devansh provides this)
- [ ] Handle LLM extractor failures gracefully — return defaults, never crash
- [ ] Expose: `extract_all_features(prompt: str) -> dict`

---

## Task 4 — Runtime Orchestration

**File:** `inference_wrapper/router_wrapper.py`

The main entrypoint. Wires everything together. This is what judges run.

### Full flow

```
User Prompt
    │
    ▼
extract_all_features(prompt)
    │
    ▼
router.predict(features)  →  (tier, confidence)
    │
    ├── "local"  → LocalModelClient.generate(prompt)
    ├── "tier1"  → FireworksClient(TIER1).generate(prompt)
    ├── "tier2"  → FireworksClient(TIER2).generate(prompt)
    └── "tier3"  → FireworksClient(TIER3).generate(prompt)
    │
    ▼
Return answer + metadata
```

### Return format

```python
def route_and_answer(prompt: str) -> dict:
    return {
        "answer": str,           # final response text
        "tier_used": str,        # "local" / "tier1" / "tier2" / "tier3"
        "confidence": float,     # router's confidence score
        "fireworks_tokens": int  # 0 if local
    }
```

### Requirements

- [ ] Load `router_model.pkl` once at startup
- [ ] Load all 4 clients once at startup
- [ ] Check `LocalModelClient.is_available()` before routing local
- [ ] Log every routing decision (prompt hash, tier, confidence, tokens)
- [ ] CLI mode: `python router_wrapper.py --prompt "..."`
- [ ] Batch mode: `python router_wrapper.py --eval-set tasks.jsonl --output results.jsonl`
- [ ] Total overhead target: **< 500ms**

---

## Task 5 — Dockerfile

**File:** `Dockerfile`

Makes the entire system runnable in one command for the standardized scoring environment.

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl
RUN curl -fsSL https://ollama.ai/install.sh | sh

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ollama serve & sleep 5 && ollama pull llama3.2:3b

ENV FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1

CMD ["python", "inference_wrapper/router_wrapper.py", \
     "--eval-set", "tasks.jsonl", "--output", "results.jsonl"]
```

### Requirements

- [ ] CPU-only (no GPU required)
- [ ] `FIREWORKS_API_KEY` passed at runtime — never baked into image
- [ ] Container ready to serve in < 60 seconds
- [ ] Test: `docker build -t hybridrouter . && docker run -e FIREWORKS_API_KEY=... hybridrouter`

---

## 📁 Your Files

```
HybridRouter/
├── feature_extractor/
│   └── llm_features.py            ← Task 1
├── inference_wrapper/
│   ├── model_clients.py           ← Task 2
│   ├── feature_pipeline.py        ← Task 3
│   └── router_wrapper.py          ← Task 4
└── Dockerfile                     ← Task 5
```

**Files Devansh provides (don't modify):**
- `feature_extractor/hardcoded_features.py`
- `router/artifacts/router_model.pkl`
- `router/artifacts/feature_schema.json`

---

## 🤝 Key Integration Points with Devansh

| What you need from Devansh | When |
|---------------------------|------|
| `hardcoded_features.py` | Before Task 3 |
| `feature_schema.json` (column order) | Before Task 3 — agree on this first |
| `router_model.pkl` | Before Task 4 |
| Confirmed Fireworks model names | Before Task 2 |

> **Agree on `feature_schema.json` column names and order before either of you writes code.** This is the critical interface between your runtime and Devansh's ML router.

---

## ✅ Full Task Checklist

- [ ] **Task 1:** `llm_features.py` — extract 6 LLM features, JSON fallback, < 300ms
- [ ] **Task 2:** `model_clients.py` — 4 clients, shared interface, token logging
- [ ] **Task 3:** `feature_pipeline.py` — merged vector, correct column order
- [ ] **Task 4:** `router_wrapper.py` — full routing flow, CLI + batch mode
- [ ] **Task 5:** `Dockerfile` — builds, runs, < 60s startup
- [ ] End-to-end test: run a batch of 10 prompts through the full container
- [ ] Coordinate `feature_schema.json` with Devansh before writing Tasks 3 & 4

---

> *"Build it clean. Build it fast. Build it so even a Docker container runs it without drama."*
