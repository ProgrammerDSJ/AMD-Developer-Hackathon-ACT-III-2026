"""
feature_extractor/llm_features.py
-----------------------------------
Hybrid LLM feature extraction for the routing dataset.

Architecture (Hybrid — NOT purely LLM-based):
  SmolLM2:360m  → 3 soft-signal features (ambiguous / hard to rule-encode)
  Rule-based     → 2 deterministic features (high accuracy from metadata)
  DROPPED        → llm_domain (redundant with source_task_type_encoded)

Output features (5 total):
  llm_reasoning_depth      int 1-5   (SmolLM)
  llm_ambiguity_score      float 0-1 (SmolLM)
  llm_context_dependency   int 0/1   (SmolLM)
  llm_requires_factual_recall int 0/1 (rule-based)
  llm_task_type            str       (rule-based)

Fallback behaviour:
  If SmolLM fails to produce valid JSON → deterministic fallback
  values are computed from already-available hardcoded features.
"""

import json
import re
import sys
from pathlib import Path

try:
    import ollama as _ollama
    _OLLAMA_OK = True
except ImportError:
    _OLLAMA_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "smollm2:360m"

# Sources that almost always require factual recall
_FACTUAL_SOURCES = {"truthfulqa", "mmlu", "arc"}

# Source → canonical task_type mapping (MCQ-style = classification)
_MCQ_SOURCES = {"mmlu", "arc"}
_CODE_SOURCES = {"humaneval"}
_MATH_SOURCES = {"gsm8k"}

# question_type → llm_task_type
_QT_TO_TASK = {
    "mathematical": "QA",
    "factual":      "QA",
    "analytical":   "QA",
    "instructional":"generation",
    "creative":     "generation",
}

# ---------------------------------------------------------------------------
# SmolLM prompt — deliberately minimal (3 fields only)
# Fewer fields = higher parse reliability on a 360M model
# ---------------------------------------------------------------------------
_SMOLLM_TEMPLATE = """\
Analyze this user prompt. Return ONLY a JSON object, no other text.

Prompt: "{prompt}"

JSON:
{{
  "reasoning_depth": <integer 1-5, where 1=trivial, 5=complex multi-step>,
  "ambiguity_score": <float 0.0-1.0, how vague/underspecified the prompt is>,
  "context_dependency": <0 or 1, whether external context beyond the prompt is needed>
}}"""

# ---------------------------------------------------------------------------
# Rule-based feature extraction (deterministic, ~100% accuracy)
# ---------------------------------------------------------------------------

def _rule_requires_factual_recall(source: str, question_type: str) -> int:
    """
    1 if the prompt likely needs specific memorised facts.
    Signals: source is a factual benchmark OR question_type == factual.
    """
    if source.lower() in _FACTUAL_SOURCES:
        return 1
    if question_type.lower() == "factual":
        return 1
    return 0


def _rule_task_type(source: str, question_type: str, has_code_block: int) -> str:
    """
    Deterministic task_type from dataset metadata + hardcoded features.

    Returns one of: generation | classification | QA | extraction
    """
    src = source.lower()
    qt  = question_type.lower()

    # Code tasks always generate code
    if has_code_block or src in _CODE_SOURCES:
        return "generation"

    # MCQ benchmarks are classification by structure
    if src in _MCQ_SOURCES:
        return "classification"

    # Math word problems: solve and give a single answer = QA
    if src in _MATH_SOURCES or qt == "mathematical":
        return "QA"

    # Creative prompts generate open-ended text
    if qt == "creative":
        return "generation"

    # Instructional prompts almost always ask for generation
    if qt == "instructional":
        return "generation"

    # Factual / analytical → short answer
    if qt in ("factual", "analytical"):
        return "QA"

    # Safe fallback
    return "QA"


def extract_rule_based_features(
    source: str,
    question_type: str,
    has_code_block: int,
) -> dict:
    """Return the 2 rule-based LLM features as a dict."""
    return {
        "llm_requires_factual_recall": _rule_requires_factual_recall(source, question_type),
        "llm_task_type":               _rule_task_type(source, question_type, has_code_block),
    }


# ---------------------------------------------------------------------------
# SmolLM extraction (3 soft-signal features)
# ---------------------------------------------------------------------------

def _parse_smollm_output(text: str) -> dict | None:
    """Try to parse the 3-field JSON from SmolLM output. Returns None on failure."""
    text = text.strip()
    # Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # Regex fallback: find first { ... } block
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def _validate_smollm_output(data: dict) -> bool:
    """Return True only if all 3 SmolLM fields are present and in-range."""
    rd  = data.get("reasoning_depth")
    amb = data.get("ambiguity_score")
    cd  = data.get("context_dependency")
    return (
        isinstance(rd, int) and 1 <= rd <= 5 and
        isinstance(amb, (int, float)) and 0.0 <= amb <= 1.0 and
        cd in (0, 1)
    )


def _smollm_fallback(complexity_heuristic: float) -> dict:
    """
    Conservative fallback when SmolLM fails.
    reasoning_depth is estimated from complexity_heuristic (0-1 → 1-5).
    """
    rd = max(1, min(5, round(1 + complexity_heuristic * 4)))
    return {
        "llm_reasoning_depth":    rd,
        "llm_ambiguity_score":    0.3,   # moderate default
        "llm_context_dependency": 0,     # most prompts are self-contained
    }


def extract_smollm_features(
    prompt_text: str,
    complexity_heuristic: float = 0.2,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Query SmolLM2:360m via Ollama for the 3 soft-signal features.
    Returns fallback values if Ollama is unavailable or output is invalid.
    """
    if not _OLLAMA_OK:
        return _smollm_fallback(complexity_heuristic)

    filled_prompt = _SMOLLM_TEMPLATE.format(
        prompt=prompt_text[:400].replace('"', "'")   # cap + sanitise quotes
    )

    try:
        response = _ollama.generate(
            model=model,
            prompt=filled_prompt,
            options={
                "temperature": 0.05,
                "num_predict": 100,
                "top_p": 0.9,
                "stop": ["\n\n", "```"],
            },
        )
        raw = response.get("response", "")
        parsed = _parse_smollm_output(raw)
        if parsed and _validate_smollm_output(parsed):
            return {
                "llm_reasoning_depth":    int(parsed["reasoning_depth"]),
                "llm_ambiguity_score":    round(float(parsed["ambiguity_score"]), 3),
                "llm_context_dependency": int(parsed["context_dependency"]),
            }
    except Exception:
        pass   # Ollama unavailable or timeout — use fallback silently

    return _smollm_fallback(complexity_heuristic)


# ---------------------------------------------------------------------------
# Combined extractor (public API — used by fill_llm_features.py)
# ---------------------------------------------------------------------------

def extract_all_llm_features(
    prompt_text: str,
    source: str,
    question_type: str,
    has_code_block: int,
    complexity_heuristic: float = 0.2,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Extract all 5 LLM-stage features using the hybrid approach.

    Args:
        prompt_text          : Raw prompt string
        source               : Dataset source (gsm8k, humaneval, mmlu, ...)
        question_type        : Hardcoded classifier output
        has_code_block       : From hardcoded extractor (0/1)
        complexity_heuristic : From hardcoded extractor (0.0-1.0)
        model                : Ollama model tag for SmolLM calls

    Returns:
        dict with keys: llm_reasoning_depth, llm_ambiguity_score,
                        llm_context_dependency, llm_requires_factual_recall,
                        llm_task_type
    """
    rule_feats   = extract_rule_based_features(source, question_type, has_code_block)
    smollm_feats = extract_smollm_features(prompt_text, complexity_heuristic, model)

    return {**smollm_feats, **rule_feats}


# ---------------------------------------------------------------------------
# Quick self-test (run: python feature_extractor/llm_features.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        {
            "prompt": "Write a Python function to compute Fibonacci numbers using DP.",
            "source": "humaneval", "question_type": "instructional",
            "has_code_block": 1, "complexity_heuristic": 0.5,
        },
        {
            "prompt": "How many telephone calls does Jason need to make to sell 15 cars?",
            "source": "gsm8k", "question_type": "mathematical",
            "has_code_block": 0, "complexity_heuristic": 0.55,
        },
        {
            "prompt": "Pseudocholinesterase deficiency is associated with sensitivity to what?\nA.Fava beans\nB.Halothane\nC.Primaquine\nD.Succinylcholine\nAnswer:",
            "source": "mmlu", "question_type": "instructional",
            "has_code_block": 0, "complexity_heuristic": 0.25,
        },
        {
            "prompt": "Write a short poem about autumn.",
            "source": "alpaca", "question_type": "creative",
            "has_code_block": 0, "complexity_heuristic": 0.0,
        },
    ]

    print("\n--- Hybrid LLM Feature Extractor Self-Test ---\n")
    for t in tests:
        feats = extract_all_llm_features(
            t["prompt"], t["source"], t["question_type"],
            t["has_code_block"], t["complexity_heuristic"]
        )
        print(f"Source      : {t['source']}")
        print(f"Prompt      : {t['prompt'][:60]!r}")
        print(f"Features    : {feats}")
        print()
