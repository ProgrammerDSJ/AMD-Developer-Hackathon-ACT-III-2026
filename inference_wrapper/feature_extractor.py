"""
inference_wrapper/feature_extractor.py
Rule-based feature extraction at inference time. No ML model required.
Produces the same 15-feature vector that router_model.joblib expects.
"""

import re
from typing import Dict, Any

TASK_TYPE_MAP   = {"code": 1, "math": 2, "mcq": 2, "open_ended": 0, "factual": 3}
Q_TYPE_MAP      = {"factual": 0, "creative": 1, "analytical": 2, "math": 3, "instructional": 4, "code": 5}
LLM_TASK_MAP    = {"classification": 0, "QA": 1, "generation": 2}


def extract_features(prompt: str) -> Dict[str, Any]:
    words    = prompt.split()
    sents    = [s.strip() for s in re.split(r"[.!?]+", prompt) if s.strip()]
    lower    = prompt.lower()

    # --- hardcoded features ---
    prompt_length    = len(words)
    has_code         = int(bool(re.search(r"```|def |class |import |#include|function\s*\(", prompt)))
    has_math         = int(bool(re.search(r"[=∑∫√π]|\\frac|\\sum|\$\$?|\^|\bsolve\b|\bcalculate\b|\d+\s*[\+\-\*\/]\s*\d+", prompt, re.I)))
    num_sentences    = max(len(sents), 1)
    avg_word_len     = round(sum(len(w) for w in words) / max(len(words), 1), 3)
    has_mcq          = bool(re.search(r"\bA[\.\)]\s|\bB[\.\)]\s|\bC[\.\)]\s|\bD[\.\)]\s", prompt))

    complexity = min(
        has_code * 0.30 + has_math * 0.25
        + (0.20 if prompt_length > 80  else 0)
        + (0.10 if prompt_length > 160 else 0)
        + (0.10 if num_sentences > 5   else 0)
        + (0.05 if prompt.count("?") > 1 else 0),
        1.0
    )

    # --- infer task / question type ---
    if has_code:
        task_type = "code"; q_type = "code"; llm_task = "generation"
    elif has_math or re.search(r"how many|how much|total|average|percent", lower):
        task_type = "math"; q_type = "math"; llm_task = "QA"
    elif has_mcq:
        task_type = "mcq";  q_type = "factual"; llm_task = "classification"
    elif re.match(r"(write|create|generate|compose|draft|tell me a)", lower):
        task_type = "open_ended"; q_type = "instructional"; llm_task = "generation"
    elif re.search(r"why|how does|analyze|compare|evaluate", lower):
        task_type = "factual"; q_type = "analytical"; llm_task = "QA"
    else:
        task_type = "factual"; q_type = "factual"; llm_task = "QA"

    source_task_enc  = TASK_TYPE_MAP.get(task_type, 3)
    q_type_enc       = Q_TYPE_MAP.get(q_type, 0)
    llm_task_enc     = LLM_TASK_MAP.get(llm_task, 1)

    # --- LLM-proxied features (rule-based) ---
    if has_code and prompt_length > 80:        depth = 4
    elif re.search(r"step|proof|derive|show that|multi.step", lower): depth = 4
    elif q_type == "analytical":               depth = 3
    elif has_mcq:                              depth = 2
    else:                                      depth = 2

    vague = sum(1 for w in ["something","anything","somehow","maybe","perhaps","generally"]
                if w in lower)
    ambiguity = round(min(vague * 0.15 + (0.30 if task_type == "open_ended" else 0), 1.0), 3)

    ctx_dep   = int(bool(re.search(r"\bthis\b|\bthat\b|\bthe above\b|\bbelow\b", lower))
                    and prompt_length < 30)
    factual_r = int(bool(re.search(r"who\b|when\b|where\b|what is the capital|name the|define\b|which year", lower))
                    or has_mcq)

    # --- interaction features ---
    len_x_depth   = prompt_length * depth
    cplx_x_code   = round(complexity * has_code, 3)
    math_x_depth  = has_math * depth

    feats = {
        "source_task_type_encoded":   source_task_enc,
        "prompt_length":              prompt_length,
        "has_code_block":             has_code,
        "has_math_symbols":           has_math,
        "question_type_encoded":      q_type_enc,
        "num_sentences":              num_sentences,
        "avg_word_length":            avg_word_len,
        "complexity_heuristic":       round(complexity, 3),
        "llm_reasoning_depth":        depth,
        "llm_ambiguity_score":        ambiguity,
        "llm_context_dependency":     ctx_dep,
        "llm_requires_factual_recall": factual_r,
        "llm_task_type_encoded":      llm_task_enc,
        "feat_length_x_depth":        len_x_depth,
        "feat_complexity_x_code":     cplx_x_code,
        "feat_math_x_depth":          math_x_depth,
        # human-readable (not fed to model)
        "_task_type":  task_type,
        "_q_type":     q_type,
        "_llm_task":   llm_task,
    }
    return feats
