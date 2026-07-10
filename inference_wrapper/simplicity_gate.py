"""
inference_wrapper/simplicity_gate.py  --  Gate 0: Simplicity Pre-Filter
------------------------------------------------------------------------
Decides if a prompt is simple enough for the local model.
Three improvements over v1:
  1. Four "always-local" categories (arithmetic, ultra-short, conversational, definitional)
  2. Calibration-aware threshold (stronger model = more aggressive local routing)
  3. Per-category source-stats check (model's known weakness blocks local routing)

Signature:
  is_trivially_simple(prompt, feats, model_acc=0.0, source_stats=None)
  -> (is_simple: bool, reason: str, confidence: float)
"""

import re

# ── Always-local: pure arithmetic ────────────────────────────────────────────
_PURE_ARITH = re.compile(
    r"^[\s\(\)]*-?\d+(?:\.\d+)?"
    r"(?:\s*[\+\-\*\/\^×÷]\s*-?\d+(?:\.\d+)?)*"
    r"[\s\?=]*$"
)
_ARITH_Q = re.compile(
    r"^(?:what\s+is|what'?s|calculate|compute|evaluate|find|solve)\s+"
    r"[\d\s\+\-\*\/\^×÷\(\)\.]+\??$",
    re.IGNORECASE
)

# ── Always-local: conversational / greetings ──────────────────────────────────
_GREETINGS = {
    "hello", "hi", "hey", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "good night",
    "bye", "goodbye", "see you", "farewell", "cya",
    "thanks", "thank you", "ty", "cheers", "np", "no problem",
    "ok", "okay", "sure", "yes", "no", "yep", "nope", "yup",
    "what time is it", "how are you", "how's it going", "what's up",
}

# ── Complexity blockers ──────────────────────────────────────────────────────
_BLOCKERS = [
    "def ", "import ", "class ", "function", "algorithm", "```",
    "implement", "write a program", "write a function", "write code",
    "solve ", "integral ", "derivative ", "differentiate", "integrate",
    "proof ", "prove ", "theorem",
    "analyze", "analyse", "compare and contrast", "ethical implication",
    "trade-off", "trade off", "pros and cons", "step by step", "step-by-step",
    "explain why", "first,", "firstly,", "secondly,",
]
_HARD_TOPICS = re.compile(
    r"\b(philosophy|paradox|geopolit|consciousness|metaphysics|"
    r"epistemology|quantum mechanics|nuclear|relativity)\b",
    re.IGNORECASE
)
_MCQ = re.compile(r"\b[A-D][.)]\s", re.IGNORECASE)

_SIMPLE_PREFIXES = [
    "what is ", "what are ", "what's ", "what was ",
    "who is ", "who was ", "who are ",
    "when is ", "when was ", "when did ",
    "where is ", "where was ",
    "how many ", "how much ", "how old ", "how long ",
    "define ", "what does ", "how does ", "how do ",
    "name the ", "list the ", "which country ", "what year ",
    "is it ", "are there ", "describe ",
]


def _local_threshold(model_acc: float) -> float:
    if   model_acc >= 0.85: return 0.40
    elif model_acc >= 0.70: return 0.55
    elif model_acc >= 0.50: return 0.70
    else:                   return 0.90


def _infer_category(prompt: str, feats: dict) -> str:
    # Use features to guide category inference if possible
    task_type = feats.get("_task_type", "")
    if task_type == "math" or feats.get("has_math_symbols", 0):
        return "gsm8k"
    if task_type == "code" or feats.get("has_code_block", 0):
        return "humaneval"
    if task_type == "mcq":
        return "arc"
    
    lower = prompt.lower()
    if any(w in lower for w in ["opinion", "believe", "should", "think", "feel"]):
        return "truthfulqa"
    if re.search(r"\b(is|are|does|can|was|were)\b", lower):
        return "arc"
    return "mmlu"


def _passes_source_check(category: str, source_stats: dict) -> tuple[bool, str]:
    if category == "humaneval":
        return False, "Code tasks always route remote (HumanEval score unreliable)"

    min_acc = {"mmlu": 0.25, "arc": 0.20, "gsm8k": 0.05, "truthfulqa": 0.15}
    acc = source_stats.get(category, {}).get("acc", 0.0)
    needed = min_acc.get(category, 0.25)
    if acc < needed:
        return False, f"Model only {acc*100:.0f}% on {category} (needs {needed*100:.0f}%)"
    return True, f"Model {acc*100:.0f}% on {category} -- ok"


# ── Public API ────────────────────────────────────────────────────────────────
def is_trivially_simple(
    prompt: str,
    feats: dict,
    model_acc: float = 0.0,
    source_stats: dict | None = None,
) -> tuple[bool, str, float]:
    """
    Gate 0 decision using pre-extracted features.
    """
    p     = prompt.strip()
    lower = p.lower().strip("?!. ")
    n     = feats.get("prompt_length", len(p.split()))

    # ── Category 1: Pure arithmetic (ALWAYS local) ────────────────────────
    clean = p.replace("?", "").replace("=", "").strip()
    if _PURE_ARITH.match(clean):
        return True, "Pure arithmetic -- always local", 1.0

    if n <= 8 and _ARITH_Q.match(p):
        return True, "Simple arithmetic question -- always local", 1.0

    # ── Category 2: Conversational / greeting (ALWAYS local) ─────────────
    if lower in _GREETINGS or n <= 2:
        return True, "Greeting / conversational -- always local", 1.0

    # ── Category 3: Ultra-short, no complexity (ALWAYS local) ─────────────
    has_code_feat = bool(feats.get("has_code_block", 0))
    has_math_feat = bool(feats.get("has_math_symbols", 0))
    has_mcq_feat  = feats.get("_task_type") == "mcq"

    if n <= 4 and not has_code_feat and not has_math_feat and not has_mcq_feat:
        return True, f"Ultra-short ({n} words) -- always local", 0.95

    # ── Hard blockers: always remote ──────────────────────────────────────
    if n > 25:
        return False, f"Too long ({n} words > 25)", 0.0

    if has_mcq_feat or _MCQ.search(p):
        return False, "MCQ question -- always remote", 0.10

    content_lines = [l for l in p.split("\n") if l.strip()]
    if len(content_lines) > 2:
        return False, "Multi-line structured prompt", 0.10

    if has_code_feat:
        return False, "Complexity signal: code block detected", 0.15

    for b in _BLOCKERS:
        if b in lower:
            return False, f"Complexity signal: '{b.strip()}'", 0.15

    m = _HARD_TOPICS.search(p)
    if m:
        return False, f"Abstract/hard topic: '{m.group()}'", 0.20

    # ── Category 4: Definitional (threshold depends on model quality) ─────
    starts_simple = any(lower.startswith(pfx) for pfx in _SIMPLE_PREFIXES)
    is_short      = n <= 12
    single_sent   = feats.get("num_sentences", 1) <= 1

    score = 0.0
    if starts_simple: score += 0.55
    if is_short:      score += 0.30
    elif n <= 25:     score += 0.10
    if single_sent:   score += 0.15

    threshold = _local_threshold(model_acc)

    if score >= threshold and starts_simple:
        # Per-source stats check
        if source_stats:
            cat = _infer_category(p, feats)
            ok, stat_reason = _passes_source_check(cat, source_stats)
            if not ok:
                return False, stat_reason, round(score, 2)

        return True, (
            f"Simple definitional question "
            f"(score={score:.2f} >= threshold={threshold:.2f})"
        ), round(score, 2)

    return False, (
        f"Not simple enough "
        f"(score={score:.2f} < threshold={threshold:.2f})"
    ), round(score, 2)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    # Add root folder to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from inference_wrapper.feature_extractor import extract_features
    stats = {"mmlu": {"acc": 0.44}, "arc": {"acc": 0.32},
             "gsm8k": {"acc": 0.0}, "truthfulqa": {"acc": 0.0}}
    acc   = 0.34

    tests = [
        ("1+1",                                              True),
        ("2 + 2?",                                           True),
        ("What is 5 * 3?",                                   True),
        ("100 / 5",                                          True),
        ("Hello",                                            True),
        ("Thanks!",                                          True),
        ("What is AI?",                                      True),
        ("Who is Einstein?",                                 True),
        ("Define machine learning.",                         True),
        ("How many planets in the solar system?",            True),
        ("Okay",                                             True),
        # Remote
        ("If 3x - 7 = 11, find x.",                         False),
        ("Write a Python function that reverses a list.",    False),
        ("Analyze ethical implications of autonomous AI.",   False),
        ("What is the integral of x^2?",                    False),
        ("A. Berlin B. Paris C. Rome Which is France?",     False),
        ("Explain step by step how neural networks work.",   False),
    ]

    passed = 0
    for prompt, expected in tests:
        feats = extract_features(prompt)
        simple, reason, conf = is_trivially_simple(prompt, feats, acc, stats)
        ok = simple == expected
        passed += ok
        tag = "PASS" if ok else "FAIL"
        dest = "LOCAL " if simple else "REMOTE"
        print(f"[{tag}] ({conf:.2f}) {dest}  {prompt[:55]}")
        if not ok:
            print(f"       reason: {reason}")

    print(f"\n{passed}/{len(tests)} correct")
