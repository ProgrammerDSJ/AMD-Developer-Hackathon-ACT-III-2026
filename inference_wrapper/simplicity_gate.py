"""
inference_wrapper/simplicity_gate.py
--------------------------------------
Gate 0: Rule-based simplicity pre-filter.
Runs before the ML router at zero token cost (<0.5ms).

A "simple" prompt is one any small local model can answer:
  - Short (< 20 words)
  - Definitional/factual question ("What is X?", "Who is X?")
  - No code, no math, no MCQ options, no multi-step reasoning
  - Single sentence

Returns: (is_simple: bool, reason: str, confidence: float)
"""

import re


# Patterns that signal a trivially answerable question
SIMPLE_PREFIXES = [
    "what is ", "what are ", "what's ", "what was ",
    "who is ", "who was ", "who are ", "who were ",
    "when is ", "when was ", "when did ", "when were ",
    "where is ", "where was ", "where are ",
    "how many ", "how much ", "how old ", "how long ",
    "define ", "what does ", "how does ", "how do ",
    "is it ", "are there ", "name the ", "list the ",
    "what year ", "in what year ", "which country ",
    "tell me about ", "describe ",
]

# Any of these in the prompt → NOT simple
COMPLEXITY_BLOCKERS = [
    # Code
    "def ", "import ", "class ", "function", "algorithm", "```",
    "implement", "write a program", "write a function", "write code",
    # Math / equations
    "solve ", "calculate ", "integral ", "derivative ", "equation",
    "differentiate", "integrate", "proof ", "prove ",
    # Multi-step reasoning / analysis
    "analyze", "analyse", "compare and contrast", "discuss",
    "what are the implications", "ethical implication",
    "trade-off", "trade off", "pros and cons", "advantages and disadvantages",
    "step by step", "step-by-step",
    # Structured / long prompts
    "explain why", "why does", "how would you",
    "first,", "firstly,", "secondly,",
]

# MCQ option markers — if present, it's NOT trivially simple
MCQ_PATTERN = re.compile(r"\b[A-D][.)]\s", re.IGNORECASE)

# Cop-out check — don't route these to local (they'll fail)
HARD_SIGNALS = re.compile(
    r"\b(philosophy|paradox|implication|morality|ethics|consciousness|"
    r"geopolit|trade.?war|climate change|quantum|nuclear|relativity|"
    r"metaphysics|epistemology)\b",
    re.IGNORECASE
)


def is_trivially_simple(prompt: str) -> tuple[bool, str, float]:
    """
    Checks if a prompt is trivially simple for local routing.

    Returns:
        (is_simple, reason, confidence)
        - is_simple:  True if should route to local
        - reason:     human-readable explanation of the decision
        - confidence: 0.0-1.0 score of simplicity
    """
    p      = prompt.strip()
    lower  = p.lower()
    words  = p.split()
    n_words = len(words)

    # ── Hard blockers ────────────────────────────────────────────────────────

    # Too long
    if n_words > 25:
        return False, f"Too long ({n_words} words > 25)", 0.0

    # MCQ question (has A./B./C./D. options)
    if MCQ_PATTERN.search(p):
        return False, "MCQ question — not trivially answerable", 0.1

    # Multi-line (structured prompt)
    content_lines = [l for l in p.split("\n") if l.strip()]
    if len(content_lines) > 2:
        return False, "Multi-line structured prompt", 0.1

    # Complexity blockers
    for blocker in COMPLEXITY_BLOCKERS:
        if blocker in lower:
            return False, f"Complexity signal: '{blocker.strip()}'", 0.15

    # Hard abstract topics (tiny models reliably fail these)
    m = HARD_SIGNALS.search(p)
    if m:
        return False, f"Abstract/hard topic: '{m.group()}'", 0.2

    # ── Positive simplicity signals ──────────────────────────────────────────

    starts_simple = any(lower.startswith(pfx) for pfx in SIMPLE_PREFIXES)
    is_short      = n_words <= 12
    is_medium     = 12 < n_words <= 25
    single_sent   = p.count("?") + p.count(".") + p.count("!") <= 1

    # Score
    score = 0.0
    if starts_simple:  score += 0.55
    if is_short:       score += 0.30
    elif is_medium:    score += 0.10
    if single_sent:    score += 0.15

    if score >= 0.70 and starts_simple:
        return True, f"Simple {'short ' if is_short else ''}definitional question", round(score, 2)

    if score >= 0.70:
        return True, f"Short factual question", round(score, 2)

    # Not clearly simple
    return False, "Insufficient simplicity signal", round(score, 2)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        ("What is AI?",                                                    True),
        ("Who is Albert Einstein?",                                        True),
        ("Define machine learning.",                                       True),
        ("How many planets are in the solar system?",                     True),
        ("What is 2 + 2?",                                                True),
        ("What is the capital of France?\nA.Berlin\nB.Paris\nAnswer:",   False),
        ("Write a Python function that reverses a linked list.",          False),
        ("Analyze the ethical implications of autonomous AI.",            False),
        ("Solve the integral of x^2 dx from 0 to 1.",                    False),
        ("Explain step by step how gradient descent works.",              False),
        ("What are the geopolitical implications of climate change?",     False),
    ]

    ok = 0
    for prompt, expected in tests:
        simple, reason, conf = is_trivially_simple(prompt)
        status = "PASS" if simple == expected else "FAIL"
        if simple == expected:
            ok += 1
        print(f"[{status}] ({conf:.2f}) {prompt[:55]:<55} -> {'SIMPLE' if simple else 'REMOTE'}")
        if simple != expected:
            print(f"        reason: {reason}")
    print(f"\n{ok}/{len(tests)} correct")
