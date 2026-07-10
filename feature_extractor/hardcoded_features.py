"""
hardcoded_features.py
---------------------
Pure Python, zero external dependencies.
Extracts 7 deterministic features from a raw prompt string.

Features produced:
    prompt_length         (int)   word count
    has_code_block        (int)   0 or 1
    has_math_symbols      (int)   0 or 1
    question_type         (str)   one of: factual | instructional | analytical
                                          | mathematical | creative
    question_type_encoded (int)   0-4  (same order as above)
    num_sentences         (int)   rough sentence count
    avg_word_length       (float) mean characters per word
    complexity_heuristic  (float) composite difficulty score in [0.0, 1.0]

Usage:
    from feature_extractor.hardcoded_features import extract_hardcoded_features
    feats = extract_hardcoded_features("Solve for x: 2x + 5 = 13")
"""

import re

# ---------------------------------------------------------------------------
# Question-type classification
# ---------------------------------------------------------------------------

# Patterns are checked in priority order.
# Earlier entries win when multiple patterns match.
_QT_PATTERNS = [
    # ── Mathematical ────────────────────────────────────────────────────────
    (
        "mathematical",
        re.compile(
            r"""
            (?:^|\b)
            (?:
                solve\b | calculate\b | compute\b | simplify\b |
                factoris?e\b | differentiate\b | integrate\b |
                find\s+the\s+(?:value|sum|product|derivative|integral|area|volume) |
                prove\s+that\b | what\s+is\s+the\s+(?:value|sum|product|result) |
                how\s+many\b | how\s+much\b
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    # ── Creative ────────────────────────────────────────────────────────────
    (
        "creative",
        re.compile(
            r"""
            (?:^|\b)
            (?:
                write\s+a\s+(?:short\s+)?(?:story|poem|haiku|song|limerick|essay|script|dialogue|letter|narrative) |
                compose\s+a\b | imagine\s+(?:a|that)\b | invent\s+a\b |
                come\s+up\s+with\s+a\b | create\s+a\s+(?:story|poem|character|world)
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    # ── Analytical ──────────────────────────────────────────────────────────
    (
        "analytical",
        re.compile(
            r"""
            (?:^|\b)
            (?:
                why\b | how\s+does\b | how\s+do\b | explain\s+why\b |
                compare\b | analyse\b | analyze\b | evaluate\b |
                discuss\b | what\s+are\s+the\s+(?:differences?|advantages?|disadvantages?|pros?|cons?) |
                what\s+factors\b | what\s+causes\b | what\s+is\s+the\s+impact\b |
                critically\b
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    # ── Factual ─────────────────────────────────────────────────────────────
    (
        "factual",
        re.compile(
            r"""
            (?:^|\b)
            (?:
                what\s+is\b | what\s+are\b | who\s+is\b | who\s+was\b |
                when\s+did\b | when\s+was\b | where\s+is\b | where\s+was\b |
                which\b | name\s+the\b | define\b | what\s+does\b
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
    # ── Instructional (default / catch-all) ─────────────────────────────────
    (
        "instructional",
        re.compile(
            r"""
            (?:^|\b)
            (?:
                write\b | create\b | generate\b | list\b | explain\b |
                describe\b | summarize\b | summarise\b | translate\b |
                implement\b | build\b | design\b | develop\b | convert\b |
                fix\b | debug\b | refactor\b | complete\b | fill\s+in\b
            )
            """,
            re.IGNORECASE | re.VERBOSE,
        ),
    ),
]

# Encoded integer for each question type
_QT_ENCODING = {
    "mathematical":  0,
    "creative":      1,
    "analytical":    2,
    "factual":       3,
    "instructional": 4,
}


def _classify_question_type(prompt: str) -> tuple[str, int]:
    """Return (label_str, encoded_int) for the prompt's question type."""
    head = prompt[:300]          # check only the first 300 chars — the type
                                 # signal is almost always at the start
    for label, pattern in _QT_PATTERNS:
        if pattern.search(head):
            return label, _QT_ENCODING[label]
    # Default fallback
    return "instructional", _QT_ENCODING["instructional"]


# ---------------------------------------------------------------------------
# Sentence counter
# ---------------------------------------------------------------------------

# Matches sentence-ending punctuation followed by whitespace or end-of-string.
# We use a negative look-behind to avoid splitting on decimal points (3.14)
# and common abbreviations (Mr., Dr., etc.)
_SENTENCE_SPLIT = re.compile(r'(?<![A-Z][a-z])(?<!\d)(?<!\d\.\d)[.!?](?:\s|$)')

def _count_sentences(prompt: str) -> int:
    """Count sentences using punctuation boundaries; minimum 1."""
    parts = _SENTENCE_SPLIT.split(prompt.strip())
    valid = [p for p in parts if len(p.strip()) > 5]
    return max(len(valid), 1)


# ---------------------------------------------------------------------------
# Complexity heuristic
# ---------------------------------------------------------------------------

def _compute_complexity(
    prompt_length: int,
    has_code_block: int,
    has_math_symbols: int,
    num_sentences: int,
    avg_word_length: float,
    question_type_encoded: int,
) -> float:
    """
    Composite difficulty score in [0.0, 1.0].
    Represents how likely the prompt needs a larger / more capable model.

    Weights are intentionally coarse — the ML router will learn fine-grained
    patterns from the raw features.  This column gives it a pre-computed
    human-authored hint.
    """
    score = 0.0

    # Length signal  (longer prompts = more context to track)
    if prompt_length > 50:
        score += 0.15
    if prompt_length > 150:     # additive — very long prompts get +0.30 total
        score += 0.15

    # Content type signals
    if has_code_block:
        score += 0.25           # code needs precision, not just fluency
    if has_math_symbols:
        score += 0.20           # math requires deductive reasoning

    # Structure signals
    if num_sentences >= 4:
        score += 0.10           # multi-part problems require more tracking
    if avg_word_length > 6.5:
        score += 0.15           # long words → technical / domain-specific

    # Question-type bonus (analytical + mathematical are harder for small models)
    if question_type_encoded in (0, 2):   # mathematical or analytical
        score += 0.10

    return round(min(score, 1.0), 3)


# ---------------------------------------------------------------------------
# Code-block detector
# ---------------------------------------------------------------------------

_CODE_PATTERN = re.compile(
    r"""
    ```                             |   # fenced code block
    `[^`\n]{2,}`                    |   # inline code (at least 2 chars)
    \bdef\s+\w+\s*\(                |   # Python function def
    \bclass\s+\w+                   |   # Python/Java class
    \bimport\s+\w+                  |   # Python import
    \bfrom\s+\w+\s+import\b         |   # Python from ... import
    \b#include\s*<                  |   # C/C++ include
    \bfunction\s+\w+\s*\(           |   # JS/TS function
    \bconst\s+\w+\s*=               |   # JS const declaration
    \bvar\s+\w+\s*=                 |   # JS var
    \bint\s+main\s*\(               |   # C main
    \bpublic\s+(?:static\s+)?void\b |   # Java method
    \bfn\s+\w+\s*\(                 |   # Rust fn
    \bfunc\s+\w+\s*\(               |   # Go func
    (?:SELECT|INSERT|UPDATE|DELETE)\s+  # SQL keywords
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Math-symbol detector
# ---------------------------------------------------------------------------

_MATH_PATTERN = re.compile(
    r"""
    [∑∫π√∞±×÷≤≥≠≈]                 |   # Unicode math symbols
    \\(?:frac|sum|int|sqrt|pi|       
        infty|cdot|times|div|        
        leq|geq|neq|approx)\b        |   # LaTeX commands
    \b(?:sqrt|log|ln|sin|cos|tan|   
        arcsin|arccos|arctan)\s*\(   |   # math functions
    \d+\s*[\^]\s*\d+                |   # exponentiation  (2^3)
    \d+\s*[=<>]\s*\d+               |   # numeric comparison / equation
    [a-z]\s*=\s*[-\d]               |   # variable assignment (x = 5)
    \d+\s*[\+\-\*\/]\s*\d+              # arithmetic expression
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_hardcoded_features(prompt: str) -> dict:
    """
    Extract all 7 hardcoded features from a raw prompt string.

    Args:
        prompt: The raw prompt text.

    Returns:
        dict with keys:
            prompt_length, has_code_block, has_math_symbols,
            question_type, question_type_encoded,
            num_sentences, avg_word_length, complexity_heuristic
    """
    words = prompt.split()
    n_words = max(len(words), 1)

    prompt_length        = len(words)
    has_code_block       = int(bool(_CODE_PATTERN.search(prompt)))
    has_math_symbols     = int(bool(_MATH_PATTERN.search(prompt)))
    question_type, qt_enc = _classify_question_type(prompt)
    num_sentences        = _count_sentences(prompt)
    avg_word_length      = round(sum(len(w) for w in words) / n_words, 3)
    complexity_heuristic = _compute_complexity(
        prompt_length, has_code_block, has_math_symbols,
        num_sentences, avg_word_length, qt_enc,
    )

    return {
        "prompt_length":         prompt_length,
        "has_code_block":        has_code_block,
        "has_math_symbols":      has_math_symbols,
        "question_type":         question_type,
        "question_type_encoded": qt_enc,
        "num_sentences":         num_sentences,
        "avg_word_length":       avg_word_length,
        "complexity_heuristic":  complexity_heuristic,
    }
