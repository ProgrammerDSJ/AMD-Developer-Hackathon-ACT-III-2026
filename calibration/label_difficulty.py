"""
Script to label existing calibration prompts with difficulty
and add new L3/L4 hard probes.
Run: python calibration/label_difficulty.py
"""
import json
from pathlib import Path

CAL = Path(__file__).resolve().parent / "calibration_prompts.jsonl"
lines = [json.loads(l) for l in CAL.read_text(encoding="utf-8").splitlines() if l.strip()]


def assign_difficulty(p):
    src    = p["source"]
    prompt = p["prompt"].lower()
    if src == "mmlu":
        if any(k in prompt for k in ["simplify", "what is", "which of", "who is", "when"]):
            return "L2"
        return "L3"
    if src == "arc":
        return "L1" if len(p["prompt"].split()) < 40 else "L2"
    if src == "gsm8k":
        return "L2" if len(p["prompt"].split()) < 45 else "L3"
    if src == "humaneval":
        keywords = ["palindrome", "reverse", "count", "sum", "add"]
        return "L2" if any(k in prompt for k in keywords) else "L3"
    if src == "truthfulqa":
        return "L1"
    return "L2"


for p in lines:
    p["difficulty"] = assign_difficulty(p)


# ---------------------------------------------------------------------------
# Hard probes (L3 / L4) — expand calibration coverage
# ---------------------------------------------------------------------------
NEW_PROBES = [
    # ── Math L3 ─────────────────────────────────────────────────────────────
    {
        "prompt_id": "math_hard_001", "source": "math_hard", "difficulty": "L3",
        "prompt": "Find all real solutions to x^4 - 5x^2 + 4 = 0.",
        "reference": "2", "evaluator": "math",
    },
    {
        "prompt_id": "math_hard_002", "source": "math_hard", "difficulty": "L3",
        "prompt": (
            "A geometric sequence has first term 3 and common ratio 2. "
            "What is the sum of the first 8 terms?"
        ),
        "reference": "765", "evaluator": "math",
    },
    {
        "prompt_id": "math_hard_003", "source": "math_hard", "difficulty": "L3",
        "prompt": (
            "How many ways can 6 distinct books be arranged on a shelf if "
            "2 specific books must always be adjacent?"
        ),
        "reference": "240", "evaluator": "math",
    },
    # ── Math L4 (competition-style) ─────────────────────────────────────────
    {
        "prompt_id": "math_hard_004", "source": "math_hard", "difficulty": "L4",
        "prompt": (
            "In how many ways can 8 non-attacking rooks be placed on an "
            "8x8 chessboard? (Each row and column contains exactly one rook.)"
        ),
        "reference": "40320", "evaluator": "math",
    },
    {
        "prompt_id": "math_hard_005", "source": "math_hard", "difficulty": "L4",
        "prompt": (
            "A fair coin is flipped 10 times. What is the probability of "
            "getting exactly 6 heads? Express as a simplified fraction."
        ),
        "reference": "105/512", "evaluator": "mcq_keyword",
    },
    # ── Code L3 ─────────────────────────────────────────────────────────────
    {
        "prompt_id": "leetcode_001", "source": "leetcode", "difficulty": "L3",
        "prompt": (
            "def longest_substring_no_repeat(s: str) -> int:\n"
            "    \"\"\"Return the length of the longest substring without repeating characters.\n"
            "    longest_substring_no_repeat('abcabcbb') == 3\n"
            "    longest_substring_no_repeat('bbbbb') == 1\n"
            "    \"\"\""
        ),
        "reference": "sliding window with dict",
        "evaluator": "code",
    },
    {
        "prompt_id": "leetcode_002", "source": "leetcode", "difficulty": "L3",
        "prompt": (
            "def two_sum(nums, target):\n"
            "    \"\"\"Return indices of two numbers that sum to target.\n"
            "    two_sum([2,7,11,15], 9) => [0, 1]\n"
            "    two_sum([3,2,4], 6) => [1, 2]\n"
            "    \"\"\""
        ),
        "reference": "hash map",
        "evaluator": "code",
    },
    {
        "prompt_id": "leetcode_003", "source": "leetcode", "difficulty": "L3",
        "prompt": (
            "def is_valid_brackets(s: str) -> bool:\n"
            "    \"\"\"Return True if all brackets in s are balanced.\n"
            "    is_valid_brackets('()[]{}') == True\n"
            "    is_valid_brackets('(]') == False\n"
            "    \"\"\""
        ),
        "reference": "stack",
        "evaluator": "code",
    },
    # ── Code L4 ─────────────────────────────────────────────────────────────
    {
        "prompt_id": "leetcode_004", "source": "leetcode", "difficulty": "L4",
        "prompt": (
            "def word_break(s: str, word_dict: list) -> bool:\n"
            "    \"\"\"Return True if s can be segmented into space-separated words from word_dict.\n"
            "    word_break('leetcode', ['leet','code']) == True\n"
            "    word_break('catsandog', ['cats','dog','sand','and','cat']) == False\n"
            "    \"\"\""
        ),
        "reference": "dynamic programming",
        "evaluator": "code",
    },
    # ── Reasoning L3 (multi-hop factual) ────────────────────────────────────
    {
        "prompt_id": "reasoning_001", "source": "musique", "difficulty": "L3",
        "prompt": (
            "Alan Turing attended King's College, Cambridge. "
            "In what year was King's College founded?\n"
            "A. 1441\nB. 1209\nC. 1352\nD. 1511\nAnswer:"
        ),
        "reference": "A", "evaluator": "mcq",
    },
    {
        "prompt_id": "reasoning_002", "source": "musique", "difficulty": "L3",
        "prompt": (
            "The author of '1984' was born in which country?\n"
            "A. United Kingdom\nB. India\nC. United States\nD. France\nAnswer:"
        ),
        "reference": "B", "evaluator": "mcq",
    },
    # ── Reasoning L4 (multi-hop, requires chaining 3+ facts) ────────────────
    {
        "prompt_id": "reasoning_003", "source": "musique", "difficulty": "L4",
        "prompt": (
            "Einstein was born in Ulm, Germany. "
            "The capital of Germany is also the birthplace of which famous composer?\n"
            "A. Mozart\nB. Bach\nC. Beethoven\nD. Brahms\nAnswer:"
        ),
        "reference": "C", "evaluator": "mcq",
    },
    # ── Instruction L2 ──────────────────────────────────────────────────────
    {
        "prompt_id": "ifeval_001", "source": "ifeval", "difficulty": "L2",
        "prompt": "Rewrite the following sentence in passive voice: 'The dog bit the man.'",
        "reference": "bitten", "evaluator": "mcq_keyword",
    },
    {
        "prompt_id": "ifeval_002", "source": "ifeval", "difficulty": "L2",
        "prompt": "List 3 synonyms for the word 'happy'.",
        "reference": "joyful", "evaluator": "mcq_keyword",
    },
    # ── Instruction L3 ──────────────────────────────────────────────────────
    {
        "prompt_id": "ifeval_003", "source": "ifeval", "difficulty": "L3",
        "prompt": (
            "Summarize the following in exactly one sentence:\n"
            "'The Apollo 11 mission, launched in July 1969, successfully landed humans on "
            "the Moon for the first time, with Neil Armstrong and Buzz Aldrin walking on "
            "the lunar surface while Michael Collins orbited above.'"
        ),
        "reference": "Apollo 11", "evaluator": "mcq_keyword",
    },
    {
        "prompt_id": "ifeval_004", "source": "ifeval", "difficulty": "L3",
        "prompt": (
            "Convert the following JSON to a Python dataclass:\n"
            "{\"name\": \"Alice\", \"age\": 30, \"scores\": [95, 87, 92]}"
        ),
        "reference": "dataclass", "evaluator": "code",
    },
]

all_prompts = lines + NEW_PROBES
out = "\n".join(json.dumps(p, ensure_ascii=False) for p in all_prompts) + "\n"
CAL.write_text(out, encoding="utf-8")
print(f"Written {len(all_prompts)} prompts "
      f"({len(lines)} original + {len(NEW_PROBES)} new)")

from collections import Counter
dist = Counter((p["source"], p["difficulty"]) for p in all_prompts)
print("\nDistribution:")
for (src, lv), cnt in sorted(dist.items()):
    print(f"  {src:<14} {lv}  x{cnt}")
