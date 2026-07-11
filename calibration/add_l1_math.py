"""
Adds L1 math calibration probes (simple arithmetic + single-step algebra)
to calibration_prompts.jsonl so 3B models get credit for basic math.
Run: python calibration/add_l1_math.py
"""
import json
from pathlib import Path

CAL = Path(__file__).resolve().parent / "calibration_prompts.jsonl"
prompts = [json.loads(l) for l in CAL.read_text(encoding="utf-8").splitlines() if l.strip()]

# Check we haven't already added these
existing_ids = {p.get("prompt_id") for p in prompts}

L1_MATH_PROBES = [
    # ── Pure arithmetic ───────────────────────────────────────────────────────
    {
        "prompt_id": "math_l1_001", "source": "math_l1", "difficulty": "L1",
        "prompt": "What is 15 + 27?",
        "reference": "42", "evaluator": "math",
    },
    {
        "prompt_id": "math_l1_002", "source": "math_l1", "difficulty": "L1",
        "prompt": "What is 8 times 9?",
        "reference": "72", "evaluator": "math",
    },
    {
        "prompt_id": "math_l1_003", "source": "math_l1", "difficulty": "L1",
        "prompt": "What is 144 divided by 12?",
        "reference": "12", "evaluator": "math",
    },
    {
        "prompt_id": "math_l1_004", "source": "math_l1", "difficulty": "L1",
        "prompt": "What is 20% of 80?",
        "reference": "16", "evaluator": "math",
    },
    {
        "prompt_id": "math_l1_005", "source": "math_l1", "difficulty": "L1",
        "prompt": "What is the square root of 64?",
        "reference": "8", "evaluator": "math",
    },
    # ── Single-step algebra ───────────────────────────────────────────────────
    {
        "prompt_id": "math_l1_006", "source": "math_l1", "difficulty": "L1",
        "prompt": "Solve for x: x + 5 = 12",
        "reference": "7", "evaluator": "math",
    },
    {
        "prompt_id": "math_l1_007", "source": "math_l1", "difficulty": "L1",
        "prompt": "Solve for x: 3x = 21",
        "reference": "7", "evaluator": "math",
    },
    {
        "prompt_id": "math_l1_008", "source": "math_l1", "difficulty": "L1",
        "prompt": "Solve for x: 5x - 10 = 15",
        "reference": "5", "evaluator": "math",
    },
    {
        "prompt_id": "math_l1_009", "source": "math_l1", "difficulty": "L1",
        "prompt": "Solve for x: 2x + 3 = 11",
        "reference": "4", "evaluator": "math",
    },
    {
        "prompt_id": "math_l1_010", "source": "math_l1", "difficulty": "L1",
        "prompt": "If a = 4 and b = 7, what is a + b?",
        "reference": "11", "evaluator": "math",
    },
    # ── Two-step arithmetic (still L1 — no word problem structure) ───────────
    {
        "prompt_id": "math_l1_011", "source": "math_l1", "difficulty": "L1",
        "prompt": "What is (12 + 8) * 3?",
        "reference": "60", "evaluator": "math",
    },
    {
        "prompt_id": "math_l1_012", "source": "math_l1", "difficulty": "L1",
        "prompt": "What is 100 - 37?",
        "reference": "63", "evaluator": "math",
    },
]

new_probes = [p for p in L1_MATH_PROBES if p["prompt_id"] not in existing_ids]
if not new_probes:
    print("L1 math probes already present. No changes made.")
else:
    all_prompts = prompts + new_probes
    out = "\n".join(json.dumps(p, ensure_ascii=False) for p in all_prompts) + "\n"
    CAL.write_text(out, encoding="utf-8")
    print(f"Added {len(new_probes)} L1 math probes. Total: {len(all_prompts)}")

from collections import Counter
all_data = [json.loads(l) for l in CAL.read_text(encoding="utf-8").splitlines() if l.strip()]
dist = Counter((p["source"], p.get("difficulty", "?")) for p in all_data)
print("\nFinal distribution:")
for (src, lv), cnt in sorted(dist.items()):
    print(f"  {src:<14} {lv}  x{cnt}")
