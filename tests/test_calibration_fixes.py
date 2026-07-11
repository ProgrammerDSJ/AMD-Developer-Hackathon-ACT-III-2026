"""Quick test for the specific failing case + extractor anchor."""
import sys
sys.path.insert(0, ".")

from inference_wrapper.difficulty_classifier import classify
from inference_wrapper.feature_extractor import extract_features
from calibration.run_calibration import _extract_number, _extract_letter, CALIBRATION_CONFIG

print("=== Exact user prompt: 'Solve for x. 5x-10=15' ===")
prompt = "Solve for x. 5x-10=15"
feats = extract_features(prompt)
r = classify(prompt, feats)
print(f"  Classified as: {r.domain}/{r.level} (conf={r.confidence:.2f})")
assert r.level == "L1", f"Expected L1, got {r.level}"
print("  [OK] Correctly classified as math/L1")

print()
print("=== CALIBRATION_CONFIG coverage ===")
cases = [
    ("mcq",         "L1"), ("mcq",         "L2"), ("mcq",         "L3"), ("mcq",         "L4"),
    ("math",        "L1"), ("math",        "L2"), ("math",        "L3"), ("math",        "L4"),
    ("code",        "L2"), ("code",        "L3"), ("code",        "L4"),
    ("mcq_keyword", "L1"), ("mcq_keyword", "L2"), ("mcq_keyword", "L3"),
]
for ev, lv in cases:
    cfg = CALIBRATION_CONFIG.get((ev, lv))
    status = "OK" if cfg else "MISSING"
    tok = cfg["max_tokens"] if cfg else "?"
    sys_short = cfg["system"][:50] if cfg else "N/A"
    print(f"  [{status}] ({ev:<12}, {lv}) max_tokens={tok:>3}  sys: {sys_short}...")

print()
print("=== Extractor priority-0 anchor test ===")
# Simulate model responses using our system prompts
responses = [
    # MCQ L3/L4: has reasoning chain then anchor
    ("mcq", "Let me think... Photosynthesis uses sunlight.\nAnswer: B", "B"),
    # MCQ L1: bare letter
    ("mcq", "A", "A"),
    # Math L3: step-by-step then anchor
    ("math", "5x - 10 = 15\n5x = 25\nx = 5\nAnswer: 5", "5"),
    # Math L4: long reasoning then anchor
    ("math", "Sum of geometric series = a(r^n - 1)/(r-1) = 3(2^8-1)/(2-1) = 3*255 = 765\nAnswer: 765", "765"),
    # Math L1: bare number (no reasoning needed)
    ("math", "5", "5"),
    # Math tricky: intermediate numbers in chain, anchor at end
    ("math", "First: 12 * 5 = 60. Then 60 + 15 = 75. Finally 75 - 10 = 65.\nAnswer: 65", "65"),
]
all_ok = True
for eval_type, resp, expected in responses:
    if eval_type == "mcq":
        got = _extract_letter(resp)
    else:
        got = _extract_number(resp)
    ok = got == expected
    if not ok:
        all_ok = False
    print(f"  [{'OK' if ok else 'FAIL'}] {eval_type}: extracted={got!r:6} expected={expected!r}  | {resp[:50]}")

print()
print(f"All extractor tests passed: {all_ok}")
