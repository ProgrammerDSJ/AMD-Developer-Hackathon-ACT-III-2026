import json, sys
sys.path.insert(0, ".")
from calibration.run_calibration import _score, CALIBRATION_CONFIG, CALIBRATION_DEFAULT
from collections import Counter

lines = [json.loads(l) for l in open("calibration/calibration_prompts.jsonl", encoding="utf-8")]

ev_lv = Counter((p["evaluator"], p.get("difficulty","L2")) for p in lines)
missing = []
print("Config coverage:")
for (ev, lv), cnt in sorted(ev_lv.items()):
    cfg = CALIBRATION_CONFIG.get((ev, lv))
    status = "OK" if cfg else "DEFAULT"
    tok = cfg["max_tokens"] if cfg else CALIBRATION_DEFAULT["max_tokens"]
    tag = "" if cfg else "  <-- fallback"
    print(f"  [{status}] ({ev:<12}, {lv})  x{cnt:3d}  max_tokens={tok}{tag}")
    if not cfg:
        missing.append((ev, lv))

print()
print("--- Keyword evaluator tests ---")
tests = [
    ("The meal was prepared by the chef.", "prepared", "keyword", True),
    ("x = 5", "5", "math", True),
    ("B", "B", "mcq", True),
    ("The answer is A", "A", "mcq", True),
    ("Answer: C", "C", "mcq", True),
    ("green and red and blue", "green", "keyword", True),
    ("hello world", "prepared", "keyword", False),
    ("The probability is 105/512", "105/512", "keyword", True),
]
all_ok = True
for resp, ref, ev, expected in tests:
    got = _score(resp, ref, ev)
    ok = got == expected
    if not ok:
        all_ok = False
    tag = "OK" if ok else "FAIL"
    print(f"  [{tag}] {ev:<10} ref={ref!r:12} -> {got}")

print()
print(f"Missing config entries (using DEFAULT): {missing}")
print(f"Total prompts: {len(lines)}")
print(f"All scorer tests passed: {all_ok}")
