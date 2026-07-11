"""
Quick end-to-end validation of the capability-aware calibration system.
"""
import sys
sys.path.insert(0, ".")

from inference_wrapper.difficulty_classifier import classify
from inference_wrapper.feature_extractor import extract_features
from inference_wrapper.simplicity_gate import is_trivially_simple
from calibration.profile import CapabilityProfile

print("=" * 60)
print("Test 1: Difficulty Classifier")
print("=" * 60)
tests = [
    ("Hello",                                             "language",  "L1"),
    ("What is 2 + 2?",                                   "math",      "L1"),
    ("What is the capital of France?",                   "factual",   "L1"),
    ("Write a Python function that reverses a list.",    "code",      "L2"),
    ("Explain step by step how neural networks learn.",  "reasoning", "L3"),
    ("Prove sqrt(2) is irrational using formal proofs.", "math",      "L4"),
    ("Implement a lock-free concurrent hash map in C++.","code",      "L4"),
]
passed = 0
for prompt, exp_dom, exp_lv in tests:
    r = classify(prompt)
    ok = r.domain == exp_dom and r.level == exp_lv
    if ok:
        passed += 1
    tag = "OK" if ok else "--"
    print(f"  [{tag}] {prompt[:42]:<42}  {r.domain}/{r.level}  (exp {exp_dom}/{exp_lv}  conf={r.confidence:.2f})")
print(f"  {passed}/{len(tests)} correct\n")

print("=" * 60)
print("Test 2: CapabilityProfile routing")
print("=" * 60)
p = CapabilityProfile("test-model")
p.acc["math"]["L1"] = 1.00
p.acc["math"]["L2"] = 0.92
p.acc["math"]["L3"] = 0.78
p.acc["math"]["L4"] = 0.40
p.acc["code"]["L1"] = 1.00
p.acc["code"]["L2"] = 0.75
p.acc["code"]["L3"] = 0.48
p.acc["code"]["L4"] = 0.18
p.acc["factual"]["L1"] = 1.00
p.acc["factual"]["L2"] = 0.88
p.acc["factual"]["L3"] = 0.70
p.acc["factual"]["L4"] = 0.30

checks = [
    ("math",    "L2", True),
    ("math",    "L3", True),
    ("math",    "L4", False),
    ("code",    "L2", True),
    ("code",    "L3", False),
    ("factual", "L3", True),
    ("factual", "L4", False),
]
for domain, level, expected_local in checks:
    local, reason = p.should_route_local(domain, level)
    ok = local == expected_local
    tag = "OK" if ok else "FAIL"
    dest = "LOCAL " if local else "REMOTE"
    print(f"  [{tag}] {domain}/{level}: {dest}  -> {reason[:55]}")

print(f"\nProfile summary: {p.summary()}\n")

print("=" * 60)
print("Test 3: Gate 0 with profile")
print("=" * 60)
gate_tests = [
    ("Hello",                                             True),
    ("What is 2+2?",                                      True),
    ("What is the capital of France?",                    True),   # factual/L1 -> local
    ("Write a Python function for quicksort.",            True),   # code/L2 -> local (0.75 >= 0.65)
    ("Implement a lock-free concurrent hashmap in C++.",  False),  # code/L4 -> remote
    ("Explain step by step why transformers work.",       False),  # reasoning/L3 -> depends
]
for prompt, expected in gate_tests:
    feats = extract_features(prompt)
    simple, reason, conf = is_trivially_simple(
        prompt, feats, 0.5, None, capability_profile=p
    )
    ok = simple == expected
    tag = "OK" if ok else "--"
    dest = "LOCAL " if simple else "REMOTE"
    print(f"  [{tag}] {dest} ({conf:.2f}): {prompt[:48]}")
    print(f"         -> {reason[:68]}")
