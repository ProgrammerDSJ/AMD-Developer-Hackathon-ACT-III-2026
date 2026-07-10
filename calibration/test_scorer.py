"""Quick test of the improved MCQ and math extractors."""
import sys
sys.path.insert(0, 'D:/Hackathon/AMD Developer Hackathon ACT II 2026')

from calibration.run_calibration import _extract_letter, _extract_number

# --- MCQ tests (responses we saw in the diagnosis) ---
mcq_cases = [
    # (response_snippet, expected_answer, description)
    (
        "A. This supports the hypothesis...\nB. This does NOT support the hypothesis "
        "because it...\nC. This is also consistent...\nD. This supports...\n"
        "Therefore, the answer is B.",
        "B", "Step-by-step analysis, concludes with B"
    ),
    (
        "B. Joining of sperm and egg.\n\nDuring fertilization, a single sperm cell fuses...",
        "B", "Starts directly with answer letter"
    ),
    (
        "When dealing with ML models, both Lasso and Ridge can be used. "
        "Lasso (option B) is particularly suitable for feature selection "
        "because it drives coefficients to zero.",
        "B", "Answer embedded mid-text"
    ),
    (
        "The correct answer is: D\nSuccinylcholine causes prolonged paralysis.",
        "D", "Explicit 'correct answer is: D'"
    ),
    (
        "Let me analyze each option:\nA. Carbon - 2 unpaired electrons\n"
        "B. Nitrogen - 3 unpaired electrons (strongest)\nC. Neon - 0\nD. Sulfur - 2\n"
        "So the answer is B.",
        "B", "Step-by-step, so-the-answer-is B"
    ),
]

# --- Math tests ---
math_cases = [
    (
        "First, 48 clips in April.\nMay: 48/2 = 24 clips.\nTotal: 48 + 24 = 72 clips.",
        "72", "Arithmetic with = at end"
    ),
    (
        "The answer is 72.",
        "72", "Direct answer marker"
    ),
    (
        "Step 1: 3x - 7 = 11\nStep 2: 3x = 18\nStep 3: x = 6",
        "6", "Last = gives answer"
    ),
]

print("=== MCQ Extractor Tests ===")
passed = 0
for resp, expected, desc in mcq_cases:
    got = _extract_letter(resp)
    ok  = got == expected
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    print(f"  [{status}] {desc}")
    if not ok:
        print(f"         Got: '{got}'  Expected: '{expected}'")
        print(f"         Response tail: {repr(resp[-100:])}")

print(f"\n  MCQ: {passed}/{len(mcq_cases)} passed\n")

print("=== Math Extractor Tests ===")
passed = 0
for resp, expected, desc in math_cases:
    got = _extract_number(resp)
    ok  = got == expected
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    print(f"  [{status}] {desc}")
    if not ok:
        print(f"         Got: '{got}'  Expected: '{expected}'")

print(f"\n  Math: {passed}/{len(math_cases)} passed")
