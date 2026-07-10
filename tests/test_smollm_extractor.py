"""
test_smollm_extractor.py
------------------------
Evaluates SmolLM2:360m via Ollama on the exact feature-extraction task
we need for the LLM feature stage.

Tests:
  1. JSON format compliance  (can it output valid JSON?)
  2. Value validity           (are values in expected ranges?)
  3. Inference speed          (seconds per prompt)
  4. Consistency              (same prompt twice → same domain/task_type?)
  5. Comparison with 135m    (qualitative notes)

Run:
    python tests/test_smollm_extractor.py
"""

import json
import time
import re

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Test prompts — 8 diverse types
# ---------------------------------------------------------------------------
TEST_PROMPTS = [
    {
        "id": "math_word_problem",
        "prompt": "Jason works as a salesperson. He needs to sell 15 cars to earn a bonus. For every 25 phone calls he makes, one person visits. For every 2 visitors, one buys a car. How many phone calls does Jason need to make?",
        "expected_domain": "math",
        "expected_task_type": "QA",
        "expected_reasoning_min": 3,
    },
    {
        "id": "code_generation",
        "prompt": "def fib(n):\n    '''Return the nth Fibonacci number using dynamic programming. Do not use recursion.'''",
        "expected_domain": "code",
        "expected_task_type": "generation",
        "expected_reasoning_min": 2,
    },
    {
        "id": "science_mcq",
        "prompt": "Pseudocholinesterase deficiency is associated with increased sensitivity to what?\nA. Fava beans\nB. Halothane\nC. Primaquine\nD. Succinylcholine\nAnswer:",
        "expected_domain": "science",
        "expected_task_type": "classification",
        "expected_reasoning_min": 1,
    },
    {
        "id": "creative_writing",
        "prompt": "Write a short poem about autumn leaves falling in the wind.",
        "expected_domain": "creative",
        "expected_task_type": "generation",
        "expected_reasoning_min": 1,
    },
    {
        "id": "factual_recall",
        "prompt": "Who won the 2032 U.S. presidential election?",
        "expected_domain": "general",
        "expected_task_type": "QA",
        "expected_reasoning_min": 1,
    },
    {
        "id": "physics_hard",
        "prompt": "Traveling at an initial speed of 1.5e6 m/s, a proton enters a region of constant magnetic field of magnitude 1.0 T. If the proton's velocity makes 30° with B, compute the proton's speed 4 seconds after entering the magnetic field.",
        "expected_domain": "science",
        "expected_task_type": "QA",
        "expected_reasoning_min": 3,
    },
    {
        "id": "general_instructional",
        "prompt": "Explain the process of photosynthesis in 3 sentences or less.",
        "expected_domain": "science",
        "expected_task_type": "generation",
        "expected_reasoning_min": 1,
    },
    {
        "id": "ambiguous_open",
        "prompt": "What's on your mind right now?",
        "expected_domain": "general",
        "expected_task_type": "generation",
        "expected_reasoning_min": 1,
    },
]

# ---------------------------------------------------------------------------
# Extraction prompt template
# ---------------------------------------------------------------------------
EXTRACTION_TEMPLATE = """\
Analyze the following user prompt and return ONLY a JSON object. No explanation, no markdown, no extra text.

User Prompt: "{prompt}"

Return this JSON with filled values:
{{
  "reasoning_depth": <integer 1-5>,
  "domain": "<math|code|science|general|creative>",
  "ambiguity_score": <float 0.0-1.0>,
  "requires_factual_recall": <0 or 1>,
  "task_type": "<QA|generation|classification|extraction>",
  "context_dependency": <0 or 1>
}}"""

VALID_DOMAINS = {"math", "code", "science", "general", "creative"}
VALID_TASK_TYPES = {"QA", "generation", "classification", "extraction"}

# ---------------------------------------------------------------------------
# JSON extraction (robust)
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict | None:
    """Try to parse a JSON dict from model output, handling noise."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find first {...} block
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def validate_features(feat: dict) -> list[str]:
    """Return list of validation errors (empty = all good)."""
    errors = []
    rd = feat.get("reasoning_depth")
    if not isinstance(rd, int) or not (1 <= rd <= 5):
        errors.append(f"reasoning_depth={rd!r} not int in [1,5]")

    dom = feat.get("domain")
    if dom not in VALID_DOMAINS:
        errors.append(f"domain={dom!r} not in {VALID_DOMAINS}")

    amb = feat.get("ambiguity_score")
    if not isinstance(amb, (int, float)) or not (0.0 <= amb <= 1.0):
        errors.append(f"ambiguity_score={amb!r} not float in [0,1]")

    fr = feat.get("requires_factual_recall")
    if fr not in (0, 1):
        errors.append(f"requires_factual_recall={fr!r} not 0/1")

    tt = feat.get("task_type")
    if tt not in VALID_TASK_TYPES:
        errors.append(f"task_type={tt!r} not in {VALID_TASK_TYPES}")

    cd = feat.get("context_dependency")
    if cd not in (0, 1):
        errors.append(f"context_dependency={cd!r} not 0/1")

    return errors


# ---------------------------------------------------------------------------
# Single inference call
# ---------------------------------------------------------------------------
def run_inference(model: str, prompt_text: str) -> tuple[str, float]:
    """Run the model and return (raw_response, elapsed_seconds)."""
    extraction_prompt = EXTRACTION_TEMPLATE.format(
        prompt=prompt_text[:400]  # cap at 400 chars to keep it fast
    )
    t0 = time.perf_counter()
    response = ollama.generate(
        model=model,
        prompt=extraction_prompt,
        options={
            "temperature": 0.05,   # near-deterministic
            "num_predict": 150,    # JSON should fit in ~80 tokens
            "top_p": 0.9,
            "stop": ["\n\n", "```"],
        },
    )
    elapsed = time.perf_counter() - t0
    return response["response"], elapsed


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_eval(model: str = "smollm2:360m"):
    print(f"\n{'='*65}")
    print(f"  SmolLM Feature Extractor Benchmark — model: {model}")
    print(f"{'='*65}\n")

    if not OLLAMA_AVAILABLE:
        print("[ERROR] `ollama` Python package not installed. Run: pip install ollama")
        return

    # Verify model is available
    try:
        ollama.show(model)
    except Exception as e:
        print(f"[ERROR] Model '{model}' not found in Ollama: {e}")
        return

    results = []
    total_parse_ok = 0
    total_valid = 0
    total_time = 0.0

    for i, test in enumerate(TEST_PROMPTS, 1):
        pid = test["id"]
        print(f"[{i}/{len(TEST_PROMPTS)}] {pid}")
        print(f"  Prompt (first 80 chars): {test['prompt'][:80].replace(chr(10),' ')!r}")

        raw, elapsed = run_inference(model, test["prompt"])
        total_time += elapsed

        feat = extract_json(raw)
        parse_ok = feat is not None

        if parse_ok:
            total_parse_ok += 1
            errors = validate_features(feat)
            is_valid = len(errors) == 0
            if is_valid:
                total_valid += 1

            domain_match = feat.get("domain") == test["expected_domain"]
            tt_match = feat.get("task_type") == test["expected_task_type"]
            rd_ok = isinstance(feat.get("reasoning_depth"), int) and feat.get("reasoning_depth", 0) >= test["expected_reasoning_min"]

            print(f"  Time       : {elapsed:.2f}s")
            print(f"  JSON parsed: OK")
            print(f"  Valid fields: {'YES' if is_valid else 'NO — ' + '; '.join(errors)}")
            print(f"  domain     : {feat.get('domain')!r:12} (expected {test['expected_domain']!r}) {'OK' if domain_match else 'MISMATCH'}")
            print(f"  task_type  : {feat.get('task_type')!r:15} (expected {test['expected_task_type']!r}) {'OK' if tt_match else 'MISMATCH'}")
            print(f"  reasoning  : {feat.get('reasoning_depth')}  (min expected {test['expected_reasoning_min']}) {'OK' if rd_ok else 'LOW'}")
            print(f"  ambiguity  : {feat.get('ambiguity_score')}")
            print(f"  factual    : {feat.get('requires_factual_recall')}")
            print(f"  ctx_dep    : {feat.get('context_dependency')}")
        else:
            print(f"  Time       : {elapsed:.2f}s")
            print(f"  JSON parsed: FAILED")
            print(f"  Raw output : {raw[:200]!r}")

        results.append({
            "id": pid,
            "parse_ok": parse_ok,
            "valid": total_valid,
            "elapsed": elapsed,
            "features": feat,
        })
        print()

    # Consistency test: run prompt 1 twice more
    print(f"[consistency] Running math_word_problem 2 more times...")
    consistency_domains = []
    consistency_task_types = []
    for _ in range(2):
        raw, _ = run_inference(model, TEST_PROMPTS[0]["prompt"])
        feat = extract_json(raw)
        if feat:
            consistency_domains.append(feat.get("domain"))
            consistency_task_types.append(feat.get("task_type"))
    print(f"  domain outputs  : {consistency_domains}")
    print(f"  task_type outputs: {consistency_task_types}")
    consistent = len(set(consistency_domains)) == 1 and len(set(consistency_task_types)) == 1
    print(f"  Consistent      : {'YES' if consistent else 'NO — varies across runs'}\n")

    # Summary
    n = len(TEST_PROMPTS)
    avg_time = total_time / n
    print(f"{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    print(f"  Model             : {model}")
    print(f"  Prompts tested    : {n}")
    print(f"  JSON parse rate   : {total_parse_ok}/{n} ({100*total_parse_ok/n:.0f}%)")
    print(f"  Field validity    : {total_valid}/{n} ({100*total_valid/n:.0f}%)")
    print(f"  Avg inference time: {avg_time:.2f}s/prompt")
    print(f"  Est. total (779)  : {avg_time * 779 / 60:.1f} minutes")
    print(f"  Consistent output : {'YES' if consistent else 'NO'}")
    print()
    print("  VERDICT:")
    if total_parse_ok >= 6 and total_valid >= 5 and avg_time < 30:
        print("  [GO] SmolLM2:360m is capable enough. Proceed with LLM feature extraction.")
        print("       Estimated runtime for 779 prompts is acceptable.")
    elif total_parse_ok >= 4:
        print("  [CONDITIONAL GO] Parse rate is moderate. Will need robust fallback logic.")
        print(f"       Consider reducing prompt complexity or truncating harder prompts.")
    else:
        print("  [NO-GO] SmolLM2:360m cannot reliably produce structured JSON.")
        print("       Consider: (1) rule-based fallback only, or (2) use transformers API directly.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    run_eval("smollm2:360m")
