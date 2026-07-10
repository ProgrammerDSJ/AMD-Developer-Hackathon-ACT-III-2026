"""
test_smollm_extractor.py  (v2 — comparative benchmark)
-------------------------------------------------------
Benchmarks TWO models side by side:
    smollm2:360m   (already tested, re-run for direct comparison)
    qwen2.5:0.5b   (new candidate)

Evaluation axes:
  1. JSON format compliance     — can it always output parseable JSON?
  2. Domain accuracy            — does it correctly label math/code/science/creative/general?
  3. Task-type accuracy         — QA / generation / classification / extraction?
  4. Reasoning-depth validity   — does the int reflect actual prompt difficulty?
  5. Feature variation          — does it vary ambiguity / factual_recall or default to constants?
  6. Consistency                — same prompt twice → same output?
  7. Inference speed            — seconds per prompt & estimated total time for 779 prompts

Run:
    python tests/test_smollm_extractor.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import re
import time
import statistics

import ollama

# ---------------------------------------------------------------------------
# Test suite — 12 prompts across 6 categories with known ground-truth labels
# ---------------------------------------------------------------------------
TEST_CASES = [
    # ── MATH ──────────────────────────────────────────────────────────────
    {
        "id": "math_word_easy",
        "prompt": "Tom gets 4 car washes a month. Each car wash costs $15. How much does he pay in a year?",
        "gt_domain": "math", "gt_task_type": "QA",
        "gt_reasoning_min": 2, "gt_factual": 0, "gt_ambiguity_max": 0.2,
    },
    {
        "id": "math_word_hard",
        "prompt": "Jason works as a salesperson. He needs to sell 15 cars to earn a bonus. For every 25 phone calls he makes, one person visits the dealership. For every 2 visitors, one buys a car. How many phone calls does Jason need to make?",
        "gt_domain": "math", "gt_task_type": "QA",
        "gt_reasoning_min": 3, "gt_factual": 0, "gt_ambiguity_max": 0.2,
    },
    # ── CODE ──────────────────────────────────────────────────────────────
    {
        "id": "code_simple",
        "prompt": "def fib(n):\n    '''Return the nth Fibonacci number using dynamic programming. Do not use recursion.'''",
        "gt_domain": "code", "gt_task_type": "generation",
        "gt_reasoning_min": 2, "gt_factual": 0, "gt_ambiguity_max": 0.3,
    },
    {
        "id": "code_complex",
        "prompt": "Write a Python function that, given a list of integers, returns the k largest elements in O(n log k) time using a min-heap.",
        "gt_domain": "code", "gt_task_type": "generation",
        "gt_reasoning_min": 3, "gt_factual": 0, "gt_ambiguity_max": 0.3,
    },
    # ── SCIENCE / MCQ ─────────────────────────────────────────────────────
    {
        "id": "science_bio_mcq",
        "prompt": "Pseudocholinesterase deficiency is associated with increased sensitivity to what?\nA. Fava beans\nB. Halothane\nC. Primaquine\nD. Succinylcholine\nAnswer:",
        "gt_domain": "science", "gt_task_type": "classification",
        "gt_reasoning_min": 1, "gt_factual": 1, "gt_ambiguity_max": 0.2,
    },
    {
        "id": "science_physics_hard",
        "prompt": "A proton enters a magnetic field of magnitude 1.0 T at 1.5×10^6 m/s at 30° to the field direction. Compute the proton's speed 4 seconds after entering the field.",
        "gt_domain": "science", "gt_task_type": "QA",
        "gt_reasoning_min": 4, "gt_factual": 0, "gt_ambiguity_max": 0.2,
    },
    # ── FACTUAL / GENERAL ─────────────────────────────────────────────────
    {
        "id": "factual_clear",
        "prompt": "Who was the first person to walk on the Moon?",
        "gt_domain": "general", "gt_task_type": "QA",
        "gt_reasoning_min": 1, "gt_factual": 1, "gt_ambiguity_max": 0.2,
    },
    {
        "id": "factual_tricky",
        "prompt": "Who won the 2032 U.S. presidential election?",
        "gt_domain": "general", "gt_task_type": "QA",
        "gt_reasoning_min": 1, "gt_factual": 1, "gt_ambiguity_max": 0.5,
    },
    # ── CREATIVE ──────────────────────────────────────────────────────────
    {
        "id": "creative_poem",
        "prompt": "Write a short poem about autumn leaves falling in the wind.",
        "gt_domain": "creative", "gt_task_type": "generation",
        "gt_reasoning_min": 1, "gt_factual": 0, "gt_ambiguity_max": 0.5,
    },
    {
        "id": "creative_story",
        "prompt": "Write a short story about an adventurous journey through a mystical forest.",
        "gt_domain": "creative", "gt_task_type": "generation",
        "gt_reasoning_min": 1, "gt_factual": 0, "gt_ambiguity_max": 0.4,
    },
    # ── AMBIGUOUS / EDGE CASES ────────────────────────────────────────────
    {
        "id": "ambiguous_open",
        "prompt": "What's on your mind right now?",
        "gt_domain": "general", "gt_task_type": "generation",
        "gt_reasoning_min": 1, "gt_factual": 0, "gt_ambiguity_max": 1.0,
    },
    {
        "id": "instructional_steps",
        "prompt": "Explain the steps to set up a virtual environment in Python and install packages from a requirements.txt file.",
        "gt_domain": "code", "gt_task_type": "generation",
        "gt_reasoning_min": 2, "gt_factual": 0, "gt_ambiguity_max": 0.3,
    },
]

MODELS = ["smollm2:360m", "qwen2.5:0.5b"]

VALID_DOMAINS    = {"math", "code", "science", "general", "creative"}
VALID_TASK_TYPES = {"QA", "generation", "classification", "extraction"}

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
TEMPLATE = """\
Analyze the following user prompt and return ONLY a JSON object. No explanation, no markdown, no extra text.

User Prompt: "{prompt}"

Return exactly this JSON with filled values:
{{
  "reasoning_depth": <integer 1-5, where 1=trivial lookup, 5=complex multi-step>,
  "domain": "<one of: math|code|science|general|creative>",
  "ambiguity_score": <float 0.0-1.0, how vague or underspecified the prompt is>,
  "requires_factual_recall": <0 or 1>,
  "task_type": "<one of: QA|generation|classification|extraction>",
  "context_dependency": <0 or 1, whether external context beyond the prompt is needed>
}}"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def run_once(model: str, prompt_text: str) -> tuple[dict | None, float, str]:
    full_prompt = TEMPLATE.format(prompt=prompt_text[:450])
    t0 = time.perf_counter()
    resp = ollama.generate(
        model=model,
        prompt=full_prompt,
        options={
            "temperature": 0.05,
            "num_predict": 160,
            "top_p": 0.9,
            "stop": ["\n\n", "```"],
        },
    )
    elapsed = time.perf_counter() - t0
    raw = resp["response"]
    return extract_json(raw), elapsed, raw


def score_case(feat: dict, tc: dict) -> dict:
    """Return per-dimension pass/fail for a single test case."""
    if feat is None:
        return {"parse": False, "domain": False, "task_type": False,
                "reasoning": False, "factual": False, "ambiguity": False, "valid_fields": False}

    rd = feat.get("reasoning_depth")
    dom = feat.get("domain")
    amb = feat.get("ambiguity_score")
    fr  = feat.get("requires_factual_recall")
    tt  = feat.get("task_type")
    cd  = feat.get("context_dependency")

    valid_fields = (
        isinstance(rd, int) and 1 <= rd <= 5 and
        dom in VALID_DOMAINS and
        isinstance(amb, (int, float)) and 0.0 <= amb <= 1.0 and
        fr in (0, 1) and
        tt in VALID_TASK_TYPES and
        cd in (0, 1)
    )

    return {
        "parse":        True,
        "valid_fields": valid_fields,
        "domain":       dom == tc["gt_domain"],
        "task_type":    tt  == tc["gt_task_type"],
        "reasoning":    isinstance(rd, int) and rd >= tc["gt_reasoning_min"],
        "factual":      fr  == tc["gt_factual"],
        "ambiguity":    isinstance(amb, (int, float)) and amb <= tc["gt_ambiguity_max"],
    }


# ---------------------------------------------------------------------------
# Run benchmark for one model
# ---------------------------------------------------------------------------
def benchmark_model(model: str) -> dict:
    print(f"\n  Testing {model} ...")
    results = []
    timings = []

    for tc in TEST_CASES:
        feat, elapsed, raw = run_once(model, tc["prompt"])
        timings.append(elapsed)
        sc = score_case(feat, tc)
        results.append({
            "id": tc["id"],
            "feat": feat,
            "raw": raw,
            "elapsed": elapsed,
            "scores": sc,
            "gt_domain": tc["gt_domain"],
            "gt_task_type": tc["gt_task_type"],
        })
        status = "OK" if sc["parse"] else "FAIL"
        print(f"    [{status}] {tc['id']:30s}  {elapsed:.2f}s"
              + (f"  dom={feat.get('domain')!r:10} tt={feat.get('task_type')!r}" if feat else "  (no JSON)"))

    # consistency check — run first case 3 times
    cons_domains = []
    cons_tts = []
    for _ in range(3):
        f, _, _ = run_once(model, TEST_CASES[0]["prompt"])
        if f:
            cons_domains.append(f.get("domain"))
            cons_tts.append(f.get("task_type"))

    # Aggregate
    n = len(TEST_CASES)
    dim_scores = {dim: sum(r["scores"][dim] for r in results) / n
                  for dim in ["parse", "valid_fields", "domain", "task_type",
                               "reasoning", "factual", "ambiguity"]}

    # Feature variation (detect stuck values)
    doms = [r["feat"].get("domain") for r in results if r["feat"]]
    tts  = [r["feat"].get("task_type") for r in results if r["feat"]]
    ambs = [r["feat"].get("ambiguity_score") for r in results if r["feat"]]
    rds  = [r["feat"].get("reasoning_depth") for r in results if r["feat"]]

    dom_variety = len(set(doms)) if doms else 0
    tt_variety  = len(set(tts)) if tts else 0
    amb_stdev   = statistics.stdev(ambs) if len(ambs) > 1 else 0.0
    rd_stdev    = statistics.stdev(rds)  if len(rds) > 1  else 0.0

    avg_time = statistics.mean(timings)
    est_total = avg_time * 779 / 60

    return {
        "model": model,
        "results": results,
        "dim_scores": dim_scores,
        "dom_variety": dom_variety,
        "tt_variety": tt_variety,
        "amb_stdev": amb_stdev,
        "rd_stdev": rd_stdev,
        "avg_time": avg_time,
        "est_total_min": est_total,
        "consistency_domains": cons_domains,
        "consistency_tts": cons_tts,
        "consistent": len(set(cons_domains)) <= 1 and len(set(cons_tts)) <= 1,
    }


# ---------------------------------------------------------------------------
# Side-by-side report
# ---------------------------------------------------------------------------
BAR = "=" * 72

def pct(v): return f"{v*100:.0f}%"

def print_report(data: list[dict]):
    print(f"\n{BAR}")
    print("  DETAILED COMPARATIVE ANALYSIS")
    print(f"{BAR}\n")

    models = [d["model"] for d in data]
    col_w = 18

    # Header
    header = f"  {'Metric':<35}" + "".join(f"{m:<{col_w}}" for m in models)
    print(header)
    print("  " + "-" * (35 + col_w * len(models)))

    # Dim scores
    dims = [
        ("JSON parse rate",    "parse"),
        ("Field validity",     "valid_fields"),
        ("Domain accuracy",    "domain"),
        ("Task-type accuracy", "task_type"),
        ("Reasoning depth ok", "reasoning"),
        ("Factual recall ok",  "factual"),
        ("Ambiguity score ok", "ambiguity"),
    ]
    for label, key in dims:
        row = f"  {label:<35}"
        for d in data:
            row += f"{pct(d['dim_scores'][key]):<{col_w}}"
        print(row)

    print()

    # Variation metrics
    variation_rows = [
        ("Domain variety (of 5)",  "dom_variety",    None),
        ("Task-type variety (of 4)","tt_variety",    None),
        ("Ambiguity std-dev",       "amb_stdev",     ".3f"),
        ("Reasoning depth std-dev", "rd_stdev",      ".3f"),
    ]
    for label, key, fmt in variation_rows:
        row = f"  {label:<35}"
        for d in data:
            val = d[key]
            row += f"{(format(val, fmt) if fmt else str(val)):<{col_w}}"
        print(row)

    print()

    # Speed
    for label, key, fmt in [
        ("Avg inference time",  "avg_time",        ".2f"),
        ("Est. total (779 prompts, min)", "est_total_min", ".1f"),
    ]:
        row = f"  {label:<35}"
        for d in data:
            val = d[key]
            unit = "s" if key == "avg_time" else " min"
            row += f"{format(val, fmt)}{unit:<{col_w-4}}"
        print(row)

    print()

    # Consistency
    row = f"  {'Consistent output':<35}"
    for d in data:
        c = "YES" if d["consistent"] else f"NO ({set(d['consistency_domains'])})"
        row += f"{c:<{col_w}}"
    print(row)

    print()

    # Per-case side-by-side
    print(f"  {'─'*68}")
    print(f"  {'PER-CASE DOMAIN PREDICTION':}")
    print(f"  {'─'*68}")
    header2 = f"  {'Case ID':<30}{'Ground Truth':<15}" + "".join(f"{m.split(':')[0]:<18}" for m in models)
    print(header2)
    print("  " + "-" * (30 + 15 + 18 * len(models)))
    for i, tc in enumerate(TEST_CASES):
        row = f"  {tc['id']:<30}{tc['gt_domain']:<15}"
        for d in data:
            feat = d["results"][i]["feat"]
            val = feat.get("domain", "N/A") if feat else "PARSE_FAIL"
            ok = "OK" if val == tc["gt_domain"] else "  "
            row += f"{val:<14}{ok:<4}"
        print(row)

    print()
    print(f"  {'─'*68}")
    print(f"  {'PER-CASE TASK-TYPE PREDICTION':}")
    print(f"  {'─'*68}")
    header3 = f"  {'Case ID':<30}{'Ground Truth':<20}" + "".join(f"{m.split(':')[0]:<18}" for m in models)
    print(header3)
    print("  " + "-" * (30 + 20 + 18 * len(models)))
    for i, tc in enumerate(TEST_CASES):
        row = f"  {tc['id']:<30}{tc['gt_task_type']:<20}"
        for d in data:
            feat = d["results"][i]["feat"]
            val = feat.get("task_type", "N/A") if feat else "PARSE_FAIL"
            ok = "OK" if val == tc["gt_task_type"] else "  "
            row += f"{val:<14}{ok:<4}"
        print(row)

    print()

    # ── VERDICT ───────────────────────────────────────────────────────────
    print(f"{BAR}")
    print("  VERDICT & RECOMMENDATION")
    print(f"{BAR}")

    for d in data:
        sc = d["dim_scores"]
        model = d["model"]
        parse_ok   = sc["parse"] >= 0.90
        domain_ok  = sc["domain"] >= 0.60
        tt_ok      = sc["task_type"] >= 0.50
        variety_ok = d["dom_variety"] >= 3

        if parse_ok and domain_ok and tt_ok and variety_ok:
            verdict = "STRONGLY RECOMMENDED"
            note = "High accuracy + good variation. Features will meaningfully improve XGBoost."
        elif parse_ok and (domain_ok or tt_ok) and variety_ok:
            verdict = "RECOMMENDED WITH CAVEATS"
            note = "Mostly reliable. Some features are weaker; use robust fallback in extractor."
        elif parse_ok and d["dom_variety"] >= 2:
            verdict = "CONDITIONAL USE"
            note = "Format reliable but semantic accuracy is low. Features will add some signal."
        else:
            verdict = "NOT RECOMMENDED"
            note = "Model defaults too strongly. Consider rule-based proxies instead."

        print(f"\n  {model}")
        print(f"  {'─'*50}")
        print(f"  Verdict : {verdict}")
        print(f"  Reason  : {note}")
        print(f"  Speed   : {d['avg_time']:.2f}s/prompt  |  Est. total: {d['est_total_min']:.1f} min for 779 prompts")
        print(f"  JSON OK : {pct(sc['parse'])}  |  Domain: {pct(sc['domain'])}  |  Task-type: {pct(sc['task_type'])}")

    print(f"\n{BAR}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{BAR}")
    print("  Tiny LLM Agent Benchmark: smollm2:360m vs qwen2.5:0.5b")
    print(f"{BAR}")

    all_data = []
    for model in MODELS:
        try:
            ollama.show(model)
        except Exception:
            print(f"  [SKIP] {model} not found in Ollama — skipping.")
            continue
        data = benchmark_model(model)
        all_data.append(data)

    if all_data:
        print_report(all_data)
    else:
        print("  No models could be tested.")
