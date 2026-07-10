"""
process_dataset.py
------------------
Reads prompts.jsonl, runs the hardcoded feature extractor on every prompt,
and writes a fully-structured dataset.csv ready for:
    - Badal to fill in the LLM-based feature columns
    - The benchmark sweep to fill in response + label columns

Output CSV column layout
------------------------
IDENTITY
    prompt_id, source, task_type, domain, difficulty,
    prompt, reference_answer

HARDCODED FEATURES  (filled by this script)
    source_task_type_encoded,                        <- NEW: from dataset metadata
    prompt_length, has_code_block, has_math_symbols,
    question_type, question_type_encoded,
    num_sentences, avg_word_length, complexity_heuristic

LLM-BASED FEATURES  (filled by Badal's tiny agent — blank for now)
    llm_reasoning_depth, llm_domain,
    llm_ambiguity_score, llm_requires_factual_recall,
    llm_task_type, llm_context_dependency

BENCHMARK SWEEP RESULTS  (filled during sweep — blank for now)
    local_response,  local_correct,
    tier1_response,  tier1_tokens,  tier1_correct,
    tier2_response,  tier2_tokens,  tier2_correct,
    tier3_response,  tier3_tokens,  tier3_correct

LABEL  (assigned after sweep — blank for now)
    label, label_encoded

Usage:
    python data_builder/process_dataset.py
    python data_builder/process_dataset.py --input  data_builder/prompt_collection/prompts.jsonl
                                           --output data_builder/dataset.csv
"""

import json
import csv
import argparse
import sys
from pathlib import Path

# Make sure the repo root is on the path so we can import feature_extractor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extractor.hardcoded_features import extract_hardcoded_features

# ---------------------------------------------------------------------------
# Column definitions — single source of truth for the CSV schema
# ---------------------------------------------------------------------------

IDENTITY_COLS = [
    "prompt_id",
    "source",
    "task_type",
    "domain",
    "difficulty",
    "prompt",
    "reference_answer",
]

# Encoding of the ground-truth task_type from the source benchmark.
# This is more reliable than the keyword-based question_type classifier
# because it comes from verified dataset metadata, not prompt text analysis.
SOURCE_TASK_TYPE_ENCODING = {
    "math":    0,
    "code":    1,
    "science": 2,
    "factual": 3,
    "general": 4,
}

HARDCODED_FEATURE_COLS = [
    # Ground-truth task type from dataset metadata (always accurate)
    "source_task_type_encoded",
    # Text-derived features
    "prompt_length",
    "has_code_block",
    "has_math_symbols",
    "question_type",
    "question_type_encoded",
    "num_sentences",
    "avg_word_length",
    "complexity_heuristic",
]

# LLM-based features — populated by Badal's sub-0.6B agent
LLM_FEATURE_COLS = [
    "llm_reasoning_depth",        # int 1-5: reasoning steps required
    "llm_domain",                 # str: science/code/math/general/legal/creative
    "llm_ambiguity_score",        # float 0-1: how underspecified the prompt is
    "llm_requires_factual_recall",# bool 0/1: needs specific memorised facts
    "llm_task_type",              # str: generation/classification/extraction/QA
    "llm_context_dependency",     # bool 0/1: requires external context
]

# Benchmark sweep results — populated during the sweep phase
SWEEP_COLS = [
    "local_response",
    "local_correct",
    "tier1_response",
    "tier1_tokens",
    "tier1_correct",
    "tier2_response",
    "tier2_tokens",
    "tier2_correct",
    "tier3_response",
    "tier3_tokens",
    "tier3_correct",
]

# Final routing label — assigned after sweep
LABEL_COLS = [
    "label",          # str: local / tier1 / tier2 / tier3
    "label_encoded",  # int: 0 / 1 / 2 / 3
]

ALL_COLS = (
    IDENTITY_COLS
    + HARDCODED_FEATURE_COLS
    + LLM_FEATURE_COLS
    + SWEEP_COLS
    + LABEL_COLS
)


# ---------------------------------------------------------------------------
# Processing logic
# ---------------------------------------------------------------------------

def process_prompt(raw: dict) -> dict:
    """
    Given one raw record from prompts.jsonl, return a full CSV row dict.
    Hardcoded features are populated; everything else is left blank ("").
    """
    prompt_text = raw.get("prompt", "")

    # --- Identity fields ------------------------------------------------
    row = {
        "prompt_id":        raw.get("prompt_id", ""),
        "source":           raw.get("source", ""),
        "task_type":        raw.get("task_type", ""),
        "domain":           raw.get("domain", ""),
        "difficulty":       raw.get("difficulty", ""),
        "prompt":           prompt_text,
        "reference_answer": raw.get("reference_answer", ""),
    }

    # --- Hardcoded features ---------------------------------------------
    feats = extract_hardcoded_features(prompt_text)

    # source_task_type_encoded: integer encoding of the benchmark's own
    # task_type label — more accurate than keyword-based question_type.
    source_task = raw.get("task_type", "general").lower()
    feats["source_task_type_encoded"] = SOURCE_TASK_TYPE_ENCODING.get(
        source_task, 4   # default to 'general' if unknown
    )
    row.update(feats)

    # --- LLM features (blank — Badal fills these) -----------------------
    for col in LLM_FEATURE_COLS:
        row[col] = ""

    # --- Benchmark sweep results (blank — filled during sweep) ----------
    for col in SWEEP_COLS:
        row[col] = ""

    # --- Label (blank — assigned after sweep) ---------------------------
    for col in LABEL_COLS:
        row[col] = ""

    return row


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    from collections import Counter

    qt_counts   = Counter(r["question_type"] for r in rows)
    src_counts  = Counter(r["source"]        for r in rows)
    diff_counts = Counter(r["difficulty"]    for r in rows)

    avg_len  = sum(r["prompt_length"]        for r in rows) / len(rows)
    avg_comp = sum(r["complexity_heuristic"] for r in rows) / len(rows)
    code_pct = 100 * sum(r["has_code_block"]   for r in rows) / len(rows)
    math_pct = 100 * sum(r["has_math_symbols"] for r in rows) / len(rows)

    print("\n--- Dataset Summary ---")
    print(f"Total rows       : {len(rows)}")
    print(f"Avg prompt length: {avg_len:.1f} words")
    print(f"Avg complexity   : {avg_comp:.3f}")
    print(f"Has code block   : {code_pct:.1f}%")
    print(f"Has math symbols : {math_pct:.1f}%")

    print("\nQuestion type distribution:")
    for k, v in qt_counts.most_common():
        print(f"  {k:<14} {v:>4}  ({100*v/len(rows):.1f}%)")

    print("\nBy source:")
    for k, v in src_counts.most_common():
        print(f"  {k:<14} {v:>4}")

    print("\nBy difficulty:")
    for k, v in diff_counts.most_common():
        print(f"  {k:<14} {v:>4}")
    print("-----------------------\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Process prompts.jsonl into a structured dataset.csv."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data_builder/prompt_collection/prompts.jsonl",
        help="Path to prompts.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_builder/dataset.csv",
        help="Output path for dataset.csv",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    print(f"[*] Loading prompts from: {input_path}")
    raw_records = load_jsonl(input_path)
    print(f"    Loaded {len(raw_records)} prompts.\n")

    print("[*] Extracting hardcoded features...")
    rows = []
    for i, raw in enumerate(raw_records, 1):
        row = process_prompt(raw)
        rows.append(row)
        if i % 100 == 0:
            print(f"    Processed {i}/{len(raw_records)} prompts...")

    print(f"    Done. {len(rows)} rows processed.\n")

    print_summary(rows)

    save_csv(rows, output_path)
    print(f"[OK] Saved dataset -> {output_path}")
    print(f"     Columns: {len(ALL_COLS)} total")
    print(f"       - {len(HARDCODED_FEATURE_COLS)} hardcoded features  (filled)")
    print(f"       - {len(LLM_FEATURE_COLS)} LLM features         (blank - Badal fills)")
    print(f"       - {len(SWEEP_COLS)} sweep result cols    (blank - filled during sweep)")
    print(f"       - {len(LABEL_COLS)} label columns        (blank - filled after sweep)")


if __name__ == "__main__":
    main()
