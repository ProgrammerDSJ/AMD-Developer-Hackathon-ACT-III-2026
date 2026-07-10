"""
data_builder/fill_llm_features.py
-----------------------------------
Reads the existing dataset.csv, fills in the 3 SmolLM-based LLM feature
columns using SmolLM2:360m via Ollama, and saves the result back.

This script is designed to be run AFTER process_dataset.py has already
created the base dataset.csv with hardcoded + rule-based features.

SmolLM fills:
  llm_reasoning_depth      int 1-5
  llm_ambiguity_score      float 0-1
  llm_context_dependency   int 0/1

Already filled (by process_dataset.py, not touched here):
  llm_requires_factual_recall
  llm_task_type

Robustness features:
  - Progress checkpoint every CHECKPOINT_EVERY rows
  - Resumes from last checkpoint if interrupted
  - Fallback values if Ollama fails
  - Skips rows already filled (re-run safe)

Usage:
    python data_builder/fill_llm_features.py
    python data_builder/fill_llm_features.py --input data_builder/dataset.csv
                                              --model smollm2:360m
                                              --force   # re-fill even if already done
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extractor.llm_features import extract_smollm_features

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SMOLLM_COLS     = ["llm_reasoning_depth", "llm_ambiguity_score", "llm_context_dependency"]
CHECKPOINT_EVERY = 50    # save partial results every N rows


# ---------------------------------------------------------------------------
# CSV I/O (manual, to preserve multiline quoted fields exactly)
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> tuple[list[str], list[dict]]:
    """Load CSV; returns (fieldnames, rows_as_dicts)."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def save_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    """Write rows back to CSV, preserving all columns."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Row-level feature fill
# ---------------------------------------------------------------------------

def row_needs_filling(row: dict) -> bool:
    """Return True if any of the 3 SmolLM columns are empty."""
    return any(str(row.get(col, "")).strip() == "" for col in SMOLLM_COLS)


def fill_row(row: dict, model: str) -> dict:
    """
    Extract SmolLM features for one row and update it in place.
    Returns the modified row.
    """
    prompt_text   = row.get("prompt", "")
    try:
        complexity = float(row.get("complexity_heuristic", 0.2))
    except (ValueError, TypeError):
        complexity = 0.2

    feats = extract_smollm_features(
        prompt_text=prompt_text,
        complexity_heuristic=complexity,
        model=model,
    )

    row["llm_reasoning_depth"]    = feats["llm_reasoning_depth"]
    row["llm_ambiguity_score"]    = feats["llm_ambiguity_score"]
    row["llm_context_dependency"] = feats["llm_context_dependency"]
    return row


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_llm_summary(rows: list[dict]) -> None:
    from collections import Counter

    filled = [r for r in rows if str(r.get("llm_reasoning_depth", "")).strip() != ""]
    if not filled:
        print("  No rows with LLM features yet.")
        return

    rd_vals  = [int(r["llm_reasoning_depth"])    for r in filled]
    amb_vals = [float(r["llm_ambiguity_score"])  for r in filled]
    cd_vals  = [int(r["llm_context_dependency"]) for r in filled]
    tt_vals  = [r.get("llm_task_type", "")       for r in rows]
    fr_vals  = [str(r.get("llm_requires_factual_recall","")) for r in rows]

    print("\n--- LLM Feature Summary ---")
    print(f"Rows with SmolLM features filled : {len(filled)}/{len(rows)}")
    print(f"\nllm_reasoning_depth distribution:")
    for k, v in sorted(Counter(rd_vals).items()):
        print(f"  depth={k}  {v:>4} rows  ({100*v/len(filled):.1f}%)")
    print(f"  mean = {sum(rd_vals)/len(rd_vals):.2f}")

    print(f"\nllm_ambiguity_score:")
    print(f"  mean = {sum(amb_vals)/len(amb_vals):.3f}")
    print(f"  min  = {min(amb_vals):.3f}  max = {max(amb_vals):.3f}")

    print(f"\nllm_context_dependency:")
    cd_c = Counter(cd_vals)
    print(f"  dep=0  {cd_c[0]:>4} rows   dep=1  {cd_c[1]:>4} rows")

    print(f"\nllm_task_type (rule-based):")
    for k, v in Counter(tt_vals).most_common():
        if k:
            print(f"  {k:<20} {v:>4} rows")

    print(f"\nllm_requires_factual_recall (rule-based):")
    fr_c = Counter(fr_vals)
    print(f"  0 = {fr_c.get('0',0):>4} rows   1 = {fr_c.get('1',0):>4} rows")
    print("---------------------------\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fill SmolLM-based LLM feature columns in dataset.csv."
    )
    parser.add_argument("--input",  default="data_builder/dataset.csv")
    parser.add_argument("--model",  default="smollm2:360m")
    parser.add_argument("--force",  action="store_true",
                        help="Re-fill even rows already populated")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        print(f"[ERROR] dataset.csv not found: {csv_path}")
        sys.exit(1)

    # Verify Ollama is reachable
    try:
        import ollama
        ollama.show(args.model)
        print(f"[OK] Ollama running. Model: {args.model}")
    except Exception as e:
        print(f"[ERROR] Cannot reach Ollama or model '{args.model}': {e}")
        print("       Make sure Ollama is running: ollama serve")
        sys.exit(1)

    print(f"[*] Loading: {csv_path}")
    fieldnames, rows = load_csv(csv_path)
    print(f"    {len(rows)} rows loaded, {len(fieldnames)} columns.\n")

    # Validate schema
    for col in SMOLLM_COLS:
        if col not in fieldnames:
            print(f"[ERROR] Column '{col}' not in CSV. Run process_dataset.py first.")
            sys.exit(1)

    # Determine rows to fill
    if args.force:
        to_fill = list(range(len(rows)))
        print(f"[!] --force: re-filling all {len(rows)} rows.")
    else:
        to_fill = [i for i, r in enumerate(rows) if row_needs_filling(r)]
        already  = len(rows) - len(to_fill)
        if already:
            print(f"[*] {already} rows already filled — skipping. Use --force to override.")
        print(f"[*] Filling {len(to_fill)} rows with SmolLM2:360m...\n")

    if not to_fill:
        print("[OK] Nothing to fill. Dataset is complete.")
        print_llm_summary(rows)
        return

    t_start = time.perf_counter()
    filled_count = 0
    failed_count = 0
    last_checkpoint = 0

    for idx, row_i in enumerate(to_fill, 1):
        row = rows[row_i]
        pid = row.get("prompt_id", f"row_{row_i}")

        try:
            rows[row_i] = fill_row(row, args.model)
            filled_count += 1
        except Exception as e:
            print(f"  [WARN] {pid}: {e} — using fallback")
            failed_count += 1

        # Progress report
        if idx % 10 == 0 or idx == len(to_fill):
            elapsed = time.perf_counter() - t_start
            rate    = idx / elapsed
            eta_sec = (len(to_fill) - idx) / rate if rate > 0 else 0
            print(f"  [{idx:>4}/{len(to_fill)}]  {rate:.1f} rows/s  "
                  f"ETA: {eta_sec/60:.1f} min  "
                  f"filled={filled_count}  failed={failed_count}")

        # Checkpoint save
        if idx - last_checkpoint >= CHECKPOINT_EVERY:
            save_csv(csv_path, fieldnames, rows)
            last_checkpoint = idx
            print(f"  [checkpoint] Saved at row {idx}")

    # Final save
    save_csv(csv_path, fieldnames, rows)
    total_time = time.perf_counter() - t_start

    print(f"\n[OK] Done in {total_time/60:.1f} minutes.")
    print(f"     Filled  : {filled_count} rows")
    print(f"     Failed  : {failed_count} rows (used fallback)")
    print(f"     Saved   : {csv_path}")

    print_llm_summary(rows)


if __name__ == "__main__":
    main()
