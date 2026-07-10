"""
calibration/extract_calibration_set.py
One-time script: extract 100 stratified prompts from dataset_sweep.csv
and save as calibration/calibration_prompts.jsonl.
Run once before distributing the tool.
"""

import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "data_builder" / "dataset_sweep.csv"
OUT  = ROOT / "calibration" / "calibration_prompts.jsonl"

# Only use deterministic sources (no Fireworks judge needed for evaluation)
SOURCES = {
    "mmlu":      {"n": 25, "evaluator": "mcq"},
    "arc":       {"n": 25, "evaluator": "mcq"},
    "gsm8k":     {"n": 25, "evaluator": "math"},
    "humaneval": {"n": 15, "evaluator": "code"},
    "truthfulqa":{"n": 10, "evaluator": "mcq_keyword"},
}

def main():
    df = pd.read_csv(SRC)
    OUT.parent.mkdir(exist_ok=True)

    rows = []
    for source, cfg in SOURCES.items():
        subset = df[df["source"] == source].sample(
            n=min(cfg["n"], len(df[df["source"] == source])),
            random_state=42
        )
        for _, row in subset.iterrows():
            rows.append({
                "prompt_id":      row["prompt_id"],
                "source":         source,
                "prompt":         row["prompt"],
                "reference":      row["reference_answer"],
                "evaluator":      cfg["evaluator"],
            })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"[OK] Saved {len(rows)} calibration prompts -> {OUT}")
    for s, cfg in SOURCES.items():
        n = sum(1 for r in rows if r["source"] == s)
        print(f"  {s:<12}: {n} prompts")

if __name__ == "__main__":
    main()
