"""
collect_prompts.py
------------------
Downloads and normalizes prompts from public benchmark datasets into a
single prompts.jsonl file for the HybridRouter benchmark sweep.

Sources:
    - GSM8K        (math reasoning)
    - HumanEval    (code generation)
    - MMLU         (science / factual MCQ)
    - TruthfulQA   (factual Q&A, hallucination-prone)
    - ARC-Challenge (hard science MCQ)
    - Alpaca       (general instructions)

Usage:
    python data_builder/prompt_collection/collect_prompts.py
    python data_builder/prompt_collection/collect_prompts.py --output custom_path.jsonl
    python data_builder/prompt_collection/collect_prompts.py --sources gsm8k humaneval --max 200
"""

import json
import uuid
import argparse
import random
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional: install datasets on first run if missing
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    from datasets import load_dataset

# ---------------------------------------------------------------------------
# Config — how many prompts to sample from each source
# ---------------------------------------------------------------------------
DEFAULT_SAMPLES = {
    "gsm8k":       150,
    "humaneval":   100,
    "mmlu":        200,
    "truthfulqa":  100,
    "arc":         100,
    "alpaca":      150,
}

TOTAL_TARGET = sum(DEFAULT_SAMPLES.values())   # 800 prompts

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Normalizers — each returns a list of dicts in the standard schema
# ---------------------------------------------------------------------------

def _make_prompt(
    source: str,
    task_type: str,
    prompt_text: str,
    reference_answer: str = "",
    difficulty: str = "medium",
    domain: str = "general",
    extra: dict = None,
) -> dict:
    """Build a standardized prompt record."""
    record = {
        "prompt_id":       f"{source}_{uuid.uuid4().hex[:8]}",
        "source":          source,
        "task_type":       task_type,    # math | code | science | factual | general
        "domain":          domain,
        "difficulty":      difficulty,
        "prompt":          prompt_text.strip(),
        "reference_answer": str(reference_answer).strip(),
    }
    if extra:
        record.update(extra)
    return record


def collect_gsm8k(n: int) -> list[dict]:
    """Grade-school math word problems with step-by-step solutions."""
    print(f"  [gsm8k] Loading {n} math problems...")
    ds = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=RANDOM_SEED).select(range(min(n, len(ds))))

    records = []
    for row in ds:
        # Extract the final numeric answer from the '#### N' convention
        answer_match = re.search(r"####\s*([\d,\-\.]+)", row["answer"])
        final_answer = answer_match.group(1).replace(",", "") if answer_match else ""

        records.append(_make_prompt(
            source="gsm8k",
            task_type="math",
            domain="math",
            difficulty="medium",
            prompt_text=row["question"],
            reference_answer=final_answer,
            extra={"full_solution": row["answer"]},
        ))
    print(f"  [gsm8k] Collected {len(records)} prompts.")
    return records


def collect_humaneval(n: int) -> list[dict]:
    """Python function completion problems with unit tests."""
    print(f"  [humaneval] Loading {n} coding problems...")
    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=RANDOM_SEED).select(range(min(n, len(ds))))

    records = []
    for row in ds:
        records.append(_make_prompt(
            source="humaneval",
            task_type="code",
            domain="code",
            difficulty="medium",
            prompt_text=row["prompt"],
            reference_answer=row["canonical_solution"],
            extra={
                "task_id":   row["task_id"],
                "test_code": row["test"],
                "entry_point": row["entry_point"],
            },
        ))
    print(f"  [humaneval] Collected {len(records)} prompts.")
    return records


def collect_mmlu(n: int) -> list[dict]:
    """Multi-subject MCQ spanning science, history, law, medicine, etc."""
    print(f"  [mmlu] Loading {n} MCQ prompts...")

    # Sample across a diverse spread of subjects
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "biology", "chemistry",
        "clinical_knowledge", "college_physics", "computer_security",
        "electrical_engineering", "elementary_mathematics", "formal_logic",
        "global_facts", "high_school_chemistry", "high_school_mathematics",
        "high_school_physics", "logical_fallacies", "machine_learning",
        "medical_genetics", "philosophy", "world_religions",
    ]

    all_rows = []
    per_subject = max(1, n // len(subjects))

    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test", trust_remote_code=True)
            ds = ds.shuffle(seed=RANDOM_SEED).select(range(min(per_subject, len(ds))))
            for row in ds:
                all_rows.append((subject, row))
        except Exception as e:
            print(f"    [mmlu] Skipping {subject}: {e}")

    random.seed(RANDOM_SEED)
    random.shuffle(all_rows)
    all_rows = all_rows[:n]

    option_labels = ["A", "B", "C", "D"]
    records = []
    for subject, row in all_rows:
        choices_text = "\n".join(
            f"{option_labels[i]}. {row['choices'][i]}"
            for i in range(len(row["choices"]))
        )
        prompt_text = f"{row['question']}\n\n{choices_text}\n\nAnswer:"
        correct_option = option_labels[row["answer"]]

        records.append(_make_prompt(
            source="mmlu",
            task_type="science",
            domain="science",
            difficulty="hard",
            prompt_text=prompt_text,
            reference_answer=correct_option,
            extra={"subject": subject, "choices": row["choices"]},
        ))
    print(f"  [mmlu] Collected {len(records)} prompts.")
    return records


def collect_truthfulqa(n: int) -> list[dict]:
    """Questions designed to surface model hallucinations."""
    print(f"  [truthfulqa] Loading {n} factual prompts...")
    ds = load_dataset("truthful_qa", "generation", split="validation", trust_remote_code=True)
    ds = ds.shuffle(seed=RANDOM_SEED).select(range(min(n, len(ds))))

    records = []
    for row in ds:
        # Use the first correct answer as reference
        best_answer = row["best_answer"] if row["best_answer"] else (
            row["correct_answers"][0] if row["correct_answers"] else ""
        )
        records.append(_make_prompt(
            source="truthfulqa",
            task_type="factual",
            domain="general",
            difficulty="medium",
            prompt_text=row["question"],
            reference_answer=best_answer,
            extra={"category": row.get("category", "")},
        ))
    print(f"  [truthfulqa] Collected {len(records)} prompts.")
    return records


def collect_arc(n: int) -> list[dict]:
    """ARC-Challenge: hard science MCQs that stump simple retrieval models."""
    print(f"  [arc] Loading {n} science MCQ prompts...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
    ds = ds.shuffle(seed=RANDOM_SEED).select(range(min(n, len(ds))))

    records = []
    for row in ds:
        choices = row["choices"]
        choices_text = "\n".join(
            f"{choices['label'][i]}. {choices['text'][i]}"
            for i in range(len(choices["label"]))
        )
        prompt_text = f"{row['question']}\n\n{choices_text}\n\nAnswer:"

        records.append(_make_prompt(
            source="arc",
            task_type="science",
            domain="science",
            difficulty="hard",
            prompt_text=prompt_text,
            reference_answer=row["answerKey"],
        ))
    print(f"  [arc] Collected {len(records)} prompts.")
    return records


def collect_alpaca(n: int) -> list[dict]:
    """General instruction-following prompts (simple to moderate complexity)."""
    print(f"  [alpaca] Loading {n} general instruction prompts...")
    ds = load_dataset("tatsu-lab/alpaca", split="train", trust_remote_code=True)

    # Filter to prompts with no additional input context (cleaner for our use case)
    ds = ds.filter(lambda x: x["input"].strip() == "")
    ds = ds.shuffle(seed=RANDOM_SEED).select(range(min(n, len(ds))))

    records = []
    for row in ds:
        records.append(_make_prompt(
            source="alpaca",
            task_type="general",
            domain="general",
            difficulty="easy",
            prompt_text=row["instruction"],
            reference_answer=row["output"],
        ))
    print(f"  [alpaca] Collected {len(records)} prompts.")
    return records


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------
COLLECTORS = {
    "gsm8k":      collect_gsm8k,
    "humaneval":  collect_humaneval,
    "mmlu":       collect_mmlu,
    "truthfulqa": collect_truthfulqa,
    "arc":        collect_arc,
    "alpaca":     collect_alpaca,
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_all(sources: list[str], sample_config: dict) -> list[dict]:
    all_records = []
    for source in sources:
        if source not in COLLECTORS:
            print(f"  [warn] Unknown source '{source}', skipping.")
            continue
        n = sample_config.get(source, DEFAULT_SAMPLES.get(source, 50))
        records = COLLECTORS[source](n)
        all_records.extend(records)
    return all_records


def deduplicate(records: list[dict]) -> list[dict]:
    """Remove exact duplicate prompts (by stripped lowercase text)."""
    seen = set()
    unique = []
    for r in records:
        key = r["prompt"].strip().lower()[:200]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n[OK] Saved {len(records)} prompts -> {path}")


def print_summary(records: list[dict]) -> None:
    from collections import Counter
    task_counts   = Counter(r["task_type"]  for r in records)
    source_counts = Counter(r["source"]     for r in records)
    diff_counts   = Counter(r["difficulty"] for r in records)

    print("\n--- Dataset Summary ---")
    print(f"Total prompts : {len(records)}")
    print("\nBy task type:")
    for k, v in task_counts.most_common():
        print(f"  {k:<12} {v:>4}  ({100*v/len(records):.1f}%)")
    print("\nBy source:")
    for k, v in source_counts.most_common():
        print(f"  {k:<12} {v:>4}")
    print("\nBy difficulty:")
    for k, v in diff_counts.most_common():
        print(f"  {k:<12} {v:>4}")
    print("-----------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="Collect and normalize prompts from public benchmarks."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_builder/prompt_collection/prompts.jsonl",
        help="Output path for prompts.jsonl",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SAMPLES.keys()),
        choices=list(COLLECTORS.keys()),
        help="Which datasets to pull from",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Override total max prompts (evenly distributed across sources)",
    )
    args = parser.parse_args()

    sample_config = dict(DEFAULT_SAMPLES)
    if args.max:
        per_source = args.max // len(args.sources)
        sample_config = {s: per_source for s in args.sources}

    print(f"\n[*] Collecting prompts from: {', '.join(args.sources)}")
    print(f"    Target total: {sum(sample_config[s] for s in args.sources)} prompts\n")

    records = collect_all(args.sources, sample_config)

    before = len(records)
    records = deduplicate(records)
    removed = before - len(records)
    if removed:
        print(f"  [dedup] Removed {removed} duplicate prompts.")

    # Shuffle final dataset
    random.seed(RANDOM_SEED)
    random.shuffle(records)

    print_summary(records)
    save_jsonl(records, Path(args.output))


if __name__ == "__main__":
    main()
