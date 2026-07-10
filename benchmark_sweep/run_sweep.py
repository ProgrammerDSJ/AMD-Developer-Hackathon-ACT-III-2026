"""
benchmark_sweep/run_sweep.py
-----------------------------
Phase 3 — Benchmark Sweep (Remote-Only)

What this script does:
  - Queries each prompt through 3 remote Fireworks tiers
  - Stores: tier1_response, tier1_tokens, tier2_response, tier2_tokens,
            tier3_response, tier3_tokens
  - Skips rows that already have tier3_response filled (resumable)
  - Checkpoints to dataset_sweep.csv every 10 rows

What this script does NOT do:
  - No local model calls
  - No evaluation / correctness scoring
  - No label assignment
  - No training

Evaluation and labeling are separate phases that run AFTER this sweep completes.

Usage:
  python benchmark_sweep/run_sweep.py
  python benchmark_sweep/run_sweep.py --limit 20
  python benchmark_sweep/run_sweep.py --workers 12
"""

import os
import csv
import time
import argparse
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

api_key = os.environ.get("FIREWORKS_API_KEY")
base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")

if not api_key:
    print("[ERROR] FIREWORKS_API_KEY not found in .env or environment.", flush=True)
    sys.exit(1)

client = openai.OpenAI(api_key=api_key, base_url=base_url)

# ---------------------------------------------------------------------------
# Model tiers (remote only)
# ---------------------------------------------------------------------------

TIER1_MODEL = "accounts/fireworks/models/gpt-oss-20b"    # Low tier
TIER2_MODEL = "accounts/fireworks/models/qwen3p7-plus"   # Mid tier
TIER3_MODEL = "accounts/fireworks/models/glm-5p2"        # High tier

# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

csv_lock   = threading.Lock()
print_lock = threading.Lock()
count_lock = threading.Lock()
processed_count = 0

# ---------------------------------------------------------------------------
# API query with exponential-backoff retry
# ---------------------------------------------------------------------------

def query_fireworks(prompt: str, model: str, max_retries: int = 6) -> tuple[str, int]:
    """
    Query a Fireworks model. Returns (response_text, total_tokens).
    Returns ("", 0) after all retries are exhausted.

    Note: reasoning models (e.g. gpt-oss-20b) may return content=None when
    max_tokens is too low to finish the reasoning chain. We fall back to
    reasoning_content in that case, and use max_tokens=1024 to give enough room.
    """
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            msg    = completion.choices[0].message
            # Reasoning models (e.g. gpt-oss-20b) put the final answer in
            # content; when content is None the model ran out of tokens mid-
            # reasoning — fall back to reasoning_content so we still get text.
            text = msg.content
            if not text:
                text = getattr(msg, "reasoning_content", None) or ""
            tokens = completion.usage.total_tokens if completion.usage else 0
            return text, tokens
        except Exception as e:
            err = str(e).lower()
            retryable = any(k in err for k in ("rate limit", "429", "overloaded", "timeout", "503", "502"))
            if retryable and attempt < max_retries - 1:
                with print_lock:
                    print(f"      [WARN] {model.split('/')[-1]} attempt {attempt+1}/{max_retries}: {e}. Retry in {backoff:.0f}s", flush=True)
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 60.0)
            else:
                with print_lock:
                    print(f"      [ERROR] {model.split('/')[-1]}: {e}", flush=True)
                return "", 0
    return "", 0

# ---------------------------------------------------------------------------
# Per-row worker
# ---------------------------------------------------------------------------

def process_row(row: dict, idx: int, total: int, dataset_path: Path, fieldnames: list, all_rows: list) -> bool:
    global processed_count

    pid    = row.get("prompt_id", "?")
    source = row.get("source", "")
    prompt = row.get("prompt", "")

    # Skip only if ALL three tier responses are already filled (resumable).
    # This catches rows where tier1/tier2 came back None (reasoning model bug)
    # even if tier3 was filled.
    t1_done = str(row.get("tier1_response", "")).strip() not in ("", "nan", "None")
    t2_done = str(row.get("tier2_response", "")).strip() not in ("", "nan", "None")
    t3_done = str(row.get("tier3_response", "")).strip() not in ("", "nan", "None")
    if t1_done and t2_done and t3_done:
        return False

    with print_lock:
        print(f"[{idx+1}/{total}] {pid} ({source})", flush=True)

    # Only re-query tiers that are missing — saves API calls on partial rows
    if not t1_done:
        t1_resp, t1_tok = query_fireworks(prompt, TIER1_MODEL)
    else:
        t1_resp = str(row.get("tier1_response", ""))
        t1_tok  = int(str(row.get("tier1_tokens", "0") or "0"))

    if not t2_done:
        t2_resp, t2_tok = query_fireworks(prompt, TIER2_MODEL)
    else:
        t2_resp = str(row.get("tier2_response", ""))
        t2_tok  = int(str(row.get("tier2_tokens", "0") or "0"))

    if not t3_done:
        t3_resp, t3_tok = query_fireworks(prompt, TIER3_MODEL)
    else:
        t3_resp = str(row.get("tier3_response", ""))
        t3_tok  = int(str(row.get("tier3_tokens", "0") or "0"))

    with print_lock:
        print(
            f"    {pid} done — "
            f"T1:{t1_tok}tok  T2:{t2_tok}tok  T3:{t3_tok}tok",
            flush=True,
        )

    # Write into the shared row dict
    row["tier1_response"] = t1_resp
    row["tier1_tokens"]   = str(t1_tok)
    row["tier2_response"] = t2_resp
    row["tier2_tokens"]   = str(t2_tok)
    row["tier3_response"] = t3_resp
    row["tier3_tokens"]   = str(t3_tok)

    with count_lock:
        processed_count += 1
        current = processed_count

    # Checkpoint every 10 processed rows
    if current % 10 == 0:
        with csv_lock:
            with print_lock:
                print(f"[*] Checkpoint at {current} rows processed — saving {dataset_path.name}...", flush=True)
            _save(dataset_path, fieldnames, all_rows)

    return True

# ---------------------------------------------------------------------------
# Save helper with retry (handles Windows file-lock edge cases)
# ---------------------------------------------------------------------------

def _save(path: Path, fieldnames: list, rows: list, max_retries: int = 5):
    for attempt in range(max_retries):
        try:
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return
        except PermissionError:
            if attempt == max_retries - 1:
                raise
            with print_lock:
                print(f"      [WARN] PermissionError saving {path.name}. Retry in 1s...", flush=True)
            time.sleep(1)

# ---------------------------------------------------------------------------
# Main sweep orchestrator
# ---------------------------------------------------------------------------

def run_sweep(dataset_path: Path, limit: int | None, workers: int):
    global processed_count

    print(f"[*] Loading dataset from {dataset_path}...", flush=True)
    with open(dataset_path, "r", encoding="utf-8", newline="") as f:
        reader    = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows      = [dict(r) for r in reader]
    print(f"    {len(rows)} total rows.", flush=True)

    # Identify rows that still need processing — any tier missing counts
    def _needs_work(row: dict) -> bool:
        for key in ("tier1_response", "tier2_response", "tier3_response"):
            val = str(row.get(key, "")).strip()
            if val in ("", "nan", "None"):
                return True
        return False

    todo = [(idx, row) for idx, row in enumerate(rows) if _needs_work(row)]
    print(f"    {len(todo)} rows still need remote responses.", flush=True)

    if limit is not None:
        todo = todo[:limit]
        print(f"    Limiting to {len(todo)} rows (--limit flag).", flush=True)

    if not todo:
        print("[OK] Nothing to do — all rows already swept.", flush=True)
        return

    start = time.time()
    print(f"[*] Starting sweep with {workers} parallel workers...\n", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_row, row, idx, len(rows), dataset_path, fieldnames, rows): idx
            for idx, row in todo
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                with print_lock:
                    print(f"[ERROR] Worker failed: {e}", flush=True)

    # Final save
    with csv_lock:
        print(f"\n[*] Final save to {dataset_path}...", flush=True)
        _save(dataset_path, fieldnames, rows)

    elapsed = time.time() - start
    done = sum(1 for r in rows if str(r.get("tier3_response", "")).strip())
    print(f"[OK] Sweep complete. {processed_count} rows processed this run ({done}/{len(rows)} total done).", flush=True)
    print(f"     Total time: {elapsed/60:.1f} min", flush=True)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark sweep — remote Fireworks tiers only.")
    parser.add_argument("--dataset", default="data_builder/dataset_sweep.csv",
                        help="Path to the sweep CSV (default: data_builder/dataset_sweep.csv)")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Max rows to process (for testing)")
    parser.add_argument("--workers", type=int, default=12,
                        help="Parallel worker threads (default: 12)")
    args = parser.parse_args()

    run_sweep(Path(args.dataset), args.limit, args.workers)
