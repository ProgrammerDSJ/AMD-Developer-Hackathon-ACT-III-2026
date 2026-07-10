"""
benchmark_sweep/evaluate.py
-----------------------------
Phase 4 — Response Evaluation

Reads  : data_builder/dataset_sweep.csv  (has tier1/2/3 responses + tokens)
Writes : data_builder/dataset_sweep.csv  (adds tier1/2/3_correct, label, label_encoded)

Scoring strategy per source:
  gsm8k      → regex number extraction  (deterministic, no LLM)
  mmlu       → letter A/B/C/D extraction (deterministic, no LLM)
  arc        → letter A/B/C/D extraction (deterministic, no LLM)
  humaneval  → execute code in subprocess (deterministic, no LLM)
  truthfulqa → MiniMax-M3 judge          (LLM, semantic)
  alpaca     → MiniMax-M3 judge          (LLM, semantic)

Label assignment (cheapest correct remote tier):
  tier1_correct = 1  →  label = "tier1", label_encoded = 0
  tier2_correct = 1  →  label = "tier2", label_encoded = 1
  tier3_correct = 1  →  label = "tier3", label_encoded = 2
  all wrong          →  label = "tier3", label_encoded = 2  (escalate fallback)

Resumable: skips rows where all tier*_correct already filled.
Checkpoints every 20 rows.
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

api_key  = os.environ.get("FIREWORKS_API_KEY")
base_url = os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")

if not api_key:
    print("[ERROR] FIREWORKS_API_KEY not found.", flush=True)
    sys.exit(1)

client = openai.OpenAI(api_key=api_key, base_url=base_url)

JUDGE_MODEL = "accounts/fireworks/models/minimax-m3"

WORKSPACE    = Path(__file__).resolve().parent.parent
DATASET_PATH = WORKSPACE / "data_builder" / "dataset_sweep.csv"
PROMPTS_JSONL = WORKSPACE / "data_builder" / "prompt_collection" / "prompts.jsonl"

# Thread-safety
csv_lock   = threading.Lock()
print_lock = threading.Lock()
count_lock = threading.Lock()
eval_count = 0

# ---------------------------------------------------------------------------
# Load HumanEval test code + entry points from prompts.jsonl
# ---------------------------------------------------------------------------

def load_humaneval_meta(path: Path) -> dict:
    meta = {}
    if not path.exists():
        print(f"[WARN] {path} not found — HumanEval code execution will be skipped.", flush=True)
        return meta
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("prompt_id", "")
                if pid and "test_code" in obj and "entry_point" in obj:
                    meta[pid] = {
                        "test_code":   obj["test_code"],
                        "entry_point": obj["entry_point"],
                    }
            except Exception:
                pass
    return meta

# ---------------------------------------------------------------------------
# Scorer: GSM8K — regex number match
# ---------------------------------------------------------------------------

def score_math(response: str, reference: str) -> int:
    if not response or not reference:
        return 0
    # Clean reference — strip currency/LaTeX symbols and commas
    ref_clean = reference.strip().replace(",", "").replace("$", "").replace("\\", "").strip()
    try:
        ref_val = float(ref_clean)
    except ValueError:
        # Reference is non-numeric — fallback to substring
        return 1 if ref_clean.lower() in response.lower() else 0

    # Strip LaTeX / currency from response before extracting numbers
    resp = (response
            .replace(",", "")
            .replace("$", "")
            .replace("\\$", "")
            .replace("\\boxed{", "")
            .replace("}", "")
            .replace("\\", ""))

    nums = re.findall(r"-?\b\d+\.?\d*\b", resp)
    if not nums:
        return 0

    # Check last 5 numbers (GLM often explains before giving the answer)
    for n in reversed(nums[-5:]):
        try:
            if abs(float(n) - ref_val) < 1e-6:
                return 1
        except ValueError:
            pass

    # Also check numbers immediately after '=' signs
    eq_nums = re.findall(r"=\s*(-?\d+\.?\d*)", resp)
    for n in reversed(eq_nums):
        try:
            if abs(float(n) - ref_val) < 1e-6:
                return 1
        except ValueError:
            pass

    return 0

# ---------------------------------------------------------------------------
# Scorer: MMLU + ARC — letter extraction
# ---------------------------------------------------------------------------

def score_mcq(response: str, reference: str) -> int:
    if not response or not reference:
        return 0
    ref  = reference.strip().upper()
    resp = response.strip().upper()

    # Direct exact or starts-with (e.g. "D" or "D. Succinylcholine")
    if resp == ref:
        return 1
    if resp.startswith(ref) and (len(resp) == 1 or resp[1] in (".", ")", " ", "\n", ":", "*", ",")):
        return 1

    # Search the ENTIRE response — handles verbose GLM explanations
    # More specific patterns first; take the LAST match as the final answer
    patterns = [
        # "The correct answer is D" / "Answer: D" / "Answer is **D"
        r"(?:THE\s+)?(?:CORRECT\s+)?(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)\s*\**([A-D])\b",
        # "D is correct" / "D is the correct answer"
        r"\b([A-D])\b\s+IS\s+(?:THE\s+)?CORRECT",
        # "**D." or "**D)" or "**D " (GLM bold with full option text e.g. **D. fossil fuels**)
        r"\*\*([A-D])[.\):\s]",
        # "**D**" (just letter bolded)
        r"\*\*([A-D])\*\*",
        # "(D)" anywhere
        r"\(([A-D])\)",
        # "option D" / "choice D"
        r"(?:OPTION|CHOICE)\s+([A-D])\b",
        # Letter at start of a line (multiline responses)
        r"(?:^|\n)([A-D])[.\):\s]",
        # Trailing standalone letter at end of response
        r"\b([A-D])\.?\s*$",
    ]
    for pat in patterns:
        matches = re.findall(pat, resp, re.MULTILINE)
        if matches and matches[-1] == ref:
            return 1

    # Last resort: only standalone letter in a very short response
    if len(resp.strip()) <= 15:
        m = re.search(r"\b([A-D])\b", resp)
        if m and m.group(1) == ref:
            return 1

    return 0

# ---------------------------------------------------------------------------
# Scorer: HumanEval — code execution
# ---------------------------------------------------------------------------

def _extract_code(response: str, entry_point: str) -> str:
    # Try fenced Python blocks first
    blocks = re.findall(r"```python(.*?)```", response, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```(.*?)```", response, re.DOTALL)
    if blocks:
        code = "\n".join(blocks).strip()
        code = re.sub(r"^python\s*\n", "", code)
        return code
    # Fallback: find def <entry_point>
    m = re.search(r"\bdef\s+" + re.escape(entry_point) + r"\b", response)
    if m:
        return response[m.start():].strip()
    return response.strip()


def score_code(response: str, prompt: str, test_code: str, entry_point: str) -> int:
    if not response or not test_code or not entry_point:
        return 0
    code = _extract_code(response, entry_point)
    # If the function def is missing, prepend the prompt (which contains the signature)
    if f"def {entry_point}" not in code:
        code = prompt + "\n" + code
    script = code + "\n\n" + test_code
    # Some HumanEval test suites need an explicit call
    if "check(" not in test_code and "check" in test_code:
        script += f"\n\ncheck({entry_point})"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as f:
            f.write(script)
            tmp = f.name
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=6,
        )
        return 1 if result.returncode == 0 else 0
    except Exception:
        return 0
    finally:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass

# ---------------------------------------------------------------------------
# Scorer: TruthfulQA + Alpaca — MiniMax-M3 judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = (
    "You are a strict evaluator. Your job is to decide if a model response "
    "correctly answers a question, consistent with the reference answer. "
    "Consider semantic equivalence — different wording can still be correct. "
    "Reply with only YES or NO. No explanation."
)


def score_with_judge(prompt: str, response: str, reference: str, max_retries: int = 5) -> int:
    if not response or not reference:
        return 0

    user_msg = (
        f"Question: {prompt[:500]}\n\n"
        f"Reference Answer: {reference[:300]}\n\n"
        f"Model Response: {response[:600]}\n\n"
        "Does the model response correctly answer the question, consistent "
        "with the reference answer? Reply YES or NO only."
    )

    backoff = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            text = (resp.choices[0].message.content or "").strip().upper()
            if "YES" in text:
                return 1
            if "NO" in text:
                return 0
            # Ambiguous reply — treat as incorrect
            return 0
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ("rate limit", "429", "503", "overloaded", "timeout")):
                time.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
            else:
                break

    # Hard fallback: simple keyword overlap
    ref_words = set(w.lower() for w in reference.split() if len(w) > 3)
    if not ref_words:
        return 1  # no meaningful reference — assume pass
    resp_lower = response.lower()
    overlap = sum(1 for w in ref_words if w in resp_lower)
    return 1 if (overlap / len(ref_words)) >= 0.4 else 0

# ---------------------------------------------------------------------------
# Route to correct scorer
# ---------------------------------------------------------------------------

def score_response(
    source: str,
    prompt: str,
    response: str,
    reference: str,
    pid: str,
    humaneval_meta: dict,
) -> int:
    if source == "gsm8k":
        return score_math(response, reference)
    elif source in ("mmlu", "arc"):
        return score_mcq(response, reference)
    elif source == "humaneval":
        meta = humaneval_meta.get(pid, {})
        return score_code(
            response, prompt,
            meta.get("test_code", ""),
            meta.get("entry_point", ""),
        )
    else:  # truthfulqa, alpaca, general
        return score_with_judge(prompt, response, reference)

# ---------------------------------------------------------------------------
# Per-row evaluator
# ---------------------------------------------------------------------------

def evaluate_row(
    row: dict,
    idx: int,
    total: int,
    humaneval_meta: dict,
    dataset_path: Path,
    fieldnames: list,
    all_rows: list,
    **kwargs,
) -> bool:
    global eval_count

    pid    = row.get("prompt_id", "?")
    source = row.get("source", "")
    prompt = row.get("prompt", "")
    ref    = row.get("reference_answer", "")

    # --- Skip if already evaluated (unless forced rescore for this source) ---
    def _is_scored(col):
        v = str(row.get(col, "")).strip()
        return v not in ("", "nan", "None")

    force_rescore = source in kwargs.get("rescore_sources", set())
    if not force_rescore and _is_scored("tier1_correct") and _is_scored("tier2_correct") and _is_scored("tier3_correct"):
        return False

    with print_lock:
        tag = " [RESCORE]" if force_rescore else ""
        print(f"[{idx+1}/{total}] {pid} ({source}){tag}", flush=True)

    t1_resp = str(row.get("tier1_response", ""))
    t2_resp = str(row.get("tier2_response", ""))
    t3_resp = str(row.get("tier3_response", ""))

    # Score each tier — always rescore if force_rescore is set
    t1c = score_response(source, prompt, t1_resp, ref, pid, humaneval_meta) if (force_rescore or not _is_scored("tier1_correct")) else int(row["tier1_correct"])
    t2c = score_response(source, prompt, t2_resp, ref, pid, humaneval_meta) if (force_rescore or not _is_scored("tier2_correct")) else int(row["tier2_correct"])
    t3c = score_response(source, prompt, t3_resp, ref, pid, humaneval_meta) if (force_rescore or not _is_scored("tier3_correct")) else int(row["tier3_correct"])

    with print_lock:
        method = "judge" if source in ("truthfulqa", "alpaca") else "deterministic"
        print(f"    T1:{t1c}  T2:{t2c}  T3:{t3c}  [{method}]", flush=True)

    # Write scores
    row["tier1_correct"] = str(t1c)
    row["tier2_correct"] = str(t2c)
    row["tier3_correct"] = str(t3c)

    # Label: cheapest correct tier
    if t1c:
        label, label_encoded = "tier1", 0
    elif t2c:
        label, label_encoded = "tier2", 1
    else:
        label, label_encoded = "tier3", 2  # tier3 correct OR all wrong → escalate

    row["label"]         = label
    row["label_encoded"] = str(label_encoded)

    with count_lock:
        eval_count += 1
        current = eval_count

    # Checkpoint every 20 rows
    if current % 20 == 0:
        with csv_lock:
            with print_lock:
                print(f"[*] Checkpoint at {current} rows — saving...", flush=True)
            _save(dataset_path, fieldnames, all_rows)

    return True

# ---------------------------------------------------------------------------
# Save helper
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
                print(f"      [WARN] PermissionError — retry in 1s...", flush=True)
            time.sleep(1)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evaluation(dataset_path: Path, workers: int = 8, rescore_sources: set = None):
    rescore_sources = rescore_sources or set()
    global eval_count

    print(f"[*] Loading {dataset_path}...", flush=True)
    with open(dataset_path, "r", encoding="utf-8", newline="") as f:
        reader     = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows       = [dict(r) for r in reader]
    print(f"    {len(rows)} total rows.", flush=True)

    humaneval_meta = load_humaneval_meta(PROMPTS_JSONL)
    print(f"    Loaded HumanEval metadata for {len(humaneval_meta)} prompts.", flush=True)

    # Ensure evaluation columns exist in fieldnames
    for col in ("tier1_correct", "tier2_correct", "tier3_correct", "label", "label_encoded"):
        if col not in fieldnames:
            fieldnames.append(col)
            for row in rows:
                row.setdefault(col, "")

    # Rows that need evaluation — either not yet scored, or forced rescore
    def _needs_eval(row):
        src = row.get("source", "")
        if src in rescore_sources:
            return True  # force rescore
        for col in ("tier1_correct", "tier2_correct", "tier3_correct"):
            v = str(row.get(col, "")).strip()
            if v not in ("0", "1"):
                return True
        return False

    todo = [(idx, row) for idx, row in enumerate(rows) if _needs_eval(row)]
    print(f"    {len(todo)} rows need evaluation.", flush=True)

    if not todo:
        print("[OK] All rows already evaluated.", flush=True)
        return

    # Separate deterministic vs judge rows for worker allocation
    det_sources = {"gsm8k", "mmlu", "arc", "humaneval"}
    det_todo    = [(i, r) for i, r in todo if r.get("source","") in det_sources]
    llm_todo    = [(i, r) for i, r in todo if r.get("source","") not in det_sources]

    print(f"    Deterministic: {len(det_todo)} rows", flush=True)
    print(f"    LLM judge    : {len(llm_todo)} rows (MiniMax-M3)", flush=True)
    print(f"\n[*] Starting evaluation with {workers} workers...\n", flush=True)

    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                evaluate_row, row, idx, len(rows),
                humaneval_meta, dataset_path, fieldnames, rows,
                rescore_sources=rescore_sources,
            ): idx
            for idx, row in todo
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                with print_lock:
                    print(f"[ERROR] Worker: {e}", flush=True)

    # Final save
    with csv_lock:
        print(f"\n[*] Final save...", flush=True)
        _save(dataset_path, fieldnames, rows)

    elapsed = time.time() - start
    done    = sum(1 for r in rows if str(r.get("tier1_correct","")).strip() in ("0","1"))
    print(f"[OK] Evaluation complete. {eval_count} rows evaluated ({done}/{len(rows)} total).", flush=True)
    print(f"     Time: {elapsed/60:.1f} min", flush=True)

    # Summary
    import collections
    labels = [r.get("label","") for r in rows if r.get("label","")]
    dist   = collections.Counter(labels)
    print(f"\n=== LABEL DISTRIBUTION ===")
    for k, v in sorted(dist.items()):
        print(f"  {k}: {v} ({v/len(rows)*100:.1f}%)")

    # Correctness rates
    print(f"\n=== MODEL ACCURACY ===")
    for col, name in [("tier1_correct","Tier1 (gpt-oss-20b)"),
                      ("tier2_correct","Tier2 (qwen3p7-plus)"),
                      ("tier3_correct","Tier3 (glm-5p2)")]:
        vals = [int(r[col]) for r in rows if str(r.get(col,"")).strip() in ("0","1")]
        if vals:
            print(f"  {name}: {sum(vals)}/{len(vals)} = {sum(vals)/len(vals)*100:.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate benchmark sweep responses.")
    parser.add_argument("--dataset", default="data_builder/dataset_sweep.csv")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers")
    parser.add_argument("--rescore-sources", nargs="+", default=[],
                        metavar="SOURCE",
                        help="Force re-score these sources even if already evaluated (e.g. gsm8k mmlu arc)")
    args = parser.parse_args()
    run_evaluation(Path(args.dataset), args.workers, rescore_sources=set(args.rescore_sources))
