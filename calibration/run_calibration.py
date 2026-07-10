"""
calibration/run_calibration.py
-------------------------------
Calibrates ALL installed Ollama models.
Stores per-model results in ~/.hybridrouter/config.json.

Usage:
  python calibration/run_calibration.py           -- calibrate all new models
  python calibration/run_calibration.py --force   -- recalibrate everything
"""

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console  import Console
from rich.panel    import Panel
from rich.table    import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich          import box
from rich.prompt   import Confirm

from inference_wrapper.local_client import detect_ollama, score_model, generate

CONSOLE     = Console()
CONFIG_PATH = Path.home() / ".hybridrouter" / "config.json"
CAL_PATH    = ROOT / "calibration" / "calibration_prompts.jsonl"

FORCE = "--force" in sys.argv


# ---------------------------------------------------------------------------
# Calibration curve: calibration accuracy -> local routing threshold
# ---------------------------------------------------------------------------
def acc_to_threshold(acc: float) -> float:
    """Higher accuracy local model -> lower threshold -> routes more locally."""
    if   acc >= 0.90: return 0.28
    elif acc >= 0.80: return 0.40
    elif acc >= 0.70: return 0.52
    elif acc >= 0.60: return 0.65
    elif acc >= 0.50: return 0.78
    else:             return 0.95   # very weak -> almost never route locally


# ---------------------------------------------------------------------------
# Deterministic scorers (no Fireworks tokens)
# ---------------------------------------------------------------------------
def _extract_letter(text: str) -> str:
    """
    Smart MCQ answer extractor — handles verbose/step-by-step responses.
    Priority order:
      1. Explicit answer markers: 'answer is D', 'answer: D', 'correct answer is D'
      2. Last standalone letter in the final 250 chars (the conclusion)
      3. First standalone letter in first 300 chars (fallback)
    """
    t = text.strip()

    # Priority 1: explicit answer/conclusion markers
    explicit = [
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*[:\-]?\s*\**([A-D])\b",
        r"(?:the\s+)?(?:correct\s+)?(?:answer|choice|option)\s*[:\-]\s*\**([A-D])\b",
        r"\btherefore[,\s]+(?:the\s+)?(?:answer|choice|option)?\s*(?:is)?\s*\**([A-D])\b",
        r"\bso[,\s]+(?:the\s+)?(?:answer)?\s*(?:is)?\s*\**([A-D])\b",
        r"^\**([A-D])[\)\.]\s",                    # starts with "D. " or "D) "
        r"\n\**([A-D])[\)\.]\s",                   # newline then "D. "
        r"(?:select|choose|pick)\s+(?:option\s+)?\**([A-D])\b",
    ]
    for pat in explicit:
        m = re.search(pat, t, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Priority 2: last standalone A-D letter in final 250 chars (usually the conclusion)
    tail = t[-250:] if len(t) > 250 else t
    matches = list(re.finditer(r"\b([A-D])\b", tail))
    if matches:
        return matches[-1].group(1).upper()

    # Priority 3: first standalone A-D letter anywhere
    m = re.search(r"\b([A-D])\b", t)
    return m.group(1).upper() if m else ""

def _extract_number(text: str) -> str:
    """
    Extract the final numerical answer from math responses.
    Looks for 'answer is X', '= X', or just the last number.
    """
    # Remove thousands commas and currency
    clean = text.replace(",", "").replace("$", "").replace("\u00a0", " ")

    # Priority 1: last '=' result (most reliable for step-by-step math)
    eq_matches = list(re.finditer(r"=\s*(-?\d+(?:\.\d+)?)", clean[-600:]))
    if eq_matches:
        return eq_matches[-1].group(1)

    # Priority 2: explicit answer markers (not 'total:' which is a label)
    m = re.search(
        r"(?:the\s+)?(?:answer|result|equals?)\s*(?:is)?[\s:=]+(-?\d+(?:\.\d+)?)",
        clean[-600:], re.IGNORECASE
    )
    if m:
        return m.group(1)

    # Priority 3: last number in the final portion
    nums = re.findall(r"-?\d+(?:\.\d+)?", clean[-400:])
    return nums[-1] if nums else ""

def _score(response: str, reference: str, evaluator: str) -> bool:
    r = response.strip()
    if evaluator in ("mcq", "mcq_keyword"):
        got = _extract_letter(r)
        exp = _extract_letter(reference) or reference.strip().upper()
        return bool(got) and got == exp
    elif evaluator == "math":
        got, exp = _extract_number(r), _extract_number(reference)
        try:
            # Explicitly wrap in bool() — Python's 'and' returns the last evaluated
            # operand, not True/False, so "" and exp returns "" not False.
            return bool(got and exp and abs(float(got) - float(exp)) < 0.02)
        except Exception:
            return bool(got == exp)
    elif evaluator == "code":
        return "def " in r or "return " in r or "print(" in r
    return False


# ---------------------------------------------------------------------------
# Calibrate one model
# ---------------------------------------------------------------------------
def calibrate_model(model_name: str, prompts: list) -> dict:
    """Run 100 calibration prompts through one model. Returns stats dict."""
    CONSOLE.print(f"\n[bold cyan]Calibrating:[/bold cyan] [white]{model_name}[/white]")

    correct = 0
    source_stats = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=38),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[dim]{task.fields[status]}"),
        TimeElapsedColumn(),
        console=CONSOLE,
    ) as prog:
        task = prog.add_task("Running", total=len(prompts), status="")
        for i, p in enumerate(prompts):
            src = p["source"]
            resp, _ = generate(p["prompt"], model_name, max_tokens=128)
            ok = bool(_score(resp, p["reference"], p["evaluator"]))
            if ok:
                correct += 1
            source_stats.setdefault(src, {"n": 0, "c": 0})
            source_stats[src]["n"] += 1
            source_stats[src]["c"] += 1 if ok else 0
            prog.update(task, advance=1,
                        status=f"{correct}/{i+1} correct ({src})")

    acc       = correct / len(prompts)
    threshold = acc_to_threshold(acc)

    # Per-source breakdown table
    rt = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    rt.add_column("Source",   style="dim", width=14)
    rt.add_column("Correct",  justify="right", width=8)
    rt.add_column("Total",    justify="right", width=8)
    rt.add_column("Accuracy", justify="right", width=10)
    for src, s in source_stats.items():
        pct   = s["c"] / s["n"] * 100
        color = "green" if pct >= 70 else "yellow" if pct >= 50 else "red"
        rt.add_row(src, str(s["c"]), str(s["n"]),
                   f"[{color}]{pct:.0f}%[/{color}]")
    rt.add_row("[bold]TOTAL[/bold]", str(correct), str(len(prompts)),
               f"[bold cyan]{acc*100:.1f}%[/bold cyan]")
    CONSOLE.print(rt)

    CONSOLE.print(
        f"  [bold]Result:[/bold] accuracy=[cyan]{acc*100:.1f}%[/cyan]  "
        f"threshold=[green]{threshold:.2f}[/green]  "
        f"(routes locally when tier1_prob >= {threshold:.2f})"
    )

    return {
        "calibration_acc": round(acc, 4),
        "local_threshold": round(threshold, 4),
        "capability_score": score_model(model_name),
        "source_stats": {k: {"acc": round(v["c"]/v["n"], 3)}
                         for k, v in source_stats.items()},
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompts_run": len(prompts),
    }


# ---------------------------------------------------------------------------
# Load / save config
# ---------------------------------------------------------------------------
def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
            # --- Migrate old flat format to new models-dict format ---
            if "local_model" in cfg and "models" not in cfg:
                name = cfg["local_model"]
                cfg = {
                    "models": {
                        name: {
                            "calibration_acc":  cfg.get("calibration_acc", 0.0),
                            "local_threshold":  cfg.get("local_threshold", 0.95),
                            "capability_score": cfg.get("capability_score", 1),
                            "calibrated_at":    cfg.get("calibrated_at", ""),
                            "prompts_run":      100,
                            "source_stats":     {},
                        }
                    },
                    "active_model": name,
                }
                CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
            return cfg
        except Exception:
            pass
    return {"models": {}, "active_model": None}

def save_config(cfg: dict):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_calibration_flow():
    CONSOLE.print(Panel.fit(
        "[bold cyan]HybridRouter -- Local Model Calibration[/bold cyan]\n"
        "[dim]Calibrates all installed Ollama models. Results saved per-model.[/dim]",
        border_style="cyan"
    ))

    # Detect Ollama
    CONSOLE.print("\n[bold]Detecting Ollama...[/bold]")
    running, models = detect_ollama()

    if not running or not models:
        CONSOLE.print("[yellow]Ollama not running or no models installed.[/yellow]")
        CONSOLE.print("[dim]All prompts will route to remote tiers.[/dim]")
        cfg = load_config()
        cfg["models"] = {}
        cfg["active_model"] = None
        save_config(cfg)
        return

    # Show available models
    mt = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    mt.add_column("Model",     style="cyan",  width=28)
    mt.add_column("Cap Score", justify="right", width=10)
    mt.add_column("Status",    width=22)
    cfg = load_config()
    already = cfg.get("models", {})
    for m in sorted(models, key=lambda x: -score_model(x["name"])):
        name   = m["name"]
        cap    = score_model(name)
        status = "[green]Calibrated[/green]" if (name in already and not FORCE) \
                 else "[yellow]Needs calibration[/yellow]"
        mt.add_row(name, str(cap), status)
    CONSOLE.print(mt)

    # Load calibration prompts
    if not CAL_PATH.exists():
        CONSOLE.print(f"[red]calibration_prompts.jsonl not found![/red]")
        CONSOLE.print(f"Run: [cyan]python calibration/extract_calibration_set.py[/cyan]")
        return
    prompts = [json.loads(l) for l in CAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    CONSOLE.print(f"\nLoaded [bold]{len(prompts)}[/bold] calibration prompts\n")

    # Calibrate each model that needs it
    calibrated_any = False
    for m in sorted(models, key=lambda x: -score_model(x["name"])):
        name = m["name"]
        if name in already and not FORCE:
            CONSOLE.print(f"[dim]Skipping {name} (already calibrated). Use --force to redo.[/dim]")
            continue
        do_it = Confirm.ask(f"Calibrate [cyan]{name}[/cyan]?", default=True)
        if not do_it:
            continue
        result = calibrate_model(name, prompts)
        cfg["models"][name] = result
        calibrated_any = True
        save_config(cfg)   # save after each model in case of interruption

    # Set active model to best calibrated
    if cfg["models"]:
        best = max(cfg["models"].keys(),
                   key=lambda n: cfg["models"][n]["capability_score"])
        cfg["active_model"] = best
        save_config(cfg)

    # Final summary
    CONSOLE.print()
    st = Table(title="[bold]All Calibrated Models[/bold]",
               box=box.ROUNDED, border_style="green")
    st.add_column("Model",     style="bold cyan", width=28)
    st.add_column("Cal Acc",   justify="right", width=10)
    st.add_column("Threshold", justify="right", width=12)
    st.add_column("Local %",   justify="right", width=10)
    active = cfg.get("active_model")
    for name, data in sorted(cfg["models"].items(),
                              key=lambda x: -x[1]["capability_score"]):
        marker = "[green]*[/green] " if name == active else "  "
        local_pct = f"~{int((1 - data['local_threshold']) * 100)}%"
        st.add_row(
            marker + name,
            f"{data['calibration_acc']*100:.1f}%",
            str(data["local_threshold"]),
            local_pct,
        )
    CONSOLE.print(st)
    CONSOLE.print(f"\n[dim]Config: {CONFIG_PATH}[/dim]")
    if active:
        CONSOLE.print(f"[bold green]Active model: {active}[/bold green]")


if __name__ == "__main__":
    run_calibration_flow()
