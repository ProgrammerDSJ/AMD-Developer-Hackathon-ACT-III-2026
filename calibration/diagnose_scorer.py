"""
calibration/diagnose_scorer.py
-------------------------------
Runs 15 calibration prompts through the local model and shows:
  - Raw model response
  - What our scorer extracted
  - Whether marked correct/incorrect
  - Whether it SHOULD have been correct (visual check)

Usage: python calibration/diagnose_scorer.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich         import box
from rich.rule    import Rule
from rich.markup  import escape

from inference_wrapper.local_client  import detect_ollama, best_model, generate
from calibration.run_calibration     import load_config, _extract_letter, _extract_number

CONSOLE  = Console()
CAL_PATH = ROOT / "calibration" / "calibration_prompts.jsonl"

# Test with a smaller sample per source
SAMPLE = {"mmlu": 3, "arc": 3, "gsm8k": 3, "humaneval": 2, "truthfulqa": 2}


def diagnose():
    CONSOLE.print(Panel.fit(
        "[bold cyan]Calibration Scorer Diagnosis[/bold cyan]\n"
        "[dim]Shows raw model responses vs what the scorer extracted.[/dim]",
        border_style="cyan"
    ))

    # Load model
    running, models = detect_ollama()
    if not running or not models:
        CONSOLE.print("[red]Ollama not running.[/red]")
        return
    cfg          = load_config()
    active_name  = cfg.get("active_model")
    model_info   = next((m for m in models if m["name"] == active_name), best_model(models))
    model_name   = model_info["name"]
    CONSOLE.print(f"\nDiagnosing with: [bold cyan]{model_name}[/bold cyan]\n")

    # Load calibration prompts, sample per source
    all_prompts = [json.loads(l) for l in CAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    sample_list = []
    counts = {}
    for p in all_prompts:
        src = p["source"]
        limit = SAMPLE.get(src, 0)
        if counts.get(src, 0) < limit:
            sample_list.append(p)
            counts[src] = counts.get(src, 0) + 1
        if len(sample_list) >= sum(SAMPLE.values()):
            break

    false_neg = 0   # correct but marked wrong
    true_neg  = 0   # genuinely wrong

    for p in sample_list:
        CONSOLE.print(Rule(f"[bold]{p['source']}  ({p['evaluator']})[/bold]", style="dim"))

        prompt    = p["prompt"]
        reference = p["reference"]
        evaluator = p["evaluator"]

        CONSOLE.print(f"[dim]Prompt (truncated):[/dim] {escape(prompt[:200])}")
        CONSOLE.print(f"[dim]Reference:[/dim]          [green]{escape(str(reference)[:100])}[/green]")

        # Generate
        CONSOLE.print("[dim]Calling local model...[/dim]", end="")
        response, latency = generate(prompt, model_name, max_tokens=256)
        CONSOLE.print(f" done ({latency:.1f}s)")

        # Show raw response
        CONSOLE.print(Panel(
            escape(response[:400]) + ("..." if len(response) > 400 else ""),
            title="[dim]Raw response[/dim]", border_style="dim", padding=(0,1)
        ))

        # What our scorer extracted and decided
        if evaluator in ("mcq", "mcq_keyword"):
            extracted = _extract_letter(response)
            expected  = _extract_letter(reference) or reference.strip().upper()
            marked_ok = bool(extracted) and extracted == expected
            CONSOLE.print(f"  Scorer extracted: [bold]{extracted or '(nothing)'}[/bold]  |  "
                          f"Expected: [bold]{expected}[/bold]  |  "
                          f"Verdict: {'[green]CORRECT[/green]' if marked_ok else '[red]WRONG[/red]'}")

            # Heuristic: does the response CONTAIN the right answer somewhere?
            contains_answer = expected in response.upper()
            if not marked_ok and contains_answer:
                CONSOLE.print(f"  [yellow]** POSSIBLE FALSE NEGATIVE: response contains '{expected}' "
                              f"but scorer missed it **[/yellow]")
                false_neg += 1
            elif not marked_ok:
                true_neg += 1

        elif evaluator == "math":
            extracted = _extract_number(response)
            expected  = _extract_number(reference)
            try:    marked_ok = bool(extracted and expected and abs(float(extracted) - float(expected)) < 0.02)
            except: marked_ok = extracted == expected
            CONSOLE.print(f"  Scorer extracted: [bold]{extracted or '(nothing)'}[/bold]  |  "
                          f"Expected: [bold]{expected}[/bold]  |  "
                          f"Verdict: {'[green]CORRECT[/green]' if marked_ok else '[red]WRONG[/red]'}")

            # Heuristic: does the response contain the expected number?
            if not marked_ok and expected and expected in response.replace(",",""):
                CONSOLE.print(f"  [yellow]** POSSIBLE FALSE NEGATIVE: response contains '{expected}' "
                              f"but scorer extracted '{extracted}' instead **[/yellow]")
                false_neg += 1
            elif not marked_ok:
                true_neg += 1

        elif evaluator == "code":
            marked_ok = "def " in response or "return " in response or "print(" in response
            CONSOLE.print(f"  Code check: has_def={'def ' in response}  "
                          f"has_return={'return ' in response}  "
                          f"Verdict: {'[green]CORRECT[/green]' if marked_ok else '[red]WRONG[/red]'}")

        CONSOLE.print()

    CONSOLE.print(Rule("[bold cyan]Diagnosis Summary[/bold cyan]", style="cyan"))
    total = len(sample_list)
    CONSOLE.print(f"  Prompts tested:          {total}")
    CONSOLE.print(f"  False negatives (scorer missed correct answer): [yellow]{false_neg}[/yellow]")
    CONSOLE.print(f"  True negatives (genuinely wrong):               {true_neg}")
    CONSOLE.print(f"  Estimated scorer error rate: [yellow]{false_neg/max(total,1)*100:.1f}%[/yellow]")
    if false_neg > 0:
        CONSOLE.print(f"\n  [yellow]Verdict: YES — scorer is too strict.[/yellow]")
        CONSOLE.print(f"  [dim]The model is giving correct answers with explanations, "
                      f"but the scorer misses them.[/dim]")
        CONSOLE.print(f"  [dim]Actual accuracy is likely {false_neg/total*100:.0f}%+ higher than reported.[/dim]")
    else:
        CONSOLE.print(f"\n  [green]Verdict: Scorer appears accurate for this sample.[/green]")


if __name__ == "__main__":
    diagnose()
