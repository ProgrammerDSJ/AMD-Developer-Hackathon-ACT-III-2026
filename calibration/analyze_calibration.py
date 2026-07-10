"""
calibration/analyze_calibration.py
-----------------------------------
Deep analysis of calibration results:
  1. Shows per-source accuracy breakdown from stored config
  2. Identifies issues with the current scorer
  3. Shows what the HumanEval scorer is actually doing
  4. Compares expected vs actual scoring strictness

Run: python calibration/analyze_calibration.py
"""

import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.rule    import Rule
from rich         import box

CONSOLE     = Console()
CONFIG_PATH = Path.home() / ".hybridrouter" / "config.json"
CAL_PATH    = ROOT / "calibration" / "calibration_prompts.jsonl"

def analyze():
    CONSOLE.print(Panel.fit(
        "[bold cyan]Calibration Deep Analysis[/bold cyan]\n"
        "[dim]Auditing what the scorer is actually measuring[/dim]",
        border_style="cyan"
    ))

    cfg    = json.loads(CONFIG_PATH.read_text())
    models = cfg.get("models", {})

    # ── 1. Per-model per-source breakdown ───────────────────────────────────
    CONSOLE.print()
    CONSOLE.print(Rule("[bold]Per-Model Accuracy Breakdown[/bold]", style="white"))
    for model_name, data in sorted(models.items(), key=lambda x: -x[1]["capability_score"]):
        ss = data.get("source_stats", {})
        t = Table(title=f"[cyan]{model_name}[/cyan]  "
                        f"(overall={data['calibration_acc']*100:.1f}%,  "
                        f"threshold={data['local_threshold']})",
                  box=box.SIMPLE, border_style="dim")
        t.add_column("Source",    style="bold", width=14)
        t.add_column("Accuracy",  justify="right", width=10)
        t.add_column("Trustworthy?", width=20)
        t.add_column("Notes", style="dim", width=36)

        EVALUATOR_NOTES = {
            "mmlu":       ("YES", "MCQ — deterministic if scorer works"),
            "arc":        ("YES", "MCQ — deterministic if scorer works"),
            "gsm8k":      ("YES", "Math — deterministic"),
            "humaneval":  ("NO ", "[red]INFLATED[/red] — code scorer too lenient"),
            "truthfulqa": ("YES", "MCQ/keyword — deterministic"),
        }

        for src, sdata in ss.items():
            acc   = sdata["acc"] * 100
            trust, note = EVALUATOR_NOTES.get(src, ("?", "unknown"))
            color = "green" if acc >= 60 else "yellow" if acc >= 40 else "red"
            t.add_row(src, f"[{color}]{acc:.0f}%[/{color}]", trust, note)

        # Compute what accuracy WOULD be without HumanEval
        he_acc  = ss.get("humaneval", {}).get("acc", 0) * 15   # 15 HumanEval prompts
        n_total = data.get("prompts_run", 100)
        correct_total = round(data["calibration_acc"] * n_total)
        correct_no_he = correct_total - round(he_acc)
        n_no_he       = n_total - 15
        adj_acc       = correct_no_he / n_no_he if n_no_he > 0 else 0

        t.add_row("[bold]w/o HumanEval[/bold]",
                  f"[yellow]{adj_acc*100:.1f}%[/yellow]",
                  "YES", "Adjusted accuracy (more realistic)")
        CONSOLE.print(t)
        CONSOLE.print()

    # ── 2. What does the HumanEval scorer actually check? ───────────────────
    CONSOLE.print()
    CONSOLE.print(Rule("[bold red]HumanEval Scorer Audit[/bold red]", style="red"))
    CONSOLE.print(
        "\nThe current code scorer accepts a response if it contains:\n"
        "    [bold]\"def \" OR \"return \" OR \"print([\"[/bold]\n\n"
        "This means ALL of the following get marked [bold green]CORRECT[/bold green]:\n\n"
        "  [dim]Response 1:[/dim] [green]\"def f(x): return x + 1\"[/green]\n"
        "                -- Genuinely correct.\n\n"
        "  [dim]Response 2:[/dim] [yellow]\"def helper(): return False\"[/yellow]\n"
        "                -- Has def+return but WRONG answer (trivial stub). False positive.\n\n"
        "  [dim]Response 3:[/dim] [red]\"I can't write code. But here's a print: print('hello')\"[/red]\n"
        "                -- Has print( but completely wrong! False positive.\n\n"
        "  [dim]Response 4:[/dim] [red]\"return is a keyword in Python used to...\"[/red]\n"
        "                -- Has 'return ' in a sentence. False positive.\n\n"
        "[bold]Effect: smollm2 models score 93-100% on HumanEval, inflating overall by ~7-10%[/bold]\n"
    )

    # ── 3. Show what sample HumanEval prompts look like ─────────────────────
    if CAL_PATH.exists():
        prompts = [json.loads(l) for l in CAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
        he_prompts = [p for p in prompts if p["source"] == "humaneval"]
        CONSOLE.print(Rule("[bold]Sample HumanEval Calibration Prompts[/bold]", style="dim"))
        for p in he_prompts[:3]:
            CONSOLE.print(f"  [dim]Reference:[/dim] [green]{p['reference'][:80]}[/green]")
            CONSOLE.print(f"  [dim]Prompt:[/dim]    {p['prompt'][:100].strip().replace(chr(10), ' ')}")
            CONSOLE.print()

    # ── 4. Verdict ───────────────────────────────────────────────────────────
    CONSOLE.print(Rule("[bold cyan]Verdict[/bold cyan]", style="cyan"))
    CONSOLE.print(
        "\n[bold]Three things happening:[/bold]\n\n"
        "[bold cyan]1. HumanEval inflation (most impactful)[/bold cyan]\n"
        "   The code scorer (def/return/print check) gives near-perfect scores to tiny models\n"
        "   that just output stubs like 'return False'. Real code ability is ~0% for <1B models.\n"
        "   Removing HumanEval from calibration gives more honest accuracy.\n\n"
        "[bold cyan]2. New MCQ scorer: slight tradeoff[/bold cyan]\n"
        "   Correctly handles B/C/D answers in step-by-step responses.\n"
        "   May occasionally miss 'A' answers if model discusses other options after.\n"
        "   Net effect: small change (32%% -> 28%% for qwen2.5:0.5b). Not the main issue.\n\n"
        "[bold cyan]3. These models are genuinely weak (most honest answer)[/bold cyan]\n"
        "   qwen2.5:0.5b, smollm2:135m, smollm2:360m are 135M-500M parameter models.\n"
        "   Real accuracy without HumanEval: 8-22%%. Threshold=0.95 is CORRECT.\n"
        "   With a 3B+ model, threshold drops to 0.52-0.40 and ~50%%+ routes locally.\n\n"
        "[bold yellow]Fix applied:[/bold yellow]\n"
        "   Code scorer now requires: def + return/yield + non-trivial body\n"
        "   AND rejects: 'I cannot', 'I don't know', 'sorry', 'as an AI'\n"
    )

if __name__ == "__main__":
    analyze()
