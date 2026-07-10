"""
cli/dry_run.py
Test the full decision flow without making any API calls.
Shows exactly what the CLI would display per prompt.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.rule    import Rule
from rich         import box
from rich.markup  import escape

from inference_wrapper.feature_extractor import extract_features
from inference_wrapper.router_core       import predict

CONSOLE = Console()

TEST_PROMPTS = [
    ("What is 2 + 2?",                                                   "simple arithmetic"),
    ("Solve: If 3x - 7 = 11, find x.",                                   "algebra"),
    ("Write a binary search algorithm in Python.",                        "code generation"),
    ("What is the capital of France?\nA. Berlin\nB. Paris\nC. Rome\nD. Madrid\nAnswer:", "science MCQ"),
    ("Explain the philosophical implications of the Ship of Theseus.",    "open-ended reasoning"),
    ("Analyze the ethical trade-offs of autonomous AI decision-making.",   "complex analytical"),
]

# Simulate config: local model with 78% calibration -> threshold = 0.42
SIMULATED_CONFIG = {
    "local_enabled":   True,
    "local_model":     "llama3.2:3b",
    "calibration_acc": 0.78,
    "local_threshold": 0.42,
}

def dry_run():
    CONSOLE.print(Panel.fit(
        "[bold cyan]HybridRouter — Dry Run (no API calls)[/bold cyan]\n"
        "[dim]Simulating full decision flow for 6 test prompts[/dim]",
        border_style="cyan"
    ))

    results = []
    threshold = SIMULATED_CONFIG["local_threshold"]

    for prompt, label in TEST_PROMPTS:
        CONSOLE.print()
        CONSOLE.print(Rule(f"[bold white]{label}[/bold white]", style="dim"))
        CONSOLE.print(Panel(f"[white]{escape(prompt[:200])}[/white]",
                            border_style="white", padding=(0,1)))

        feats = extract_features(prompt)
        tier, t1p, t2p = predict(feats)

        use_local = t1p >= threshold

        # Feature highlights
        ft = Table(box=box.SIMPLE, show_header=False, padding=(0,1))
        ft.add_column("Feature", style="dim", width=28)
        ft.add_column("Value", width=12)
        ft.add_column("Feature", style="dim", width=28)
        ft.add_column("Value", width=12)
        items = [
            ("task_type",          feats["_task_type"]),
            ("prompt_length",      feats["prompt_length"]),
            ("has_code_block",     feats["has_code_block"]),
            ("has_math_symbols",   feats["has_math_symbols"]),
            ("reasoning_depth",    feats["llm_reasoning_depth"]),
            ("complexity",         feats["complexity_heuristic"]),
            ("requires_factual",   feats["llm_requires_factual_recall"]),
            ("ambiguity_score",    feats["llm_ambiguity_score"]),
        ]
        for i in range(0, len(items), 2):
            k1,v1 = items[i]; k2,v2 = items[i+1] if i+1<len(items) else ("","")
            ft.add_row(k1, str(v1), k2, str(v2))
        CONSOLE.print(ft)

        # Router decision
        p1b = "#" * int(t1p * 25) + "-" * (25 - int(t1p * 25))
        p2b = "#" * int(t2p * 25) + "-" * (25 - int(t2p * 25))
        CONSOLE.print(f"  [green]{p1b}[/green]  tier1_prob={t1p:.3f}  [red]{p2b}[/red]  tier2_prob={t2p:.3f}")

        if use_local:
            dest = "local"
            CONSOLE.print(f"  [bold green]>> LOCAL ({SIMULATED_CONFIG['local_model']})[/bold green]"
                          f"  [dim]tier1_prob {t1p:.3f} >= threshold {threshold:.2f}[/dim]"
                          f"  [green]0 tokens[/green]")
        elif tier == "tier1":
            dest = "tier1"
            CONSOLE.print(f"  [bold cyan]>> TIER1 (gpt-oss-20b)[/bold cyan]"
                          f"  [dim]tier1_prob {t1p:.3f} < threshold {threshold:.2f}[/dim]"
                          f"  [dim]~300 tokens estimated[/dim]")
        else:
            dest = "tier2"
            CONSOLE.print(f"  [bold yellow]>> TIER2 (glm-5p2)[/bold yellow]"
                          f"  [dim]remote router says hard prompt[/dim]"
                          f"  [dim]~800 tokens estimated[/dim]")

        results.append(dest)

    # Summary
    CONSOLE.print()
    CONSOLE.print(Rule("[bold cyan]Dry Run Summary[/bold cyan]", style="cyan"))
    st = Table(box=box.ROUNDED, border_style="cyan")
    st.add_column("Destination", style="bold")
    st.add_column("Count", justify="right")
    st.add_column("Token Range", justify="right")
    for dest, label, color, tok in [
        ("local", "Local (Ollama)", "green", "0"),
        ("tier1", "Tier 1 — gpt-oss-20b", "cyan", "~200-500"),
        ("tier2", "Tier 2 — glm-5p2", "yellow", "~500-1500"),
    ]:
        n = results.count(dest)
        if n:
            st.add_row(f"[{color}]{label}[/{color}]", str(n), tok)
    CONSOLE.print(st)
    CONSOLE.print(f"\n  [dim]Simulated with: local_model=llama3.2:3b, "
                  f"calibration_acc=78%, threshold={threshold}[/dim]")

if __name__ == "__main__":
    dry_run()
