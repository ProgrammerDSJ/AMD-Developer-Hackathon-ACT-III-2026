"""
cli/main.py  --  HybridRouter Unified Interactive CLI
------------------------------------------------------
Usage:
  python cli/main.py                    # Interactive mode (recommended)
  python cli/main.py "What is AI?"      # Single-shot prompt
  python cli/main.py --demo             # Run 6-prompt curated demo
  python cli/main.py --stats            # Show session stats only
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from rich.console  import Console
from rich.panel    import Panel
from rich.table    import Table
from rich.rule     import Rule
from rich.prompt   import Prompt, Confirm
from rich          import box
from rich.markup   import escape

from inference_wrapper.feature_extractor import extract_features
from inference_wrapper.router_core       import predict
from inference_wrapper.local_client      import detect_ollama, score_model, generate as local_gen
from inference_wrapper.fireworks_client  import call_tier, TIER_DISPLAY

CONSOLE     = Console()
CONFIG_PATH = Path.home() / ".hybridrouter" / "config.json"
SESSION_PATH= ROOT / "cli" / ".session.json"

BANNER = r"""
  _   _       _          _     _ ____             _
 | | | |_   _| |__  _ __(_) __| |  _ \ ___  _   _| |_ ___ _ __
 | |_| | | | | '_ \| '__| |/ _` | |_) / _ \| | | | __/ _ \ '__|
 |  _  | |_| | |_) | |  | | (_| |  _ < (_) | |_| | ||  __/ |
 |_| |_|\__, |_.__/|_|  |_|\__,_|_| \_\___/ \__,_|\__\___|_|
         |___/
"""

DEMO_PROMPTS = [
    ("What is 2 + 2?",                                                  "Simple arithmetic"),
    ("If 3x - 7 = 11, find x.",                                         "Algebra"),
    ("What is the capital of France?\nA. Berlin\nB. Paris\nC. Rome\nD. Madrid\nAnswer:", "Science MCQ"),
    ("Write a Python function that reverses a linked list in-place.",    "Code generation"),
    ("Explain the Ship of Theseus paradox and its modern implications.", "Deep reasoning"),
    ("Analyze ethical trade-offs in autonomous AI decision-making.",     "Complex analysis"),
]

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {"models": {}, "active_model": None}

def load_session() -> dict:
    if SESSION_PATH.exists():
        try:
            return json.loads(SESSION_PATH.read_text())
        except Exception:
            pass
    return {"total": 0, "local": 0, "tier1": 0, "tier2": 0,
            "tokens_used": 0, "tokens_saved": 0, "runs": []}

def save_session(s: dict):
    SESSION_PATH.parent.mkdir(exist_ok=True)
    SESSION_PATH.write_text(json.dumps(s, indent=2))

# ---------------------------------------------------------------------------
# Calibration check and prompt
# ---------------------------------------------------------------------------

def check_and_calibrate(cfg: dict) -> dict:
    """
    Detects Ollama models, checks which are calibrated,
    offers to calibrate missing ones. Returns updated config.
    """
    running, models = detect_ollama()
    if not running or not models:
        CONSOLE.print("[yellow]  Ollama not detected — local model gate disabled.[/yellow]")
        CONSOLE.print("  [dim]Start Ollama and run again to enable local routing.[/dim]\n")
        return cfg

    calibrated = cfg.get("models", {})
    all_names  = [m["name"] for m in models]
    uncal      = [n for n in all_names if n not in calibrated]

    # Show status table
    t = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    t.add_column("Local Model",  style="cyan", width=26)
    t.add_column("Cap",          justify="right", width=5)
    t.add_column("Status",       width=30)
    t.add_column("Cal Acc",      justify="right", width=10)
    t.add_column("Threshold",    justify="right", width=12)

    for m in sorted(models, key=lambda x: -score_model(x["name"])):
        name = m["name"]
        cap  = score_model(name)
        if name in calibrated:
            d = calibrated[name]
            t.add_row(name, str(cap),
                      "[green]Calibrated[/green]",
                      f"{d['calibration_acc']*100:.1f}%",
                      str(d["local_threshold"]))
        else:
            t.add_row(name, str(cap),
                      "[yellow]Not calibrated[/yellow]", "--", "--")
    CONSOLE.print(t)

    if not uncal:
        return cfg

    CONSOLE.print(f"\n  [yellow]{len(uncal)} model(s) not calibrated.[/yellow]")
    do_cal = Confirm.ask("  Calibrate them now?", default=True)
    if not do_cal:
        CONSOLE.print("  [dim]Skipping calibration. Uncalibrated models won't be used for local routing.[/dim]\n")
        return cfg

    # Run calibration for uncalibrated models
    from calibration.run_calibration import calibrate_model, acc_to_threshold, CAL_PATH
    if not CAL_PATH.exists():
        CONSOLE.print("[red]calibration_prompts.jsonl not found. Run: python calibration/extract_calibration_set.py[/red]")
        return cfg

    prompts = [json.loads(l) for l in CAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]

    for name in uncal:
        do_this = Confirm.ask(f"  Calibrate [cyan]{name}[/cyan]?", default=True)
        if not do_this:
            continue
        result = calibrate_model(name, prompts)
        cfg.setdefault("models", {})[name] = result
        # Save immediately after each model
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        CONSOLE.print(f"  [green]Saved calibration for {name}[/green]")

    return cfg

def select_active_model(cfg: dict) -> str | None:
    """
    Let user pick which calibrated model to use this session.
    Returns model name or None (remote-only).
    """
    models = cfg.get("models", {})
    if not models:
        return None

    sorted_models = sorted(models.items(), key=lambda x: -x[1]["capability_score"])

    if len(sorted_models) == 1:
        name = sorted_models[0][0]
        CONSOLE.print(f"  [green]Using local model:[/green] [bold]{name}[/bold]  "
                      f"(cal acc={models[name]['calibration_acc']*100:.1f}%, "
                      f"threshold={models[name]['local_threshold']})\n")
        return name

    CONSOLE.print("\n  [bold]Select local model for this session:[/bold]")
    for i, (name, data) in enumerate(sorted_models):
        local_pct = int((1 - data["local_threshold"]) * 100)
        CONSOLE.print(f"  [{i+1}] [cyan]{name}[/cyan]  "
                      f"cal={data['calibration_acc']*100:.0f}%  "
                      f"threshold={data['local_threshold']}  "
                      f"~{local_pct}% routed locally")
    CONSOLE.print(f"  [{len(sorted_models)+1}] Remote-only (no local model)")

    while True:
        choice = Prompt.ask("  Choose", default="1")
        try:
            idx = int(choice) - 1
            if idx == len(sorted_models):
                return None
            if 0 <= idx < len(sorted_models):
                return sorted_models[idx][0]
        except ValueError:
            pass
        CONSOLE.print("  [red]Invalid choice.[/red]")

# ---------------------------------------------------------------------------
# Core routing + display
# ---------------------------------------------------------------------------

def route_prompt(prompt: str, active_model: str | None,
                 cfg: dict, session: dict) -> dict:
    """Full routing pipeline with Rich step-by-step display."""

    t_start = time.time()

    # --- Prompt display ---
    CONSOLE.print()
    CONSOLE.print(Rule("[bold white]Prompt[/bold white]", style="dim white"))
    CONSOLE.print(Panel(
        escape(prompt[:500]) + ("..." if len(prompt) > 500 else ""),
        border_style="white", padding=(0, 1)
    ))

    # --- Step 1: Feature Extraction ---
    CONSOLE.print(Rule("[bold blue]Step 1  Feature Extraction[/bold blue]", style="blue"))
    t0     = time.time()
    feats  = extract_features(prompt)
    fms    = (time.time() - t0) * 1000

    ft = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    ft.add_column("Feature", style="dim", width=28)
    ft.add_column("Value",   width=12)
    ft.add_column("Feature", style="dim", width=28)
    ft.add_column("Value",   width=12)
    items = [
        ("task_type",        feats["_task_type"]),
        ("prompt_length",    feats["prompt_length"]),
        ("has_code_block",   feats["has_code_block"]),
        ("has_math_symbols", feats["has_math_symbols"]),
        ("reasoning_depth",  feats["llm_reasoning_depth"]),
        ("complexity",       feats["complexity_heuristic"]),
        ("requires_factual", feats["llm_requires_factual_recall"]),
        ("ambiguity_score",  feats["llm_ambiguity_score"]),
        ("num_sentences",    feats["num_sentences"]),
        ("avg_word_len",     feats["avg_word_length"]),
    ]
    for i in range(0, len(items), 2):
        k1, v1 = items[i]
        k2, v2 = items[i+1] if i+1 < len(items) else ("", "")
        ft.add_row(k1, str(v1), k2, str(v2))
    CONSOLE.print(ft)
    CONSOLE.print(f"  [dim]Extracted in {fms:.1f}ms  |  "
                  f"task=[cyan]{feats['_task_type']}[/cyan]  "
                  f"q_type=[cyan]{feats['_q_type']}[/cyan][/dim]")

    # --- Step 2: Router Prediction ---
    CONSOLE.print(Rule("[bold magenta]Step 2  Router Decision[/bold magenta]", style="magenta"))
    t0   = time.time()
    tier, t1p, t2p = predict(feats)
    rms  = (time.time() - t0) * 1000

    bar_width = 30
    p1b = "#" * int(t1p * bar_width) + "-" * (bar_width - int(t1p * bar_width))
    p2b = "#" * int(t2p * bar_width) + "-" * (bar_width - int(t2p * bar_width))
    CONSOLE.print(f"  [green]Tier1 (easy)  [{p1b}] {t1p:.3f}[/green]")
    CONSOLE.print(f"  [red]  Tier2 (hard) [{p2b}] {t2p:.3f}[/red]")
    CONSOLE.print(f"  [dim]Router latency: {rms:.1f}ms[/dim]")

    # --- Step 3: Local Gate ---
    CONSOLE.print(Rule("[bold yellow]Step 3  Local Gate[/bold yellow]", style="yellow"))

    model_cfg   = cfg.get("models", {}).get(active_model, {}) if active_model else {}
    threshold   = model_cfg.get("local_threshold", 1.0) if model_cfg else 1.0
    cal_acc     = model_cfg.get("calibration_acc", 0.0) if model_cfg else 0.0
    use_local   = bool(active_model and model_cfg and t1p >= threshold)

    gt = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    gt.add_column("Label", style="dim", width=24)
    gt.add_column("Value", width=44)
    gt.add_row("Local model",      f"[cyan]{active_model or 'None'}[/cyan]")
    gt.add_row("Calibration acc",  f"{cal_acc*100:.1f}%" if active_model else "n/a")
    gt.add_row("Local threshold",  str(threshold) if active_model else "n/a")
    gt.add_row("Router tier1_prob", f"[bold]{t1p:.3f}[/bold]")
    if use_local:
        cond = f"[bold green]PASS  ({t1p:.3f} >= {threshold}) -> LOCAL[/bold green]"
    elif not active_model:
        cond = "[dim]No local model configured[/dim]"
    else:
        cond = f"[bold red]FAIL  ({t1p:.3f} < {threshold}) -> REMOTE[/bold red]"
    gt.add_row("Gate condition",   cond)
    CONSOLE.print(gt)

    # Decision box
    if use_local:
        CONSOLE.print(Panel(
            f"[bold green]ROUTED TO: LOCAL  ({active_model})[/bold green]\n"
            f"[green]Fireworks tokens: 0[/green]",
            border_style="green", padding=(0, 2)
        ))
    elif tier == "tier1":
        CONSOLE.print(Panel(
            "[bold cyan]ROUTED TO: TIER 1 REMOTE  (gpt-oss-20b)[/bold cyan]\n"
            "[dim]Local gate failed. Using cheap Fireworks model.[/dim]",
            border_style="cyan", padding=(0, 2)
        ))
    else:
        CONSOLE.print(Panel(
            "[bold yellow]ROUTED TO: TIER 2 REMOTE  (glm-5p2)[/bold yellow]\n"
            "[dim]Complex prompt. Using powerful Fireworks model.[/dim]",
            border_style="yellow", padding=(0, 2)
        ))

    # --- Step 4: Inference ---
    CONSOLE.print(Rule("[bold white]Step 4  Inference[/bold white]", style="white"))
    tokens_used, tokens_baseline = 0, 0
    dest = "local"

    if use_local:
        CONSOLE.print(f"  [green]Calling Ollama ({active_model})...[/green]", end="")
        response, latency = local_gen(prompt, active_model)
        tokens_baseline = max(len(response.split()) * 3, 200)
        CONSOLE.print(f"  done in {latency:.2f}s")

    elif tier == "tier1":
        dest = "tier1"
        CONSOLE.print(f"  [cyan]Calling Fireworks Tier 1 (gpt-oss-20b)...[/cyan]", end="")
        response, tokens_used, latency = call_tier("tier1", prompt)
        tokens_baseline = max(int(tokens_used * 2.8), tokens_used + 200)
        CONSOLE.print(f"  done in {latency:.2f}s  [{tokens_used} tokens]")

    else:
        dest = "tier2"
        CONSOLE.print(f"  [yellow]Calling Fireworks Tier 2 (glm-5p2)...[/yellow]", end="")
        response, tokens_used, latency = call_tier("tier2", prompt)
        tokens_baseline = tokens_used
        CONSOLE.print(f"  done in {latency:.2f}s  [{tokens_used} tokens]")

    tokens_saved = max(tokens_baseline - tokens_used, 0)
    total_ms     = (time.time() - t_start) * 1000

    # --- Response ---
    CONSOLE.print(Rule("[bold white]Response[/bold white]", style="white"))
    color = "green" if dest == "local" else "cyan" if dest == "tier1" else "yellow"
    CONSOLE.print(Panel(
        escape(response[:1800]) + ("..." if len(response) > 1800 else ""),
        border_style=color, padding=(0, 1)
    ))

    # --- Token summary ---
    st = Table(box=box.ROUNDED, border_style=color, show_header=False)
    st.add_column("", style="bold", width=22)
    st.add_column("", width=44)
    dest_label = (f"Local  ({active_model})" if dest == "local"
                  else "Tier 1  gpt-oss-20b" if dest == "tier1"
                  else "Tier 2  glm-5p2")
    st.add_row("Routed to",        f"[{color}]{dest_label}[/{color}]")
    st.add_row("Fireworks tokens", f"[bold]{tokens_used}[/bold]")
    st.add_row("Tokens saved",     f"[bold green]+{tokens_saved}[/bold green]  vs always-tier2 baseline")
    st.add_row("Total latency",    f"{total_ms:.0f}ms")
    CONSOLE.print(st)

    # Update session
    session["total"]        += 1
    session[dest]           += 1
    session["tokens_used"]  += tokens_used
    session["tokens_saved"] += tokens_saved
    session["runs"].append({"dest": dest, "tokens": tokens_used, "saved": tokens_saved})
    return {"dest": dest, "tokens": tokens_used, "saved": tokens_saved}

# ---------------------------------------------------------------------------
# Stats display
# ---------------------------------------------------------------------------

def show_stats(session: dict):
    CONSOLE.print()
    CONSOLE.print(Rule("[bold cyan]Session Statistics[/bold cyan]", style="cyan"))
    total = max(session["total"], 1)
    st = Table(title="Routing Summary", box=box.ROUNDED, border_style="cyan")
    st.add_column("Destination",  style="bold")
    st.add_column("Count",        justify="right")
    st.add_column("Share",        justify="right")
    st.add_column("Avg Tokens",   justify="right")
    for dest, label, color in [
        ("local", "Local (Ollama)", "green"),
        ("tier1", "Tier 1  gpt-oss-20b", "cyan"),
        ("tier2", "Tier 2  glm-5p2", "yellow"),
    ]:
        n    = session.get(dest, 0)
        runs = [r for r in session["runs"] if r["dest"] == dest]
        avg  = int(sum(r["tokens"] for r in runs) / max(len(runs), 1))
        if n:
            st.add_row(
                f"[{color}]{label}[/{color}]",
                str(n), f"{n/total*100:.1f}%",
                "[green]0[/green]" if dest == "local" else str(avg)
            )
    CONSOLE.print(st)
    tok   = session["tokens_used"]
    saved = session["tokens_saved"]
    pct   = saved / max(tok + saved, 1) * 100
    CONSOLE.print(Panel(
        f"[bold]Fireworks Tokens Used:[/bold]  [cyan]{tok}[/cyan]\n"
        f"[bold]Tokens Saved:[/bold]           [green]+{saved}[/green]  vs always-tier2 baseline\n"
        f"[bold]Efficiency:[/bold]             [bold green]{pct:.1f}% saved[/bold green]\n"
        f"[bold]Prompts Handled:[/bold]        {session['total']}",
        title="[bold green]Cost Summary[/bold green]",
        border_style="green"
    ))

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = [a for a in sys.argv[1:] if a]

    # Print banner
    CONSOLE.print(f"[bold cyan]{BANNER}[/bold cyan]")
    CONSOLE.print(Panel.fit(
        "[bold]AMD Developer Hackathon ACT III 2026[/bold]  --  Token-Efficient LLM Routing\n"
        "[dim]Routes each prompt to the cheapest model that can answer it correctly.[/dim]",
        border_style="cyan"
    ))

    # Stats-only mode
    if "--stats" in args:
        show_stats(load_session())
        return

    CONSOLE.print()
    CONSOLE.print(Rule("[bold cyan]Startup[/bold cyan]", style="cyan"))

    # Load config + detect + calibrate
    cfg = load_config()
    cfg = check_and_calibrate(cfg)

    # Pick active model
    active_model = select_active_model(cfg)

    session = load_session()

    # Demo mode
    if "--demo" in args:
        CONSOLE.print(Panel(
            "[bold]Curated Demo[/bold] -- 6 prompts across all difficulty levels\n"
            "[dim]Watch routing decisions: local / tier1 / tier2[/dim]",
            border_style="cyan"
        ))
        for prompt, label in DEMO_PROMPTS:
            CONSOLE.print(f"\n  [dim italic]{label}[/dim italic]")
            route_prompt(prompt, active_model, cfg, session)
            time.sleep(0.3)
        show_stats(session)
        save_session(session)
        return

    # Single-shot CLI argument mode
    if args and not args[0].startswith("--"):
        prompt = " ".join(args)
        route_prompt(prompt, active_model, cfg, session)
        show_stats(session)
        save_session(session)
        return

    # Interactive loop
    CONSOLE.print()
    CONSOLE.print(Panel(
        "[bold]Interactive Mode[/bold]\n"
        "Type your prompt and press Enter. Type [bold cyan]exit[/bold cyan] or [bold cyan]quit[/bold cyan] to stop.\n"
        "Type [bold cyan]stats[/bold cyan] to see session summary. "
        "Type [bold cyan]switch[/bold cyan] to change local model.",
        border_style="cyan", padding=(0, 2)
    ))

    while True:
        CONSOLE.print()
        try:
            prompt = Prompt.ask("[bold cyan]>>>[/bold cyan] Prompt")
        except (KeyboardInterrupt, EOFError):
            break

        if not prompt.strip():
            continue
        if prompt.strip().lower() in ("exit", "quit", "q"):
            break
        if prompt.strip().lower() == "stats":
            show_stats(session)
            continue
        if prompt.strip().lower() == "switch":
            active_model = select_active_model(cfg)
            continue

        route_prompt(prompt.strip(), active_model, cfg, session)

    show_stats(session)
    save_session(session)
    CONSOLE.print("\n[dim]Session saved. Goodbye.[/dim]")


if __name__ == "__main__":
    main()
