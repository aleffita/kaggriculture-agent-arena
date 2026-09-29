"""Dream-RSI Multi-Stage Arena & Research Notes Engine.

100% LLM Agents competing on the NVIDIA GeForce GTX 1050 Ti via Direct3D 12.
Decouples horizon stages, runs Swiss matchmaking, generates DuckDB telemetry,
and iteratively writes research notes in kaggriculture/notes/ while refining prompt personalities.
"""

from __future__ import annotations

import argparse
import io
import pathlib
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from kaggriculture.db.schema import (
    initialize_schema,
    register_agent,
    ensure_stage_rating,
    get_connection,
)
from kaggriculture.arena.stages import STAGES, STAGE_ORDER
from kaggriculture.arena.arena_engine import run_stage_swiss_round
from kaggriculture.dream.dream_rsi import run_dream_rsi

console = Console(highlight=False)


def seed_initial_roster():
    """Seeds the initial roster with 100% LLM Personalities running on GTX 1050 Ti."""
    initialize_schema()

    llm_personalities = [
        ("llm_sprint_rusher", "v1", "LLM Sprint Rusher: 2-day wheat turnaround, zero weed tolerance"),
        ("llm_land_baron", "v1", "LLM Land Baron: Day 4-5 NE territorial expansion gate ($1050)"),
        ("llm_labor_magnate", "v1", "LLM Labor Magnate: Daily Fibonacci labor scaling and action economy"),
        ("llm_melon_monopolist", "v1", "LLM Melon Monopolist: High-margin Melon compounding & batched selling"),
        ("llm_market_arbitrageur", "v1", "LLM Market Arbitrageur: Dynamic crop portfolio exploiting price spikes"),
        ("llm_cautious_farmer", "v1", "LLM Cautious Farmer: Low-variance shed perimeter carrot cultivation"),
    ]

    for name, ver, desc in llm_personalities:
        ag_id = register_agent(
            name=name,
            version=ver,
            league="Sprint",
            initial_elo=600.0,
            description=desc,
            code_path=f"kaggriculture/agents/{name}.py",
        )
        for stage_id in STAGE_ORDER:
            ensure_stage_rating(ag_id, stage_id, initial_elo=600.0, bracket="Upper")


def display_global_leaderboard(epoch: int):
    """Renders the 100% LLM multi-stage Elo matrix leaderboard and highlights stage champions."""
    con = get_connection(read_only=True)
    rows = con.execute("""
        SELECT 
            a.agent_id,
            COALESCE(MAX(CASE WHEN r.stage = 'Sprint' THEN r.elo ELSE NULL END), 600.0) as elo_sprint,
            COALESCE(MAX(CASE WHEN r.stage = 'Expansion' THEN r.elo ELSE NULL END), 600.0) as elo_exp,
            COALESCE(MAX(CASE WHEN r.stage = 'Scaling' THEN r.elo ELSE NULL END), 600.0) as elo_scal,
            COALESCE(MAX(CASE WHEN r.stage = 'FullSeason' THEN r.elo ELSE NULL END), 600.0) as elo_full,
            COALESCE(MAX(r.peak_elo), 600.0) as max_peak,
            COALESCE(SUM(r.matches_played), 0) as total_matches,
            COALESCE(SUM(r.wins), 0) as total_wins,
            COALESCE(SUM(r.draws), 0) as total_draws,
            COALESCE(SUM(r.losses), 0) as total_losses,
            COALESCE(SUM(r.total_coins), 0.0) as total_coins
        FROM agents a
        LEFT JOIN agent_stage_ratings r ON a.agent_id = r.agent_id
        GROUP BY a.agent_id
        ORDER BY max_peak DESC, total_coins DESC
    """).fetchall()
    con.close()

    table = Table(title=f"Kaggriculture 100% LLM Arena Leaderboard — Epoch {epoch}")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("LLM Personality", style="cyan")
    table.add_column("Sprint (72s)", justify="right", style="green")
    table.add_column("Expansion (144s)", justify="right", style="green")
    table.add_column("Scaling (240s)", justify="right", style="green")
    table.add_column("Full Season (720s)", justify="right", style="green")
    table.add_column("Peak Elo", justify="right", style="bold yellow")
    table.add_column("W - D - L", justify="center")
    table.add_column("Matches", justify="right")
    table.add_column("Total Coins", justify="right", style="magenta")

    for rank, r in enumerate(rows, start=1):
        ag_id, s_elo, exp_elo, sca_elo, ful_elo, peak, matches, w, d, l, coins = r
        table.add_row(
            str(rank),
            ag_id,
            f"{s_elo:.1f}",
            f"{exp_elo:.1f}",
            f"{sca_elo:.1f}",
            f"{ful_elo:.1f}",
            f"{peak:.1f}",
            f"{w}-{d}-{l}",
            str(matches),
            f"${coins:,.0f}",
        )

    console.print("\n", table)

    # Stage Champions Summary
    if rows:
        top_sprint = max(rows, key=lambda x: x[1])
        top_exp = max(rows, key=lambda x: x[2])
        top_scal = max(rows, key=lambda x: x[3])
        top_full = max(rows, key=lambda x: x[4])

        console.print(f"\n[bold yellow][*] TOP ELO LLM STAGE CHAMPIONS (Epoch {epoch}):[/bold yellow]")
        console.print(f"  * [bold cyan]Sprint (3d)[/bold cyan]:      [bold green]{top_sprint[0]}[/bold green] (Elo: {top_sprint[1]:.1f})")
        console.print(f"  * [bold cyan]Expansion (6d)[/bold cyan]:   [bold green]{top_exp[0]}[/bold green] (Elo: {top_exp[2]:.1f})")
        console.print(f"  * [bold cyan]Scaling (10d)[/bold cyan]:    [bold green]{top_scal[0]}[/bold green] (Elo: {top_scal[3]:.1f})")
        console.print(f"  * [bold cyan]Full Season (30d)[/bold cyan]:[bold green]{top_full[0]}[/bold green] (Elo: {top_full[4]:.1f})")


def run_dream_loop(epochs: int = 10, rounds_per_stage: int = 1, active_stages: Optional[list[str]] = None):
    """Executes the full Dream-RSI evolutionary cycle over N epochs across LLM personalities."""
    stages_to_run = active_stages or STAGE_ORDER

    console.print(Panel(
        "[bold cyan]Kaggriculture 100% LLM Arena & Dream-RSI Knowledge Engine[/bold cyan]\n"
        "[dim]Direct step-by-step neural play on NVIDIA GTX 1050 Ti (Google LiteRT-LM / Direct3D 12).\n"
        "Features prompt personality diversity, Swiss pairing, and dialectical knowledge synthesis.[/dim]",
        border_style="cyan"
    ))

    seed_initial_roster()

    for ep in range(1, epochs + 1):
        console.print(f"\n[bold magenta]===========================================================================[/bold magenta]")
        console.print(f"[bold magenta]                         STARTING ARENA EPOCH {ep}/{epochs}                         [/bold magenta]")
        console.print(f"[bold magenta]===========================================================================[/bold magenta]\n")

        # 1. Inner Loop: Play Swiss rounds across stages
        for stage_id in stages_to_run:
            stg = STAGES[stage_id]
            console.print(f"[yellow]Executing {stg.name} (Horizon: {stg.step_horizon} steps / {stg.in_game_days} days | Swiss LLM Duels on GTX 1050 Ti)...[/yellow]")
            res = run_stage_swiss_round(epoch=ep, stage_id=stage_id, rounds=rounds_per_stage)
            console.print(f"  Completed {len(res)} Swiss duels on GTX 1050 Ti in {stage_id}.\n")

        # 2. Outer Loop: Dream-RSI Reflection over DuckDB telemetry
        console.print(f"[cyan]Entering Dream-RSI Knowledge Synthesis for Epoch {ep}...[/cyan]")
        rsi_result = run_dream_rsi(ep)
        console.print(f"  [bold green][NOTE CREATED][/bold green] {rsi_result['note_path']}")
        if rsi_result["refinements"]:
            console.print(f"  [bold yellow][PROMPTS REFINED][/bold yellow] Dialectical patches applied to: {', '.join(rsi_result['refinements'])}\n")
        else:
            console.print("  [dim]All LLM personalities maintained balanced performance.[/dim]\n")

        # 3. Render Epoch Summary Leaderboard
        display_global_leaderboard(ep)


def main():
    parser = argparse.ArgumentParser(description="Run the Kaggriculture 100% LLM Arena with Dream-RSI.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of evolutionary epochs (default: 10)")
    parser.add_argument("--rounds", type=int, default=1, help="Swiss rounds per stage per epoch (default: 1)")
    parser.add_argument("--stages", type=str, default="", help="Comma-separated stages to run (e.g. Sprint,Expansion)")
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()] if args.stages else None
    run_dream_loop(epochs=args.epochs, rounds_per_stage=args.rounds, active_stages=stages)


if __name__ == "__main__":
    main()
