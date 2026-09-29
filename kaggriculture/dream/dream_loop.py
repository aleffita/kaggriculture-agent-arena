"""Dream-AGI Recursive Multi-Stage Arena & DuckDB Telemetry Loop.

Executes the Unified Swiss-Gated Double-Elimination Curriculum across 4 stages:
Stage 1 (Sprint: 72 steps), Stage 2 (Expansion: 144 steps),
Stage 3 (Scaling: 240 steps), Stage 4 (Full Season: 720 steps).

Each stage maintains an independent Elo rating starting at 600.0 (scaling to 3200.0).
Telemetry and match replays are continuously persisted into DuckDB.
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
from kaggriculture.dream.dream_reflector import introspect_epoch
from kaggriculture.dream.agent_mutator import mutate_agent_for_flaw

console = Console(highlight=False)


def seed_initial_roster():
    """Seeds the initial roster across all 4 stages with 600.0 baseline Elo."""
    initialize_schema()

    initial_agents = [
        ("random", "v1", "Built-in uniform random movement baseline"),
        ("starter", "v1", "Built-in single-tile carrot farmer"),
        ("wheat_looper", "v1", "Fast 2-day wheat turnaround looper"),
        ("carrot_crew", "v1", "6-carrot rotation around central shed"),
        ("crew_partitioner", "v1", "12-tile partitioned crew with Fibonacci hiring"),
        ("market_arbitrage", "v1", "BFS dynamic pricing and multi-crop arbitrage"),
        ("scale_compounder", "v1", "Day 4-5 NE expansion with 8-10 crew and melon compounding"),
        ("melon_expander", "v1", "Multi-quadrant NE+SE expansion with 12 crew and melon compounding"),
    ]

    for name, ver, desc in initial_agents:
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
    """Renders the multi-stage Elo matrix leaderboard and highlights stage champions."""
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

    insights = con.execute("""
        SELECT agent_id, flaw_diagnosed, mutation_applied, successor_agent_id
        FROM dream_insights
        WHERE epoch = ?
    """, [epoch]).fetchall()

    con.close()

    table = Table(title=f"Kaggriculture Unified Stage Matrix Leaderboard — Epoch {epoch}")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Agent ID", style="cyan")
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

        console.print(f"\n[bold yellow][*] TOP ELO STAGE CHAMPIONS (Epoch {epoch}):[/bold yellow]")
        console.print(f"  * [bold cyan]Sprint (3d)[/bold cyan]:      [bold green]{top_sprint[0]}[/bold green] (Elo: {top_sprint[1]:.1f})")
        console.print(f"  * [bold cyan]Expansion (6d)[/bold cyan]:   [bold green]{top_exp[0]}[/bold green] (Elo: {top_exp[2]:.1f})")
        console.print(f"  * [bold cyan]Scaling (10d)[/bold cyan]:    [bold green]{top_scal[0]}[/bold green] (Elo: {top_scal[3]:.1f})")
        console.print(f"  * [bold cyan]Full Season (30d)[/bold cyan]:[bold green]{top_full[0]}[/bold green] (Elo: {top_full[4]:.1f})")

    # Show Dream Insights
    if insights:
        console.print(f"\n[bold cyan][DREAM] Dream-AGI Evolutionary Mutations in Epoch {epoch}:[/bold cyan]")
        for ins in insights:
            console.print(f"  * Diagnosed [red]{ins[0]}[/red]: {ins[1]}")
            console.print(f"    -> Spawned [green]{ins[3]}[/green] in Lower Bracket: {ins[2]}")


def run_dream_loop(epochs: int = 10, rounds_per_stage: int = 2):
    """Executes the full Dream-AGI evolutionary cycle over N epochs across all stages."""
    console.print(Panel(
        "[bold cyan]Kaggriculture Unified Stage Arena & DuckDB Telemetry Broker[/bold cyan]\n"
        "[dim]DeepMind-inspired Swiss matchmaking, double-elimination lower bracket, and horizon-gated curriculum.[/dim]",
        border_style="cyan"
    ))

    seed_initial_roster()

    for ep in range(1, epochs + 1):
        console.print(f"\n[bold magenta]===========================================================================[/bold magenta]")
        console.print(f"[bold magenta]                         STARTING ARENA EPOCH {ep}/{epochs}                         [/bold magenta]")
        console.print(f"[bold magenta]===========================================================================[/bold magenta]\n")

        # 1. Inner Loop: Play Swiss rounds across all 4 stages in parallel
        for stage_id in STAGE_ORDER:
            stg = STAGES[stage_id]
            console.print(f"[yellow]Executing {stg.name} (Horizon: {stg.step_horizon} steps / {stg.in_game_days} days | Swiss Pairing)...[/yellow]")
            res = run_stage_swiss_round(epoch=ep, stage_id=stage_id, rounds=rounds_per_stage)
            console.print(f"  Completed {len(res)} Swiss duels in {stage_id}.\n")

        # 2. Outer Loop: Dream reflection over DuckDB match telemetry
        console.print(f"[cyan]Entering Dream Reflection phase for Epoch {ep}...[/cyan]")
        diagnoses = introspect_epoch(ep)
        
        if diagnoses:
            worst = diagnoses[0]
            console.print(f"  Flaw identified on [bold]{worst['agent_id']}[/bold]: {', '.join(worst['flaws'])}")
            successor_id = mutate_agent_for_flaw(worst, ep)
            console.print(f"  Synthesized successor: [bold green]{successor_id}[/bold green] (Entered at 600.0 Elo in Lower Bracket).\n")
        else:
            console.print("  No critical failure bottlenecks detected in this epoch.\n")

        # 3. Render Epoch Summary Leaderboard
        display_global_leaderboard(ep)


def main():
    parser = argparse.ArgumentParser(description="Run the Kaggriculture Unified Multi-Stage Dream-AGI loop.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of evolutionary epochs (default: 10)")
    parser.add_argument("--rounds", type=int, default=2, help="Swiss rounds per stage per epoch (default: 2)")
    args = parser.parse_args()

    run_dream_loop(epochs=args.epochs, rounds_per_stage=args.rounds)


if __name__ == "__main__":
    main()
