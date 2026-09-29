"""Dream-AGI Recursive Self-Improvement Loop for Kaggriculture Arena.

Coordinates multi-league curriculum matches, updates DuckDB telemetry,
reflects on game pathologies, and synthesizes successor agents over successive epochs.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import io

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
    get_connection,
)
from kaggriculture.arena.leagues import LEAGUES, LEAGUE_ORDER
from kaggriculture.arena.arena_engine import run_league_season
from kaggriculture.dream.dream_reflector import introspect_epoch
from kaggriculture.dream.agent_mutator import mutate_agent_for_flaw

console = Console(highlight=False)


def seed_initial_roster():
    """Seeds the initial league hierarchy with foundational strategic archetypes."""
    initialize_schema()

    initial_agents = [
        ("random", "v1", "Wood", 600.0, "Built-in uniform random movement baseline"),
        ("starter", "v1", "Wood", 750.0, "Built-in single-tile carrot farmer"),
        ("wheat_looper", "v1", "Bronze", 900.0, "Fast 2-day wheat turnaround looper"),
        ("carrot_crew", "v1", "Bronze", 1050.0, "6-carrot rotation around central shed"),
        ("crew_partitioner", "v1", "Silver", 1300.0, "12-tile partitioned crew with Fibonacci hiring"),
        ("market_arbitrage", "v1", "Silver", 1250.0, "BFS dynamic pricing and multi-crop arbitrage"),
        ("scale_compounder", "v1", "Gold", 1650.0, "Day 5-6 NE land expansion with 8-10 crew and melon compounding"),
        ("melon_expander", "v1", "Gold", 1600.0, "Multi-quadrant NE+SE expansion with 12 crew and melon compounding"),
    ]

    for name, ver, league, elo, desc in initial_agents:
        register_agent(
            name=name,
            version=ver,
            league=league,
            initial_elo=elo,
            description=desc,
            code_path=f"kaggriculture/agents/{name}.py",
        )


def display_global_leaderboard(epoch: int):
    """Renders the current DuckDB Elo standings and league distribution."""
    con = get_connection(read_only=True)
    rows = con.execute("""
        SELECT 
            agent_id, name, version, league, elo, matches_played, 
            wins, losses, draws, total_coins
        FROM agents
        ORDER BY elo DESC, total_coins DESC
    """).fetchall()

    promotions = con.execute("""
        SELECT agent_id, from_league, to_league, reason
        FROM league_promotions
        WHERE epoch = ?
    """, [epoch]).fetchall()

    insights = con.execute("""
        SELECT agent_id, flaw_diagnosed, mutation_applied, successor_agent_id
        FROM dream_insights
        WHERE epoch = ?
    """, [epoch]).fetchall()

    con.close()

    table = Table(title=f"Kaggriculture Arena Leaderboard — Epoch {epoch}")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Agent ID", style="cyan")
    table.add_column("League", justify="center", style="bold")
    table.add_column("Elo Rating", justify="right", style="green")
    table.add_column("W - D - L", justify="center")
    table.add_column("Matches", justify="right")
    table.add_column("Total Coins", justify="right", style="magenta")

    for rank, r in enumerate(rows, start=1):
        ag_id, name, ver, league, elo, matches, w, l, d, coins = r
        league_color = {
            "Gold": "[bold gold1]Gold[/bold gold1]",
            "Silver": "[bold grey78]Silver[/bold grey78]",
            "Bronze": "[bold dark_orange]Bronze[/bold dark_orange]",
            "Wood": "[dim]Wood[/dim]",
        }.get(league, league)

        table.add_row(
            str(rank),
            ag_id,
            league_color,
            f"{elo:.1f}",
            f"{w}-{d}-{l}",
            str(matches),
            f"${coins:,.0f}",
        )

    console.print("\n", table)

    # Show promotions
    if promotions:
        console.print(f"\n[bold yellow][!] League Transitions in Epoch {epoch}:[/bold yellow]")
        for p in promotions:
            console.print(f"  * [green]PROMOTION[/green]: [bold]{p[0]}[/bold] ({p[1]} -> [bold]{p[2]}[/bold]): {p[3]}")

    # Show Dream Insights
    if insights:
        console.print(f"\n[bold cyan][DREAM] Dream-AGI Evolutionary Mutations in Epoch {epoch}:[/bold cyan]")
        for ins in insights:
            console.print(f"  * Diagnosed [red]{ins[0]}[/red]: {ins[1]}")
            console.print(f"    -> Spawned [green]{ins[3]}[/green]: {ins[2]}")


def run_dream_loop(epochs: int = 3):
    """Executes the full Dream-AGI evolutionary cycle over N epochs."""
    console.print(Panel(
        "[bold cyan]Kaggriculture Dream-AGI Recursive Arena & DuckDB Telemetry Loop[/bold cyan]\n"
        "[dim]Gradient-free evolutionary self-improvement via 1v1 duels, Elo curriculum, and DuckDB replay reflection.[/dim]",
        border_style="cyan"
    ))

    seed_initial_roster()

    for ep in range(1, epochs + 1):
        console.print(f"\n[bold magenta]===========================================================================[/bold magenta]")
        console.print(f"[bold magenta]                         STARTING ARENA EPOCH {ep}/{epochs}                         [/bold magenta]")
        console.print(f"[bold magenta]===========================================================================[/bold magenta]\n")

        # 1. Inner Loop: Play matches in each league tier
        for league in LEAGUE_ORDER:
            tier = LEAGUES[league]
            console.print(f"[yellow]Executing Division {tier.division} ({league} League | Horizon: {tier.step_horizon} steps / {tier.in_game_days} days)...[/yellow]")
            res = run_league_season(epoch=ep, league_name=league)
            console.print(f"  Completed {len(res)} duels in {league} League.\n")

        # 2. Outer Loop: Dream reflection over DuckDB match telemetry
        console.print(f"[cyan]Entering Dream Reflection phase for Epoch {ep}...[/cyan]")
        diagnoses = introspect_epoch(ep)
        
        if diagnoses:
            # Mutate the most critical underperforming agent
            worst = diagnoses[0]
            console.print(f"  Flaw identified on [bold]{worst['agent_id']}[/bold]: {', '.join(worst['flaws'])}")
            successor_id = mutate_agent_for_flaw(worst, ep)
            console.print(f"  Synthesized successor: [bold green]{successor_id}[/bold green] (Entered at 600.0 Elo in Wood League).\n")
        else:
            console.print("  No critical failure bottlenecks detected in this epoch.\n")

        # 3. Render Epoch Summary Leaderboard
        display_global_leaderboard(ep)


def main():
    parser = argparse.ArgumentParser(description="Run the Kaggriculture Dream-AGI evolutionary loop.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of evolutionary epochs (default: 2)")
    args = parser.parse_args()

    run_dream_loop(epochs=args.epochs)


if __name__ == "__main__":
    main()
