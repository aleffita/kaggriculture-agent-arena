"""Multi-Agent Tournament & Round-Robin League for Kaggriculture Arena.

Runs all-play-all home/away series, neutralizes seat bias, and computes
leaderboard rankings based on total coins and net margins.
"""

from __future__ import annotations

import argparse
import itertools
import pathlib
import sys
import time

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "kaggriculture" / "agents"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from kaggriculture.arena.match_runner import run_1v1_match

console = Console()


def run_tournament(agent_names: list[str], steps_per_match: int = 720) -> dict:
    """Executes a full double round-robin tournament across all listed agents."""
    console.print(f"\n[bold green]=== Starting Kaggriculture Double Round-Robin Tournament ===[/bold green]")
    console.print(f"Competitors: {', '.join(agent_names)} | Match Length: {steps_per_match} steps\n")

    stats = {
        name: {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "coins_for": 0.0,
            "coins_against": 0.0,
            "matches_played": 0,
        }
        for name in agent_names
    }

    pairs = list(itertools.permutations(agent_names, 2))
    console.print(f"Total fixtures to play: {len(pairs)} matches\n")

    for idx, (p0, p1) in enumerate(pairs, start=1):
        console.print(f"[bold yellow]Fixture #{idx}/{len(pairs)}: {p0} (Home) vs {p1} (Away)[/bold yellow]")
        res = run_1v1_match(p0, p1, steps=steps_per_match)

        p0_coins = res["p0_reward"]
        p1_coins = res["p1_reward"]

        stats[p0]["coins_for"] += p0_coins
        stats[p0]["coins_against"] += p1_coins
        stats[p0]["matches_played"] += 1

        stats[p1]["coins_for"] += p1_coins
        stats[p1]["coins_against"] += p0_coins
        stats[p1]["matches_played"] += 1

        if p0_coins > p1_coins:
            stats[p0]["wins"] += 1
            stats[p1]["losses"] += 1
        elif p1_coins > p0_coins:
            stats[p1]["wins"] += 1
            stats[p0]["losses"] += 1
        else:
            stats[p0]["draws"] += 1
            stats[p1]["draws"] += 1

    # Leaderboard Table
    table = Table(title="Kaggriculture Tournament Final Standings")
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Agent", style="cyan")
    table.add_column("W - D - L", justify="center")
    table.add_column("Win Rate", justify="right")
    table.add_column("Total Coins", justify="right", style="green")
    table.add_column("Coins / Match", justify="right")
    table.add_column("Net Margin", justify="right", style="magenta")

    # Sort by wins, then net margin, then total coins
    ranked = sorted(
        stats.items(),
        key=lambda item: (item[1]["wins"], item[1]["coins_for"] - item[1]["coins_against"], item[1]["coins_for"]),
        reverse=True,
    )

    for rank, (name, s) in enumerate(ranked, start=1):
        matches = s["matches_played"]
        win_rate = (s["wins"] / matches * 100) if matches > 0 else 0.0
        net_margin = s["coins_for"] - s["coins_against"]
        avg_coins = s["coins_for"] / matches if matches > 0 else 0.0

        table.add_row(
            str(rank),
            name,
            f"{s['wins']}-{s['draws']}-{s['losses']}",
            f"{win_rate:.1f}%",
            f"${s['coins_for']:,.1f}",
            f"${avg_coins:,.1f}",
            f"+${net_margin:,.1f}" if net_margin >= 0 else f"-${abs(net_margin):,.1f}",
        )

    console.print("\n", table)
    champion = ranked[0][0]
    console.print(Panel(f"[bold green]Tournament Champion: {champion}[/bold green]", border_style="gold1"))

    return {"standings": ranked, "stats": stats}


def main():
    parser = argparse.ArgumentParser(description="Run a Kaggriculture tournament.")
    parser.add_argument("--steps", type=int, default=720, help="Steps per match (default 720 = 30 days)")
    args = parser.parse_args()

    agents = ["random", "starter", "wheat_looper", "market_arbitrage"]
    run_tournament(agents, steps_per_match=args.steps)


if __name__ == "__main__":
    main()
