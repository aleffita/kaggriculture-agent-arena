"""1v1 Match Runner for Kaggriculture Arena.

Runs head-to-head simulations between two agent strategies, tracking coins,
market dynamics, and victory margins.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pathlib
import sys
import time

# Ensure repo root and agents dir are in sys.path
repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "kaggriculture" / "agents"))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def resolve_agent_callable(name: str):
    """Resolves an agent name into a callable or standard string."""
    builtins = {"starter", "random", "pass"}
    if name.lower() in builtins:
        return name.lower()

    # Try importing from kaggriculture/agents/<name>.py
    agent_file = repo_root / "kaggriculture" / "agents" / f"{name}.py"
    if agent_file.exists():
        module = importlib.import_module(f"kaggriculture.agents.{name}")
        if hasattr(module, "agent"):
            return module.agent

    # Try direct file path
    if os.path.exists(name):
        return name

    raise ValueError(f"Could not resolve agent '{name}'. Available: starter, random, pass, or custom agents in kaggriculture/agents/.")


def run_1v1_match(agent1_name: str, agent2_name: str, steps: int = 720) -> dict:
    """Executes a 1v1 match between agent1 (Player 0) and agent2 (Player 1)."""
    from kaggle_environments import make

    console.print(f"\n[bold cyan]=== Kaggriculture 1v1 Match: {agent1_name} (P0) vs {agent2_name} (P1) ===[/bold cyan]")
    console.print(f"[dim]Episode Duration: {steps} steps ({steps // 24} in-game days)[/dim]\n")

    a1 = resolve_agent_callable(agent1_name)
    a2 = resolve_agent_callable(agent2_name)

    env = make("kaggriculture", configuration={"episodeSteps": steps}, debug=True)

    t0 = time.time()
    env.run([a1, a2])
    match_duration = time.time() - t0

    final_step = env.steps[-1]
    p0_state = final_step[0]
    p1_state = final_step[1]

    p0_reward = p0_state.reward or 0.0
    p1_reward = p1_state.reward or 0.0

    # Determine winner
    if p0_reward > p1_reward:
        winner = f"Player 0 ({agent1_name})"
        margin = p0_reward - p1_reward
    elif p1_reward > p0_reward:
        winner = f"Player 1 ({agent2_name})"
        margin = p1_reward - p0_reward
    else:
        winner = "Draw"
        margin = 0.0

    # Extract farm stats
    last_obs = p0_state.observation
    farms = last_obs.get("farms", [{}, {}]) if last_obs else [{}, {}]
    p0_money = farms[0].get("money", p0_reward)
    p1_money = farms[1].get("money", p1_reward)

    # Render summary table
    table = Table(title="1v1 Final Match Outcome")
    table.add_column("Metric", style="bold")
    table.add_column(f"Player 0: {agent1_name}", justify="right", style="cyan")
    table.add_column(f"Player 1: {agent2_name}", justify="right", style="magenta")

    table.add_row("Final Reward / Coins", f"${p0_reward:,.1f}", f"${p1_reward:,.1f}")
    table.add_row("Status", str(p0_state.status), str(p1_state.status))
    table.add_row("Execution Time", f"{match_duration:.2f}s", f"{match_duration:.2f}s")
    table.add_row("Turns Played", str(len(env.steps)), str(len(env.steps)))

    console.print(table)

    verdict_text = f"[bold green]Winner: {winner}[/bold green]" if winner != "Draw" else "[yellow]Match Drawn[/yellow]"
    if margin > 0:
        verdict_text += f" (Margin: +${margin:,.1f})"

    console.print(Panel(verdict_text, title="Result", border_style="green" if "Player 0" in winner else "magenta"))

    return {
        "agent1": agent1_name,
        "agent2": agent2_name,
        "p0_reward": p0_reward,
        "p1_reward": p1_reward,
        "winner": winner,
        "margin": margin,
        "duration_s": match_duration,
    }


def main():
    parser = argparse.ArgumentParser(description="Run a Kaggriculture 1v1 match.")
    parser.add_argument("--agent1", default="market_arbitrage", help="First agent (Player 0)")
    parser.add_argument("--agent2", default="starter", help="Second agent (Player 1)")
    parser.add_argument("--steps", type=int, default=240, help="Total episode steps (24 steps/day, default: 240 = 10 days)")
    args = parser.parse_args()

    run_1v1_match(args.agent1, args.agent2, steps=args.steps)


if __name__ == "__main__":
    main()
