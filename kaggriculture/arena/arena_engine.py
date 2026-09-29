"""Arena Engine: Executes League Matches, Computes Elo, and Persists to DuckDB."""

from __future__ import annotations

import collections
import importlib
import itertools
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "kaggriculture" / "agents"))

from kaggle_environments import make

from kaggriculture.arena.elo import update_elo
from kaggriculture.arena.leagues import (
    LEAGUES,
    LEAGUE_ORDER,
    evaluate_promotion_demotion,
    get_league_for_elo,
)
from kaggriculture.db.schema import (
    get_connection,
    record_match,
    log_promotion,
)


def load_agent_callable(name: str):
    """Loads an agent callable from built-ins or kaggriculture/agents/."""
    builtins = {"starter", "random", "pass"}
    if name.lower() in builtins:
        return name.lower()

    agent_path = repo_root / "kaggriculture" / "agents" / f"{name}.py"
    if agent_path.exists():
        importlib.invalidate_caches()
        mod_name = f"kaggriculture.agents.{name}"
        if mod_name in sys.modules:
            mod = importlib.reload(sys.modules[mod_name])
        else:
            mod = importlib.import_module(mod_name)
        if hasattr(mod, "agent"):
            return mod.agent

    raise ValueError(f"Could not load agent: {name}")


def run_single_duel(
    agent_a_name: str,
    agent_b_name: str,
    steps: int,
) -> Tuple[float, float, str, float, List[Dict[str, Any]]]:
    """Runs a single 1v1 match between agent A (Player 0) and agent B (Player 1)."""
    func_a = load_agent_callable(agent_a_name)
    func_b = load_agent_callable(agent_b_name)

    env = make("kaggriculture", configuration={"episodeSteps": steps}, debug=True)
    t0 = time.time()
    env.run([func_a, func_b])
    duration = time.time() - t0

    final_step = env.steps[-1]
    p0_bank = float(final_step[0].reward or 0.0)
    p1_bank = float(final_step[1].reward or 0.0)

    if p0_bank > p1_bank:
        winner = agent_a_name
        margin = p0_bank - p1_bank
    elif p1_bank > p0_bank:
        winner = agent_b_name
        margin = p1_bank - p0_bank
    else:
        winner = "Draw"
        margin = 0.0

    # Sample telemetry across turns
    telemetry = []
    total_turns = len(env.steps)
    sample_points = sorted(set([0, 23, 71, 143, 239, 479, total_turns - 1]))

    for t in sample_points:
        if t < total_turns:
            step_data = env.steps[t]
            obs = step_data[0].get("observation", {})
            farms = obs.get("farms", [{}, {}])
            market = obs.get("market", {})
            prices = market.get("prices", {})

            f0 = farms[0] if len(farms) > 0 else {}
            f1 = farms[1] if len(farms) > 1 else {}

            p0_tiles = sum(
                1 for row in f0.get("tiles", []) for cell in row
                if isinstance(cell, dict) and cell.get("kind") == "PLANT"
            )
            p1_tiles = sum(
                1 for row in f1.get("tiles", []) for cell in row
                if isinstance(cell, dict) and cell.get("kind") == "PLANT"
            )

            telemetry.append({
                "turn": t,
                "day": obs.get("day", t // 24),
                "hour": obs.get("hour", t % 24),
                "p0_money": float(f0.get("money", 0.0)),
                "p1_money": float(f1.get("money", 0.0)),
                "p0_crew": len(f0.get("hands", [])),
                "p1_crew": len(f1.get("hands", [])),
                "p0_tiles_planted": p0_tiles,
                "p1_tiles_planted": p1_tiles,
                "p0_quadrants": len(f0.get("unlocked_quadrants", ["NW"])),
                "p1_quadrants": len(f1.get("unlocked_quadrants", ["NW"])),
                "market_wheat": float(prices.get("WHEAT", 25)),
                "market_carrot": float(prices.get("CARROT", 35)),
                "market_melon": float(prices.get("MELON", 250)),
            })

    return p0_bank, p1_bank, winner, margin, duration, telemetry


def run_league_season(epoch: int, league_name: str) -> List[Dict[str, Any]]:
    """Runs a competitive round-robin round for all agents currently in a specific league."""
    tier = LEAGUES[league_name]
    con = get_connection()
    agents = con.execute(
        "SELECT agent_id, name, elo, matches_played, wins FROM agents WHERE league = ?",
        [league_name],
    ).fetchall()
    con.close()

    if len(agents) < 2:
        return []

    results = []
    # Double round-robin (Home & Away)
    pairs = list(itertools.permutations(agents, 2))

    for idx, (ag_a, ag_b) in enumerate(pairs, start=1):
        id_a, name_a, elo_a, matches_a, wins_a = ag_a
        id_b, name_b, elo_b, matches_b, wins_b = ag_b

        # Execute duel
        p0_bank, p1_bank, winner_name, margin, duration, telem = run_single_duel(
            name_a, name_b, steps=tier.step_horizon
        )

        winner_id = id_a if winner_name == name_a else (id_b if winner_name == name_b else "Draw")
        score_a = 1.0 if winner_name == name_a else (0.5 if winner_name == "Draw" else 0.0)

        # Update Elo
        new_elo_a, new_elo_b = update_elo(elo_a, elo_b, score_a, matches_a, matches_b)

        match_id = f"m_{epoch}_{league_name}_{id_a}_vs_{id_b}_{idx}"
        record_match(
            match_id=match_id,
            epoch=epoch,
            league=league_name,
            steps=tier.step_horizon,
            agent_p0=id_a,
            agent_p1=id_b,
            p0_bank=p0_bank,
            p1_bank=p1_bank,
            winner=winner_id,
            margin=margin,
            p0_elo_before=elo_a,
            p0_elo_after=new_elo_a,
            p1_elo_before=elo_b,
            p1_elo_after=new_elo_b,
            duration_s=duration,
            telemetry_samples=telem,
        )

        results.append({
            "match_id": match_id,
            "p0": id_a,
            "p1": id_b,
            "p0_bank": p0_bank,
            "p1_bank": p1_bank,
            "winner": winner_id,
            "margin": margin,
            "elo_a": new_elo_a,
            "elo_b": new_elo_b,
        })

    # Evaluate Promotions and Demotions
    con = get_connection()
    updated_agents = con.execute(
        "SELECT agent_id, elo, matches_played, wins FROM agents WHERE league = ?",
        [league_name],
    ).fetchall()
    con.close()

    for ag in updated_agents:
        ag_id, elo, matches, wins = ag
        win_rate = (wins / matches) if matches > 0 else 0.0
        promo = evaluate_promotion_demotion(ag_id, league_name, elo, matches, win_rate)
        if promo:
            new_league, reason = promo
            log_promotion(ag_id, league_name, new_league, epoch, reason)

    return results
