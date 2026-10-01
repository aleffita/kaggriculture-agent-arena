"""Arena Engine: Executes League Matches, Computes Elo, and Persists to DuckDB."""

from __future__ import annotations

import collections
import concurrent.futures
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
from kaggriculture.arena.stages import STAGES, STAGE_ORDER
from kaggriculture.arena.matchmaking import swiss_pairings, split_upper_lower_brackets
from kaggriculture.arena.leagues import (
    LEAGUES,
    LEAGUE_ORDER,
    evaluate_promotion_demotion,
    get_league_for_elo,
)
from kaggriculture.db.schema import (
    get_connection,
    record_match,
    record_stage_match,
    log_promotion,
)


import threading

_LOAD_LOCK = threading.Lock()
_AGENT_CACHE: Dict[str, Any] = {}


def load_agent_callable(name: str):
    """Loads an agent callable from built-ins or kaggriculture/agents/ thread-safely."""
    builtins = {"starter", "random", "pass"}
    if name.lower() in builtins:
        return name.lower()

    with _LOAD_LOCK:
        if name in _AGENT_CACHE:
            return _AGENT_CACHE[name]

        base_name = name
        agent_path = repo_root / "kaggriculture" / "agents" / f"{base_name}.py"
        if not agent_path.exists():
            for sfx in ["_v1", "_v2", "_v3"]:
                if base_name.endswith(sfx):
                    candidate = base_name[:-len(sfx)]
                    if (repo_root / "kaggriculture" / "agents" / f"{candidate}.py").exists():
                        base_name = candidate
                        agent_path = repo_root / "kaggriculture" / "agents" / f"{base_name}.py"
                        break

        if agent_path.exists():
            mod_name = f"kaggriculture.agents.{base_name}"
            if mod_name in sys.modules:
                mod = sys.modules[mod_name]
            else:
                mod = importlib.import_module(mod_name)
            if hasattr(mod, "agent"):
                _AGENT_CACHE[name] = mod.agent
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

    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": steps,
            "actTimeout": 60,
            "runTimeout": 3600,
        },
        debug=True,
    )
    t0 = time.time()
    env.run([func_a, func_b])
    duration = time.time() - t0

    final_step = env.steps[-1]
    p0_bank = float(final_step[0].reward if final_step[0].reward is not None else 0.0)
    p1_bank = float(final_step[1].reward if final_step[1].reward is not None else 0.0)
    if p0_bank == 0.0 and p1_bank == 0.0:
        obs_last = final_step[0].get("observation", {})
        farms_last = obs_last.get("farms", [{}, {}])
        if farms_last and len(farms_last) >= 2:
            p0_bank = float(farms_last[0].get("money", p0_bank))
            p1_bank = float(farms_last[1].get("money", p1_bank))

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


def run_stage_swiss_round(
    epoch: int,
    stage_id: str,
    rounds: int = 2,
) -> List[Dict[str, Any]]:
    """Runs a multi-round Swiss tournament for a horizon stage across Upper and Lower brackets."""
    stage = STAGES[stage_id]
    con = get_connection()
    
    rows = con.execute("""
        SELECT r.agent_id, a.name, r.elo, r.matches_played, r.wins, r.bracket
        FROM agent_stage_ratings r
        JOIN agents a ON r.agent_id = a.agent_id
        WHERE r.stage = ?
    """, [stage_id]).fetchall()
    con.close()

    if len(rows) < 2:
        return []

    agent_records = [
        {
            "agent_id": r[0],
            "name": r[1],
            "elo": float(r[2]),
            "matches": int(r[3]),
            "wins": int(r[4]),
            "bracket": r[5],
        }
        for r in rows
    ]

    results = []
    played_pairs: set[Tuple[str, str]] = set()

    # For rosters <= 6 agents, keep unified Swiss pool to pair all agents concurrently
    if len(agent_records) <= 6:
        brackets = [("Swiss", agent_records)]
    else:
        upper_bracket, lower_bracket = split_upper_lower_brackets(agent_records)
        brackets = [("Upper", upper_bracket)]
        if lower_bracket:
            brackets.append(("Lower", lower_bracket))

    match_counter = 1
    for bracket_name, pool in brackets:
        if len(pool) < 2:
            continue

        for r_num in range(1, rounds + 1):
            pairs = swiss_pairings(pool, played_pairs)
            for ag_a, ag_b in pairs:
                played_pairs.add((ag_a["agent_id"], ag_b["agent_id"]))

            def _get_vram():
                try:
                    import subprocess
                    out = subprocess.check_output(
                        ["nvidia-smi", "--id=1", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                        encoding="utf-8", errors="ignore"
                    ).strip()
                    parts = [p.strip() for p in out.split(",")]
                    return f"{parts[0]}/{parts[1]} MB (Util: {parts[2]}%)"
                except Exception:
                    return "N/A"

            vram_before = _get_vram()
            print(f"  [Round {r_num}] Executing {len(pairs)} duels simultaneously in parallel on GTX 1050 Ti ({stage.step_horizon} steps) | VRAM: {vram_before}...", flush=True)

            def _run_pair(pair_info):
                idx, ag_a, ag_b = pair_info
                p0_id, name_a = ag_a["agent_id"], ag_a["name"]
                p1_id, name_b = ag_b["agent_id"], ag_b["name"]
                p0_bank, p1_bank, winner_name, margin, duration, telem = run_single_duel(
                    name_a, name_b, steps=stage.step_horizon
                )
                return idx, ag_a, ag_b, p0_bank, p1_bank, winner_name, margin, duration, telem

            indexed_pairs = [(match_counter + i, ag_a, ag_b) for i, (ag_a, ag_b) in enumerate(pairs)]
            match_counter += len(pairs)

            t_round_start = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(2, len(pairs)))) as pool_exec:
                duel_outcomes = list(pool_exec.map(_run_pair, indexed_pairs))
            t_round_duration = time.time() - t_round_start
            vram_after = _get_vram()

            total_steps = len(pairs) * stage.step_horizon
            print(f"  [Round {r_num} Finished] {len(pairs)} duels completed in {t_round_duration:.2f}s ({t_round_duration/60:.2f} min) | VRAM: {vram_after} | Throughput: {total_steps/t_round_duration:.2f} steps/s.", flush=True)

            for m_idx, ag_a, ag_b, p0_bank, p1_bank, winner_name, margin, duration, telem in duel_outcomes:
                p0_id, name_a, elo_a, m_a = ag_a["agent_id"], ag_a["name"], ag_a["elo"], ag_a["matches"]
                p1_id, name_b, elo_b, m_b = ag_b["agent_id"], ag_b["name"], ag_b["elo"], ag_b["matches"]

                winner_id = p0_id if winner_name == name_a else (p1_id if winner_name == name_b else "Draw")
                score_a = 1.0 if winner_name == name_a else (0.5 if winner_name == "Draw" else 0.0)

                new_elo_a, new_elo_b = update_elo(elo_a, elo_b, score_a, m_a, m_b)

                print(f"    -> [Duel {m_idx}] Winner: {winner_name} (${p0_bank:.0f} vs ${p1_bank:.0f} in {duration:.1f}s) | Elo: {name_a}={new_elo_a:.0f}, {name_b}={new_elo_b:.0f}", flush=True)

                ag_a["elo"] = new_elo_a
                ag_a["matches"] += 1
                ag_b["elo"] = new_elo_b
                ag_b["matches"] += 1

                mid = f"m_{epoch}_{stage_id}_{bracket_name}_{p0_id}_vs_{p1_id}_{m_idx}"

                record_stage_match(
                    match_id=mid,
                    epoch=epoch,
                    stage=stage_id,
                    steps=stage.step_horizon,
                    bracket=bracket_name,
                    agent_p0=p0_id,
                    agent_p1=p1_id,
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
                    "match_id": mid,
                    "stage": stage_id,
                    "bracket": bracket_name,
                    "p0": p0_id,
                    "p1": p1_id,
                    "p0_bank": p0_bank,
                    "p1_bank": p1_bank,
                    "winner": winner_id,
                    "margin": margin,
                    "elo_a": new_elo_a,
                    "elo_b": new_elo_b,
                })

    return results
