"""Regenerate all truncated arena replays to ensure full 720, 240, and 144-step trajectories.

Ensures every replay file in kaggriculture/data/replays/ has the full step count recorded in DuckDB,
providing complete state trajectories for training RL/GRPO models.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import pathlib
import sys
import time
from typing import Dict, Tuple

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "kaggriculture" / "agents"))

import duckdb
from kaggle_environments import make

import kaggriculture.agents.scale_compounder as a_labor
import kaggriculture.agents.wheat_looper as a_sprint
import kaggriculture.agents.crew_partitioner as a_land
import kaggriculture.agents.market_arbitrage as a_market
import kaggriculture.agents.carrot_crew as a_carrot
import kaggriculture.agents.melon_expander as a_melon

POLICY_MAP = {
    "llm_labor_magnate_v1": a_labor.agent,
    "llm_sprint_rusher_v1": a_sprint.agent,
    "llm_land_baron_v1": a_land.agent,
    "llm_market_arbitrageur_v1": a_market.agent,
    "llm_cautious_farmer_v1": a_carrot.agent,
    "llm_melon_monopolist_v1": a_melon.agent,
}

REPLAYS_DIR = repo_root / "kaggriculture" / "data" / "replays"
DB_PATH = repo_root / "kaggriculture" / "data" / "arena.duckdb"


def clean_name(agent_id: str) -> str:
    return agent_id.replace("llm_", "").replace("_v1", "").replace("_", " ").title()


def simulate_single_match(task: Tuple[str, str, int, str, str]) -> Dict:
    match_id, stage, steps, p0_id, p1_id = task
    fn0 = POLICY_MAP.get(p0_id, a_labor.agent)
    fn1 = POLICY_MAP.get(p1_id, a_sprint.agent)

    clean_p0 = clean_name(p0_id)
    clean_p1 = clean_name(p1_id)

    t0 = time.time()
    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": steps,
            "actTimeout": 999999,
            "runTimeout": 999999,
        },
        debug=False,
    )
    env.run([fn0, fn1])
    duration = time.time() - t0

    rep = env.toJSON()
    rep.setdefault("info", {})
    rep["info"]["TeamNames"] = [clean_p0, clean_p1]
    rep["info"]["Agents"] = [
        {"index": 0, "name": clean_p0},
        {"index": 1, "name": clean_p1},
    ]

    out_file = REPLAYS_DIR / f"{match_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(rep, f)

    final_step = env.steps[-1]
    p0_bank = float(final_step[0].get("reward") or 0.0)
    p1_bank = float(final_step[1].get("reward") or 0.0)

    if p0_bank == 0.0 and p1_bank == 0.0:
        obs_last = final_step[0].get("observation", {})
        farms_last = obs_last.get("farms", [{}, {}])
        if farms_last and len(farms_last) >= 2:
            p0_bank = float(farms_last[0].get("money", p0_bank))
            p1_bank = float(farms_last[1].get("money", p1_bank))

    if p0_bank > p1_bank:
        winner = p0_id
        margin = p0_bank - p1_bank
    elif p1_bank > p0_bank:
        winner = p1_id
        margin = p1_bank - p0_bank
    else:
        winner = "Draw"
        margin = 0.0

    return {
        "match_id": match_id,
        "steps_generated": len(rep.get("steps", [])),
        "target_steps": steps,
        "p0_bank": p0_bank,
        "p1_bank": p1_bank,
        "winner": winner,
        "margin": margin,
        "duration": duration,
    }


def main():
    con = duckdb.connect(str(DB_PATH))
    matches = con.execute(
        "SELECT match_id, stage, steps, agent_p0, agent_p1 FROM matches ORDER BY steps, match_id"
    ).fetchall()
    con.close()

    tasks_to_run = []
    for mid, stage, steps, p0, p1 in matches:
        f = REPLAYS_DIR / f"{mid}.json"
        needs_work = True
        if f.exists():
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                if len(data.get("steps", [])) >= steps:
                    needs_work = False
            except Exception:
                needs_work = True
        if needs_work:
            tasks_to_run.append((mid, stage, steps, p0, p1))

    print(f"Total matches in DB: {len(matches)}")
    print(f"Matches needing full generation: {len(tasks_to_run)}")

    if not tasks_to_run:
        print("All matches already have full trajectories!")
        return

    workers = min(6, os.cpu_count() or 4)
    print(f"Starting parallel generation with {workers} workers...")
    t_start = time.time()

    completed = 0
    updates = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(simulate_single_match, t): t for t in tasks_to_run}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            completed += 1
            updates.append(res)
            print(
                f"[{completed}/{len(tasks_to_run)}] {res['match_id']}: "
                f"{res['steps_generated']}/{res['target_steps']} steps "
                f"(${res['p0_bank']:.0f} vs ${res['p1_bank']:.0f}) in {res['duration']:.2f}s"
            )

    elapsed = time.time() - t_start
    print(f"\nAll {completed} matches regenerated in {elapsed:.2f}s ({elapsed/60:.2f} min)!")

    # Update database records with true full-match results
    con = duckdb.connect(str(DB_PATH))
    for u in updates:
        con.execute(
            """
            UPDATE matches
            SET p0_bank = ?, p1_bank = ?, winner = ?, margin = ?, duration_s = ?
            WHERE match_id = ?
            """,
            [u["p0_bank"], u["p1_bank"], u["winner"], u["margin"], u["duration"], u["match_id"]],
        )
    con.close()
    print("DuckDB matches table updated with full-horizon match metrics.")


if __name__ == "__main__":
    main()
