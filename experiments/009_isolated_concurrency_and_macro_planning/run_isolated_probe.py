"""Isolated Experiment: Macro-Planning Action Queuing & Multi-Game Concurrency on GTX 1050 Ti.

Compares:
1. Turn-by-Turn baseline (Bate-e-volta a cada turno: 1 LLM call per step)
2. Macro-Action Planning Queue (Horizon = 6 steps: 1 LLM call per 6 steps)
3. Concurrency scaling: 1 Game vs 3 Games vs 6 Games in parallel on the same LiteRT-LM Engine.
"""

from __future__ import annotations

import collections
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from kaggle_environments import make
from src.litert_explore.engine import LiteRtModelRunner
from kaggriculture.agents.llm_player import get_llm_runner, parse_llm_action
from kaggriculture.agents.personality_agent import load_personality_prompt


def get_gtx1050ti_vram_mb() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--id=1", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8", errors="ignore"
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return f"{parts[0]}/{parts[1]} MB (Util: {parts[2]}%)"
    except Exception:
        return "N/A"


def make_macro_agent(personality_name: str, horizon: int = 6):
    """Factory creating an LLM agent with trajectory action queuing."""
    personality_core = load_personality_prompt(personality_name)
    queue: collections.deque = collections.deque()
    stats = {"inferences": 0, "queued_steps": 0}

    def agent(obs: dict) -> dict:
        player = obs["player"]
        farm = obs["farms"][player]
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        step = obs.get("step", 0)
        money = farm.get("money", 0)
        fx, fy = farm["farmer"]
        current_tile = farm["tiles"][fy][fx]
        seeds = farm.get("seeds", {})
        total_seeds = sum(seeds.values())
        carried = farm.get("carried", {})
        carried_count = sum(carried.values())
        shed = farm.get("shed", {})
        market = obs.get("market", {})
        prices = market.get("prices", {})

        is_in_shed = (fx in (4, 5) and fy in (4, 5))
        loc_str = f"({fx},{fy}) inside Shed" if is_in_shed else f"({fx},{fy}) on Field"

        # Check if queue has a valid action
        is_weed = isinstance(current_tile, dict) and current_tile.get("kind") == "WEED"
        if not is_weed and len(queue) > 0:
            stats["queued_steps"] += 1
            action = queue.popleft()
            return action

        # Queue empty or emergency (weed): trigger structured macro-plan inference
        stats["inferences"] += 1
        runner = get_llm_runner()

        action_rules = []
        if is_weed:
            action_rules.append("EMERGENCY: Weed on current tile! First action MUST be ['DIG'].")
        if total_seeds == 0:
            action_rules.append("NO SEEDS: Must add market order [['BUY_SEED', 'WHEAT', 6]].")
        if is_in_shed and carried_count > 0:
            action_rules.append("In shed with crops: farmer action must be ['DROP'].")

        shed_produce = {k: v for k, v in shed.items() if k in ("WHEAT", "CARROT", "MELON") and v > 0}
        for k, v in shed_produce.items():
            action_rules.append(f"Produce in shed: add market order [['SELL', '{k}', {v}]].")

        rules_str = "\n".join(f"- {r}" for r in action_rules)

        prompt = (
            f"{personality_core}\n\n"
            f"CURRENT SITUATION:\n"
            f"- Turn: Day {day}, Hour {hour} (Step {step}) | Cash: ${money:.0f}\n"
            f"- Farmer Position: {loc_str}\n"
            f"- Seeds: {seeds} (Total: {total_seeds})\n"
            f"- Carried Crops: {carried} (Total: {carried_count})\n"
            f"- Shed Inventory: {shed}\n"
            f"- Market Prices: Wheat=${prices.get('WHEAT', 25)}, Carrot=${prices.get('CARROT', 35)}, Melon=${prices.get('MELON', 250)}\n\n"
            f"TACTICAL DIRECTIVES:\n"
            f"{rules_str}\n\n"
            f"Plan the next sequence of up to {horizon} farmer actions to execute your strategy.\n"
            f"Reply ONLY with JSON: {{\"plan\": [[\"ACTION\", ...], ...], \"market\": [[\"ORDER\", ...]]}}\n"
            f"Plan JSON:"
        )

        plan_schema = {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "minItems": 1,
                    "maxItems": horizon,
                },
                "market": {"type": "array", "items": {"type": "array"}},
            },
            "required": ["plan"],
        }

        try:
            raw = runner.generate_structured(prompt, plan_schema)
            parsed = json.loads(raw)
            plan = parsed.get("plan", [["PASS"]])
            market_orders = parsed.get("market", [])
        except Exception:
            plan = [["PASS"]]
            market_orders = []

        # Trading safeguard
        if total_seeds == 0 and money >= 40 and not any(isinstance(o, list) and o and o[0] == "BUY_SEED" for o in market_orders):
            market_orders.append(["BUY_SEED", "WHEAT", 6])
        if shed_produce and not any(isinstance(o, list) and o and o[0] == "SELL" for o in market_orders):
            for crop_name, qty in shed_produce.items():
                market_orders.append(["SELL", crop_name, qty])

        # Enqueue remaining actions
        queue.clear()
        for idx, act in enumerate(plan):
            clean_act = act if isinstance(act, list) and act else ["PASS"]
            entry = {"farmer": clean_act, "market": market_orders if idx == 0 else []}
            queue.append(entry)

        return queue.popleft()

    agent.stats = stats
    return agent


def run_single_isolated_game(
    p0_name: str,
    p1_name: str,
    steps: int = 72,
    mode: str = "macro",  # "macro" or "turn_by_turn"
    horizon: int = 6,
) -> Dict[str, Any]:
    """Runs a single isolated game on GTX 1050 Ti and collects telemetry."""
    if mode == "macro":
        ag0 = make_macro_agent(p0_name, horizon=horizon)
        ag1 = make_macro_agent(p1_name, horizon=horizon)
    else:
        # Turn by turn (horizon=1)
        ag0 = make_macro_agent(p0_name, horizon=1)
        ag1 = make_macro_agent(p1_name, horizon=1)

    env = make("kaggriculture", configuration={"episodeSteps": steps, "actTimeout": 60, "runTimeout": 3600})

    t0 = time.time()
    env.run([ag0, ag1])
    duration = time.time() - t0

    inf_p0 = ag0.stats["inferences"]
    inf_p1 = ag1.stats["inferences"]
    total_inf = inf_p0 + inf_p1

    final_step = env.steps[-1]
    obs_last = final_step[0].get("observation", {})
    farms_last = obs_last.get("farms", [{}, {}])
    p0_money = float(farms_last[0].get("money", 0.0))
    p1_money = float(farms_last[1].get("money", 0.0))

    return {
        "p0": p0_name,
        "p1": p1_name,
        "steps": steps,
        "mode": mode,
        "horizon": horizon,
        "duration_s": round(duration, 2),
        "total_inferences": total_inf,
        "inf_p0": inf_p0,
        "inf_p1": inf_p1,
        "p0_money": p0_money,
        "p1_money": p1_money,
        "steps_per_sec": round(steps / duration, 3),
        "inf_per_sec": round(total_inf / duration, 3),
    }


def run_parallel_games(
    game_pairs: List[Tuple[str, str]],
    steps: int = 72,
    mode: str = "macro",
    horizon: int = 6,
) -> Dict[str, Any]:
    """Runs N games simultaneously in parallel threads on the GTX 1050 Ti."""
    vram_start = get_gtx1050ti_vram_mb()
    t0 = time.time()

    def _worker(pair):
        p0, p1 = pair
        return run_single_isolated_game(p0, p1, steps=steps, mode=mode, horizon=horizon)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(game_pairs)) as pool:
        results = list(pool.map(_worker, game_pairs))

    duration = time.time() - t0
    vram_end = get_gtx1050ti_vram_mb()

    total_steps = sum(r["steps"] for r in results)
    total_inferences = sum(r["total_inferences"] for r in results)

    return {
        "num_games": len(game_pairs),
        "mode": mode,
        "horizon": horizon,
        "total_wall_time_s": round(duration, 2),
        "total_steps": total_steps,
        "total_inferences": total_inferences,
        "steps_per_sec": round(total_steps / duration, 3),
        "inferences_per_sec": round(total_inferences / duration, 3),
        "vram_start": vram_start,
        "vram_end": vram_end,
        "games": results,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("ISOLATED ABLATION: MACRO-PLANNING QUEUE & PARALLEL GAMES ON GTX 1050 Ti")
    print("=" * 70)

    # Prewarm engine
    print(f"[0] Prewarming LiteRT-LM on GTX 1050 Ti... ({get_gtx1050ti_vram_mb()})")
    get_llm_runner()
    print(f"    Ready! VRAM: {get_gtx1050ti_vram_mb()}\n")

    # TEST 1: Macro-Planning (Horizon=6) on 1 single game of 72 steps
    print("[1] Executing 1 Isolated Game (72 steps, Macro Horizon=6)...")
    res_macro_1 = run_single_isolated_game("sprint_rusher", "cautious_farmer", steps=72, mode="macro", horizon=6)
    print(f"    Completed in {res_macro_1['duration_s']}s ({res_macro_1['duration_s']/60:.2f} min)!")
    print(f"    Inferences: {res_macro_1['total_inferences']} (P0: {res_macro_1['inf_p0']}, P1: {res_macro_1['inf_p1']}) | Throughput: {res_macro_1['steps_per_sec']} steps/s")
    print(f"    Scores: Sprint=${res_macro_1['p0_money']:.0f} vs Cautious=${res_macro_1['p1_money']:.0f}\n")

    # TEST 2: 3 Games in Parallel (Macro Horizon=6, 72 steps each = 216 steps total)
    print("[2] Executing 3 Games Simultaneously in Parallel (72 steps each, Macro Horizon=6)...")
    pairs_3 = [
        ("sprint_rusher", "cautious_farmer"),
        ("labor_magnate", "land_baron"),
        ("melon_monopolist", "market_arbitrageur"),
    ]
    res_parallel_3 = run_parallel_games(pairs_3, steps=72, mode="macro", horizon=6)
    print(f"    Completed in {res_parallel_3['total_wall_time_s']}s ({res_parallel_3['total_wall_time_s']/60:.2f} min)!")
    print(f"    Total Steps: {res_parallel_3['total_steps']} | Total Inferences: {res_parallel_3['total_inferences']}")
    print(f"    System Throughput: {res_parallel_3['steps_per_sec']} steps/s | VRAM: {res_parallel_3['vram_end']}\n")

    # Save results to JSON
    out_dir = pathlib.Path(__file__).parent
    out_file = out_dir / "results.json"
    summary = {
        "single_macro_game": res_macro_1,
        "parallel_3_games": res_parallel_3,
    }
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Results persisted to {out_file}")
