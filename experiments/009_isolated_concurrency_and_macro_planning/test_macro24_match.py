"""Isolated Match Test: Macro-Planning 24-Action Daily Horizon on GTX 1050 Ti.

Simulates 1 full 72-step (3 days) match between 2 LLM agents using:
- 1 call per day (Day 0, Day 1, Day 2) = 3 calls per agent = 6 calls total!
- Measures exact match duration, inference count, VRAM, and game rewards.
"""

from __future__ import annotations

import collections
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
from kaggriculture.agents.llm_player import get_llm_runner
from kaggriculture.agents.personality_agent import load_personality_prompt


def get_vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--id=1", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8", errors="ignore"
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return f"{parts[0]}/{parts[1]} MB (Util: {parts[2]}%)"
    except Exception:
        return "N/A"


def parse_action_string(act_str: str) -> List[str]:
    act = act_str.strip().upper()
    if act.startswith("PLANT_"):
        crop = act.replace("PLANT_", "")
        return ["PLANT", crop]
    if act in ("NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "DIG", "DROP", "PASS"):
        return [act]
    if "PLANT" in act:
        return ["PLANT", "WHEAT"]
    return ["PASS"]


def make_macro24_agent(personality_name: str):
    """Creates a macro-planning agent that plans 24 actions at the start of each in-game day."""
    personality_core = load_personality_prompt(personality_name)
    queue: collections.deque = collections.deque()
    stats = {"inferences": 0, "inference_latencies": []}

    schema = {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 24,
                "maxItems": 24,
            },
            "market_seed": {"type": "string"},
        },
        "required": ["actions"],
    }

    def agent(obs: dict) -> dict:
        player = obs["player"]
        opp_player = 1 - player
        my_farm = obs["farms"][player]
        opp_farm = obs["farms"][opp_player]
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        step = obs.get("step", 0)
        money = my_farm.get("money", 0)
        opp_money = opp_farm.get("money", 0)
        fx, fy = my_farm["farmer"]
        seeds = my_farm.get("seeds", {})
        total_seeds = sum(seeds.values())
        shed = my_farm.get("shed", {})
        market = obs.get("market", {})
        prices = market.get("prices", {})

        # If we have actions in the daily queue, pop and execute
        if len(queue) > 0:
            return queue.popleft()

        # New Day (or empty queue): Generate full 24-hour daily plan
        stats["inferences"] += 1
        t_start = time.time()
        runner = get_llm_runner()

        # Personality tactical crop choice
        if "melon" in personality_name.lower():
            pref_crop = "MELON"
        elif "carrot" in personality_name.lower() or "cautious" in personality_name.lower():
            pref_crop = "CARROT"
        else:
            pref_crop = "WHEAT"

        prompt = (
            f"{personality_core}\n\n"
            f"DAILY STRATEGIC BRIEFING (Day {day}, Hour {hour} | Step {step}):\n"
            f"- Your Cash: ${money:.0f} | Opponent Cash: ${opp_money:.0f}\n"
            f"- Farmer Location: ({fx},{fy})\n"
            f"- Seeds in Inventory: {seeds} (Total: {total_seeds})\n"
            f"- Shed Inventory: {shed}\n"
            f"- Market Prices Today: Wheat=${prices.get('WHEAT', 25)}, Carrot=${prices.get('CARROT', 35)}, Melon=${prices.get('MELON', 250)}\n\n"
            f"DIRECTIVES FOR TODAY (24 Hours):\n"
            f"1. Plan 24 sequential farmer actions for the full 24 hours of Day {day}.\n"
            f"   Valid actions: NORTH, SOUTH, EAST, WEST, PLANT_{pref_crop}, WATER, HARVEST, DIG, DROP, PASS.\n"
            f"2. Indicate daily seed purchase in 'market_seed': 'WHEAT', 'CARROT', 'MELON', or 'NONE'.\n\n"
            f"Reply ONLY in JSON format: {{\"actions\": [\"ACT_0\", ... \"ACT_23\"], \"market_seed\": \"{pref_crop}\"}}"
        )

        try:
            raw = runner.generate_structured(prompt, schema)
            dur = time.time() - t_start
            stats["inference_latencies"].append(dur)
            parsed = json.loads(raw)
            actions_list = parsed.get("actions", [])
            seed_choice = parsed.get("market_seed", pref_crop).upper()
        except Exception as e:
            dur = time.time() - t_start
            stats["inference_latencies"].append(dur)
            actions_list = ["PASS"] * 24
            seed_choice = pref_crop

        market_orders = []
        # Daily seed buy
        if money >= 40 and seed_choice in ("WHEAT", "CARROT", "MELON"):
            market_orders.append(["BUY_SEED", seed_choice, 6])
        # Daily shed sell
        for crop_name in ("WHEAT", "CARROT", "MELON"):
            qty = shed.get(crop_name, 0)
            if qty > 0:
                market_orders.append(["SELL", crop_name, qty])

        # Enqueue the 24 actions
        for idx, act_name in enumerate(actions_list[:24]):
            parsed_farmer = parse_action_string(act_name)
            # Market orders execute on hour 0
            order_for_step = market_orders if idx == 0 else []
            queue.append({"farmer": parsed_farmer, "market": order_for_step})

        # Pad if less than 24
        while len(queue) < 24:
            queue.append({"farmer": ["PASS"], "market": []})

        return queue.popleft()

    agent.stats = stats
    return agent


def run_test():
    print("=" * 80)
    print("ISOLATED MATCH: 72 STEPS (3 DAYS) WITH MACRO-24 PLANNING ON GTX 1050 Ti")
    print("=" * 80)

    print(f"[0] Pre-warming Engine on GPU 1 (GTX 1050 Ti)... Initial VRAM: {get_vram()}")
    runner = get_llm_runner()
    print(f"    Engine Loaded! VRAM: {get_vram()}\n")

    p0_name = "sprint_rusher"
    p1_name = "cautious_farmer"

    ag0 = make_macro24_agent(p0_name)
    ag1 = make_macro24_agent(p1_name)

    env = make("kaggriculture", configuration={"episodeSteps": 72, "actTimeout": 60, "runTimeout": 3600})

    print(f"[1] Starting Match: {p0_name} vs {p1_name} (72 steps / 3 in-game days)...")
    t0 = time.time()
    env.run([ag0, ag1])
    total_match_time = time.time() - t0

    inf_p0 = ag0.stats["inferences"]
    inf_p1 = ag1.stats["inferences"]
    lat_p0 = ag0.stats["inference_latencies"]
    lat_p1 = ag1.stats["inference_latencies"]

    final_step = env.steps[-1]
    obs_last = final_step[0].get("observation", {})
    farms_last = obs_last.get("farms", [{}, {}])
    p0_bank = float(farms_last[0].get("money", 0.0))
    p1_bank = float(farms_last[1].get("money", 0.0))

    winner = p0_name if p0_bank > p1_bank else (p1_name if p1_bank > p0_bank else "Draw")

    print("\n" + "=" * 80)
    print("MATCH RESULTS & EMPIRICAL METRICS")
    print("=" * 80)
    print(f"  * Total Match Wall Time : {total_match_time:.2f}s ({total_match_time/60:.2f} MINUTES!)")
    print(f"  * Total Inferences       : {inf_p0 + inf_p1} (P0: {inf_p0}, P1: {inf_p1})")
    print(f"  * P0 Latencies per Day   : {[round(l, 2) for l in lat_p0]} s")
    print(f"  * P1 Latencies per Day   : {[round(l, 2) for l in lat_p1]} s")
    print(f"  * Final Scores           : {p0_name}=${p0_bank:.0f} vs {p1_name}=${p1_bank:.0f}")
    print(f"  * Winner                 : {winner}")
    print(f"  * Final VRAM             : {get_vram()}")
    print("=" * 80)

    # Comparison against turn-by-turn baseline (757.6s)
    speedup = 757.6 / total_match_time
    print(f"  --> SPEEDUP vs Turn-by-Turn (12.63 min): {speedup:.1f}x FASTER!")


if __name__ == "__main__":
    run_test()
