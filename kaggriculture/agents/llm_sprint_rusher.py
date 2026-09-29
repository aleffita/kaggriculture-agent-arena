"""LiteRT-LM Sprint Rusher: Fast-Cycle LLM Policy on NVIDIA GTX 1050 Ti.

Specialized prompt strategy for Stage 1 (Sprint: 72 steps / 3 days):
- Prioritizes rapid 2-day wheat turnaround and weed elimination.
- Executes direct neural inference on the local GTX 1050 Ti via Direct3D 12.
"""

from __future__ import annotations

import json
import re
import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from kaggriculture.agents.llm_player import get_llm_runner, parse_llm_action


def format_sprint_prompt(obs: dict) -> str:
    player = obs["player"]
    farm = obs["farms"][player]
    private = obs.get("private", {})
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    step = obs.get("step", 0)
    money = farm.get("money", 0)
    fx, fy = farm["farmer"]
    current_tile = farm["tiles"][fy][fx]

    if current_tile is None:
        tile_desc = "EMPTY (ready to PLANT WHEAT)"
    elif isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
        tile_desc = "WEED (CRITICAL: MUST DIG NOW)"
    elif isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
        crop = current_tile.get("crop", "UNKNOWN")
        age = day - current_tile.get("planted_day", 0)
        watered = current_tile.get("watered_today", False)
        yield_u = current_tile.get("yield_units", 0)
        tile_desc = f"{crop} (Age: {age}d, WateredToday: {watered}, Yield: {yield_u})"
    else:
        tile_desc = str(current_tile)

    seeds = private.get("seeds", {})
    carried = private.get("inventories", [{}])[0] if private.get("inventories") else {}
    shed = private.get("shed", {})

    prompt = (
        f"You are the SPRINT CHAMPION in Kaggriculture. Objective: Maximize coins by Day 3.\n"
        f"TURN: Day {day}, Hour {hour} (Step {step}) | Bank: ${money:.0f} | Pos: ({fx},{fy}) on {tile_desc}\n"
        f"SEEDS: {seeds} | CARRIED: {carried} | SHED: {shed}\n"
        f"RULE 1: If on WEED, DIG immediately. If mature crop (Wheat age>=2), HARVEST immediately.\n"
        f"RULE 2: If on EMPTY tile and have Wheat seeds, PLANT WHEAT. If plant is unwatered today, WATER.\n"
        f"RULE 3: If carried items > 0 and adjacent to shed (4,4), DROP items into shed.\n"
        f"RULE 4: If shed has wheat, SELL wheat in market.\n"
        f"Reply ONLY with JSON: {{\"farmer\": [\"ACTION\", ...], \"market\": [[\"ORDER\", ...]]}}\n"
        f"Action JSON:"
    )
    return prompt


def agent(obs: dict) -> dict:
    runner = get_llm_runner()
    prompt = format_sprint_prompt(obs)
    raw = runner.generate(prompt)
    action = parse_llm_action(raw, obs)
    hands = obs["farms"][obs["player"]].get("hands", [])
    if hands:
        action["hands"] = [["PASS"] for _ in hands]
    return action
