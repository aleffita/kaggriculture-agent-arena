"""LiteRT-LM Step-by-Step Autonomous LLM Agent running on NVIDIA GTX 1050 Ti.

Executes direct neural inference on the local GPU via Direct3D 12 and the DXGI hook.
Evaluates the real-time farm observation at every step and generates structured action JSON.
"""

from __future__ import annotations

import json
import os
import re
import sys
import pathlib
from typing import Any, Dict, List, Optional, Tuple

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.litert_explore.engine import LiteRtModelRunner

_GLOBAL_RUNNER: Optional[LiteRtModelRunner] = None


def get_llm_runner() -> LiteRtModelRunner:
    """Singleton getter to keep model weights loaded in GTX 1050 Ti VRAM."""
    global _GLOBAL_RUNNER
    if _GLOBAL_RUNNER is None:
        # Target GPU 1 (GTX 1050 Ti) via DXGI vtable interceptor
        _GLOBAL_RUNNER = LiteRtModelRunner(
            backend="gpu",
            gpu_target="1050ti",
            max_num_tokens=512,
        )
    return _GLOBAL_RUNNER


def format_observation_for_llm(obs: dict) -> str:
    """Transforms raw Kaggle environment observation into a compact, token-dense prompt."""
    player = obs["player"]
    farm = obs["farms"][player]
    private = obs.get("private", {})
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    step = obs.get("step", 0)
    money = farm.get("money", 0)
    farmer_pos = tuple(farm["farmer"])
    fx, fy = farmer_pos
    current_tile = farm["tiles"][fy][fx]

    # Tile description
    if current_tile is None:
        tile_desc = "EMPTY (ready for PLANT)"
    elif isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
        tile_desc = "WEED (needs DIG)"
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
    prices = obs.get("market", {}).get("prices", {})

    prompt = (
        f"You are the Farmer in Kaggriculture. Decide actions for this turn.\n"
        f"STATUS: Day {day}, Hour {hour} (Step {step}) | Cash: ${money:.0f} | Pos: ({fx},{fy}) on {tile_desc}\n"
        f"INVENTORY: Seeds={seeds} | Carried={carried} | Shed={shed}\n"
        f"MARKET PRICES: Wheat=${prices.get('WHEAT', 25)}, Carrot=${prices.get('CARROT', 35)}, Melon=${prices.get('MELON', 250)}\n"
        f"SHED TILES: (4,4), (5,4), (4,5), (5,5). HIRE costs $200+, NE land costs $1000.\n"
        f"Choose ONE farmer action and optional market orders.\n"
        f"Reply ONLY with a JSON object in this exact format:\n"
        f'{{"farmer": ["ACTION", ...], "market": [["ORDER", ...]]}}\n'
        f"Valid farmer actions: ['PLANT', 'WHEAT'], ['PLANT', 'CARROT'], ['WATER'], ['HARVEST'], ['DIG'], ['DROP'], ['NORTH'], ['SOUTH'], ['EAST'], ['WEST'], ['PASS'].\n"
        f"Valid market orders: ['BUY_SEED', 'WHEAT', 3], ['SELL', 'WHEAT', 3], ['HIRE'], ['BUY_LAND'].\n"
        f"Action JSON:"
    )
    return prompt


def parse_llm_action(raw_text: str, obs: dict) -> dict:
    """Parses JSON action emitted by LiteRT LLM with robust fallback repair."""
    action_dict = {"farmer": ["PASS"], "market": []}
    
    # Try regex JSON extraction
    json_match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, dict):
                if "farmer" in parsed and isinstance(parsed["farmer"], list) and parsed["farmer"]:
                    action_dict["farmer"] = parsed["farmer"]
                if "market" in parsed and isinstance(parsed["market"], list):
                    # Validate market orders format
                    clean_market = []
                    for order in parsed["market"]:
                        if isinstance(order, list) and order:
                            clean_market.append(order)
                    action_dict["market"] = clean_market[:10]
                return action_dict
        except Exception:
            pass

    # Fallback heuristic parsing if JSON was truncated or malformed
    clean = raw_text.upper()
    if "HARVEST" in clean:
        action_dict["farmer"] = ["HARVEST"]
    elif "WATER" in clean:
        action_dict["farmer"] = ["WATER"]
    elif "PLANT" in clean:
        if "CARROT" in clean:
            action_dict["farmer"] = ["PLANT", "CARROT"]
        else:
            action_dict["farmer"] = ["PLANT", "WHEAT"]
    elif "DIG" in clean:
        action_dict["farmer"] = ["DIG"]
    elif "NORTH" in clean:
        action_dict["farmer"] = ["NORTH"]
    elif "SOUTH" in clean:
        action_dict["farmer"] = ["SOUTH"]
    elif "EAST" in clean:
        action_dict["farmer"] = ["EAST"]
    elif "WEST" in clean:
        action_dict["farmer"] = ["WEST"]

    return action_dict


def agent(obs: dict) -> dict:
    """Standard Kaggle Environments agent callable.
    
    Invoked every single turn by the engine. Runs inference on the GTX 1050 Ti.
    """
    runner = get_llm_runner()
    prompt = format_observation_for_llm(obs)
    
    # Neural inference on GTX 1050 Ti via D3D12
    raw_response = runner.generate(prompt)
    action = parse_llm_action(raw_response, obs)
    
    # Handle hands if any exist (repeat pass or worker step)
    player = obs["player"]
    me = obs["farms"][player]
    hands = me.get("hands", [])
    if hands:
        action["hands"] = [["PASS"] for _ in hands]

    return action
