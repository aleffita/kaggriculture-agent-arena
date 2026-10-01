"""Dynamic Personality-Driven LLM Macro-Planning Agent for Kaggriculture on NVIDIA GTX 1050 Ti.

Features:
1. Champion Macro-Planning Architectures: H48-Hybrid (wheat cycle) and H24 (daily horizon).
2. Debounce Filter: Prevents chattering on weed/harvest/opponent event triggers.
3. Dual-Engine Binding: Assigns Engine 0 for Player 0 and Engine 1 for Player 1 in GTX 1050 Ti VRAM.
4. Structured Constrained Decoding: Uses LL_GUIDANCE via LiteRtModelRunner.generate_structured.
5. Per-Player Isolated State: Zero state collision in self-play and Swiss rounds.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from kaggriculture.agents.llm_player import get_llm_runner, parse_action_string

PERSONALITIES_DIR = repo_root / "kaggriculture" / "personalities"
_PROMPT_CACHE: Dict[str, str] = {}


def clear_personality_cache():
    """Clears the in-memory prompt cache so patched personality prompts are reloaded."""
    _PROMPT_CACHE.clear()


def load_personality_prompt(name: str) -> str:
    """Reads personality markdown file and extracts the core prompt."""
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]

    clean_name = name.replace("ag-", "").replace("llm_", "").replace("-v1", "").replace("-v2", "").replace("_v1", "").replace("_v2", "")
    p_file = PERSONALITIES_DIR / f"{clean_name}.md"
    if not p_file.exists():
        p_file = PERSONALITIES_DIR / f"{name}.md"
    if not p_file.exists():
        return "You are an autonomous Farmer in Kaggriculture. Maximize coins."

    content = p_file.read_text(encoding="utf-8")
    if "## Core Prompt" in content:
        prompt_core = content.split("## Core Prompt", 1)[1].strip()
    else:
        prompt_core = content.strip()

    _PROMPT_CACHE[name] = prompt_core
    return prompt_core


def make_personality_agent(
    personality_name: str,
    horizon: Optional[int] = None,
    is_hybrid: Optional[bool] = None,
) -> Callable[[dict], dict]:
    """Factory creating an LLM macro-planning agent bound to a specific personality.
    
    Defaults:
    - cautious_farmer, market_arbitrageur -> H24 (horizon=24, is_hybrid=False)
    - sprint_rusher, land_baron, labor_magnate, melon_monopolist -> H48-Hybrid (horizon=48, is_hybrid=True)
    """
    clean_pname = personality_name.lower()
    if horizon is None:
        if "cautious" in clean_pname or "market_arbitrageur" in clean_pname or "h24" in clean_pname:
            horizon = 24
            is_hybrid = False if is_hybrid is None else is_hybrid
        else:
            horizon = 48
            is_hybrid = True if is_hybrid is None else is_hybrid
    else:
        if is_hybrid is None:
            is_hybrid = True if horizon >= 48 else False

    debounce_threshold = max(6, horizon // 4)
    personality_core = load_personality_prompt(personality_name)

    # Per-player isolated state to prevent state collision in self-play and Swiss rounds
    player_states: Dict[int, Dict[str, Any]] = {
        0: {
            "queue": collections.deque(),
            "prev_opp_money": 100.0,
            "steps_since_replan": 999,
            "last_step": -1,
            "inferences": 0,
            "replan_events": 0,
        },
        1: {
            "queue": collections.deque(),
            "prev_opp_money": 100.0,
            "steps_since_replan": 999,
            "last_step": -1,
            "inferences": 0,
            "replan_events": 0,
        },
    }

    plan_schema = {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": min(horizon, 48),
                "maxItems": min(horizon, 48),
            },
            "market_seed": {"type": "string"},
        },
        "required": ["actions"],
    }

    def agent(obs: dict) -> dict:
        player = obs["player"]
        opp_player = 1 - player
        farms = obs.get("farms", [{}, {}])
        my_farm = farms[player] if len(farms) > player else {}
        opp_farm = farms[opp_player] if len(farms) > opp_player else {}

        p_state = player_states[player]
        step = obs.get("step", 0)

        # Reset state on new match
        if step == 0 or step < p_state["last_step"]:
            p_state["queue"].clear()
            p_state["prev_opp_money"] = 100.0
            p_state["steps_since_replan"] = 999
            p_state["inferences"] = 0
            p_state["replan_events"] = 0

        p_state["last_step"] = step
        p_state["steps_since_replan"] += 1

        day = obs.get("day", step // 24)
        hour = obs.get("hour", step % 24)
        money = float(my_farm.get("money", 0.0))
        opp_money = float(opp_farm.get("money", 0.0))
        fx, fy = my_farm.get("farmer", (4, 4))
        tiles = my_farm.get("tiles", [])
        current_tile = tiles[fy][fx] if (fy < len(tiles) and fx < len(tiles[fy])) else None

        private = obs.get("private", {})
        seeds = private.get("seeds", my_farm.get("seeds", {}))
        total_seeds = sum(seeds.values())
        shed = private.get("shed", my_farm.get("shed", {}))
        carried = private.get("inventories", [{}])[0] if private.get("inventories") else {}
        market = obs.get("market", {})
        prices = market.get("prices", {})
        hands = my_farm.get("hands", [])

        # Check for event triggers in Hybrid mode
        triggered_replan = False
        replan_reason = ""
        can_trigger = (p_state["steps_since_replan"] >= debounce_threshold)

        if is_hybrid and can_trigger and len(p_state["queue"]) > 0:
            if isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
                triggered_replan = True
                replan_reason = "WEED_SPAWN"
            elif isinstance(current_tile, dict) and current_tile.get("yield_units", 0) > 0:
                triggered_replan = True
                replan_reason = "HARVEST_READY"
            elif (opp_money - p_state["prev_opp_money"]) >= 200:
                triggered_replan = True
                replan_reason = "OPPONENT_SPIKE"

        p_state["prev_opp_money"] = opp_money

        if triggered_replan:
            p_state["replan_events"] += 1
            p_state["steps_since_replan"] = 0
            p_state["queue"].clear()

        # If queue has planned actions, pop and execute immediately
        if len(p_state["queue"]) > 0:
            return p_state["queue"].popleft()

        # Re-plan required: Invoke LiteRT-LM Engine on GTX 1050 Ti
        p_state["inferences"] += 1
        p_state["steps_since_replan"] = 0
        runner = get_llm_runner(player)

        # Context tile description
        shed_tiles = {(4, 4), (5, 4), (4, 5), (5, 5)}
        is_in_shed = (fx, fy) in shed_tiles
        if is_in_shed:
            loc_str = f"SHED ({fx},{fy})"
        elif current_tile is None:
            loc_str = f"SOIL ({fx},{fy}) - Plantable"
        elif isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
            loc_str = f"WEED ({fx},{fy}) - Needs DIG"
        elif isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
            crop = current_tile.get("crop", "WHEAT")
            loc_str = f"{crop} ({fx},{fy}) (Yield: {current_tile.get('yield_units', 0)})"
        else:
            loc_str = f"({fx},{fy})"

        reason_str = f" [EVENT RE-PLAN: {replan_reason}]" if triggered_replan else ""
        prompt = (
            f"{personality_core}\n\n"
            f"STRATEGIC BRIEFING (Day {day}, Hour {hour} | Step {step}){reason_str}:\n"
            f"- Your Cash: ${money:.0f} | Opponent Cash: ${opp_money:.0f}\n"
            f"- Farmer Position: {loc_str}\n"
            f"- Seeds: {seeds} (Total: {total_seeds})\n"
            f"- Shed Inventory: {shed} | Carried: {carried}\n"
            f"- Market Prices: Wheat=${prices.get('WHEAT', 25)}, Carrot=${prices.get('CARROT', 35)}, Melon=${prices.get('MELON', 250)}\n\n"
            f"Plan next {min(horizon, 48)} actions.\n"
            f"Reply ONLY in JSON: {{\"actions\": [\"ACT_0\", ...], \"market_seed\": \"WHEAT\"}}"
        )

        try:
            raw = runner.generate_structured(prompt, plan_schema)
            parsed = json.loads(raw)
            actions_list = parsed.get("actions", ["PASS"] * horizon)
            seed_choice = parsed.get("market_seed", "WHEAT").upper()
        except Exception:
            try:
                raw = runner.generate(prompt)
                parsed = json.loads(raw)
                actions_list = parsed.get("actions", ["PASS"] * horizon)
                seed_choice = parsed.get("market_seed", "WHEAT").upper()
            except Exception:
                actions_list = ["PASS"] * horizon
                seed_choice = "WHEAT"

        # Trading safeguard
        if "melon" in clean_pname:
            seed_choice = "MELON" if money >= 80 else "WHEAT"
        elif "carrot" in clean_pname or "cautious" in clean_pname:
            seed_choice = "CARROT" if money >= 40 else "WHEAT"

        market_orders = []
        if money >= 40 and seed_choice in ("WHEAT", "CARROT", "MELON"):
            market_orders.append(["BUY_SEED", seed_choice, 6])
        elif total_seeds == 0 and money >= 40:
            market_orders.append(["BUY_SEED", "WHEAT", 6])

        for crop_name in ("WHEAT", "CARROT", "MELON"):
            qty = shed.get(crop_name, 0)
            if qty > 0:
                market_orders.append(["SELL", crop_name, qty])

        for idx, act_name in enumerate(actions_list):
            parsed_farmer = parse_action_string(act_name)
            order_for_step = market_orders if idx == 0 else []
            step_action = {"farmer": parsed_farmer, "market": order_for_step}
            if hands:
                step_action["hands"] = [["PASS"] for _ in hands]
            p_state["queue"].append(step_action)

        while len(p_state["queue"]) < 1:
            step_action = {"farmer": ["PASS"], "market": []}
            if hands:
                step_action["hands"] = [["PASS"] for _ in hands]
            p_state["queue"].append(step_action)

        return p_state["queue"].popleft()

    agent.stats = player_states
    agent.horizon = horizon
    agent.is_hybrid = is_hybrid
    return agent
