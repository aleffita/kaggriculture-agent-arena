"""Dynamic Personality-Driven LLM Agent for Kaggriculture on NVIDIA GTX 1050 Ti.

Loads strategic worldview from kaggriculture/personalities/<name>.md,
injects prompt instructions, and evaluates steps on the local GTX 1050 Ti via Direct3D 12.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Callable, Dict

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from kaggriculture.agents.llm_player import get_llm_runner, parse_llm_action

PERSONALITIES_DIR = repo_root / "kaggriculture" / "personalities"
_PROMPT_CACHE: Dict[str, str] = {}


def load_personality_prompt(name: str) -> str:
    """Reads personality markdown file and extracts the core prompt."""
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]

    p_file = PERSONALITIES_DIR / f"{name}.md"
    if not p_file.exists():
        return "You are an autonomous Farmer in Kaggriculture. Maximize coins."

    content = p_file.read_text(encoding="utf-8")
    # Extract from '## Core Prompt' onwards if present
    if "## Core Prompt" in content:
        prompt_core = content.split("## Core Prompt", 1)[1].strip()
    else:
        prompt_core = content.strip()

    _PROMPT_CACHE[name] = prompt_core
    return prompt_core


def make_personality_agent(personality_name: str) -> Callable[[dict], dict]:
    """Factory creating a Kaggle agent function bound to a specific personality prompt."""
    personality_core = load_personality_prompt(personality_name)

    def agent(obs: dict) -> dict:
        runner = get_llm_runner()
        player = obs["player"]
        farm = obs["farms"][player]
        private = obs.get("private", {})
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        step = obs.get("step", 0)
        money = farm.get("money", 0)
        fx, fy = farm["farmer"]
        current_tile = farm["tiles"][fy][fx]

        # Tile state representation
        if current_tile is None:
            tile_desc = "EMPTY (ready for planting)"
        elif isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
            tile_desc = "WEED (needs DIG immediately)"
        elif isinstance(current_tile, dict) and current_tile.get("kind") == "PLANT":
            crop = current_tile.get("crop", "CARROT")
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
            f"{personality_core}\n\n"
            f"CURRENT SITUATION:\n"
            f"- Turn: Day {day}, Hour {hour} (Step {step}) | Cash: ${money:.0f}\n"
            f"- Farmer Position: ({fx},{fy}) on tile: {tile_desc}\n"
            f"- Inventory: Seeds={seeds}, Carried={carried}, Shed={shed}\n"
            f"- Market Prices: Wheat=${prices.get('WHEAT', 25)}, Carrot=${prices.get('CARROT', 35)}, Melon=${prices.get('MELON', 250)}\n"
            f"Choose ONE farmer action and optional market orders.\n"
            f"Reply ONLY with JSON: {{\"farmer\": [\"ACTION\", ...], \"market\": [[\"ORDER\", ...]]}}\n"
            f"Action JSON:"
        )

        raw = runner.generate(prompt)
        action = parse_llm_action(raw, obs)

        # Pass for hired hands if any
        hands = farm.get("hands", [])
        if hands:
            action["hands"] = [["PASS"] for _ in hands]

        return action

    return agent
