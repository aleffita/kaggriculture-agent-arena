"""Deterministic Wheat Looper Agent for Kaggriculture.

Focuses on high-turnover Wheat production in the unlocked NW quadrant.
"""

from __future__ import annotations

def agent(obs: dict) -> dict:
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    fx, fy = me["farmer"]
    tiles = me["tiles"]
    tile = tiles[fy][fx]

    market_orders = []

    # 1. Market Operations: Buy seed if running out and we have funds
    wheat_seeds = private["seeds"].get("WHEAT", 0)
    if wheat_seeds < 3 and me["money"] >= 30:
        market_orders.append(["BUY_SEED", "WHEAT", 2])

    # 2. Market Operations: Sell any harvested wheat sitting in the shed
    wheat_in_shed = private["shed"].get("WHEAT", 0)
    if wheat_in_shed > 0:
        market_orders.append(["SELL", "WHEAT", wheat_in_shed])

    # 3. Tile Operations
    # If standing on a plant
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop_age = obs["day"] - tile["planted_day"]
        # Wheat peaks around day 2-4; harvest if ripe
        if crop_age >= 2 and tile.get("yield_units", 0) > 0:
            return {"farmer": ["HARVEST"], "hands": [], "market": market_orders}
        # Water if unwatered today
        if not tile.get("watered_today", False):
            return {"farmer": ["WATER"], "hands": [], "market": market_orders}

    # If standing on empty unlocked tile, plant if we have seeds
    if tile is None and wheat_seeds > 0:
        return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": market_orders}

    # If standing on weed, dig it out
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return {"farmer": ["DIG"], "hands": [], "market": market_orders}

    # If carrying inventory and adjacent to center shed, drop it
    farmer_inv = private["inventories"][0] if private.get("inventories") else {}
    if sum(farmer_inv.values()) > 0 and (fx in (4, 5) and fy in (4, 5)):
        return {"farmer": ["DROP"], "hands": [], "market": market_orders}

    # 4. Movement: Simple patrol within the starting 5x5 NW quadrant (x: 0..4, y: 0..4)
    # Simple snake/spiral patrol
    if fy % 2 == 0:
        if fx < 4:
            move = "EAST"
        elif fy < 4:
            move = "SOUTH"
        else:
            move = "WEST"
    else:
        if fx > 0:
            move = "WEST"
        elif fy < 4:
            move = "SOUTH"
        else:
            move = "EAST"

    return {"farmer": [move], "hands": [], "market": market_orders}
