"""Market Arbitrage & Dynamic Cropping Agent for Kaggriculture.

Evaluates market demand, navigates greedily to unwatered or harvestable crops,
and diversifies high-margin crops (Melon / Carrot / Wheat) based on real-time prices.
"""

from __future__ import annotations

import collections
from typing import List, Tuple, Optional


def find_nearest_target(start: Tuple[int, int], targets: set[Tuple[int, int]], grid_size: int = 10) -> Optional[str]:
    """BFS to find the next movement direction towards the closest target coordinate."""
    if not targets or start in targets:
        return None

    queue = collections.deque([(start[0], start[1], [])])
    visited = {start}

    directions = [
        ("NORTH", (0, -1)),
        ("SOUTH", (0, 1)),
        ("EAST", (1, 0)),
        ("WEST", (-1, 0)),
    ]

    while queue:
        cx, cy, path = queue.popleft()
        if (cx, cy) in targets:
            return path[0] if path else None

        for move_name, (dx, dy) in directions:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, path + [move_name]))

    return None


def agent(obs: dict) -> dict:
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    fx, fy = me["farmer"]
    tiles = me["tiles"]
    tile = tiles[fy][fx]
    day = obs["day"]

    market_orders = []
    prices = obs["market"].get("prices", {})

    # 1. Market Operations: Sell any harvested produce sitting in shed
    shed = private.get("shed", {})
    for item, count in shed.items():
        if count > 0 and item not in ("FERTILIZER",):
            # Sell up to 5 units per turn to avoid crashing price
            sell_amt = min(count, 5)
            market_orders.append(["SELL", item, sell_amt])

    # 2. Crop Selection & Seed Purchasing
    # Check margins: Melon (seed 80, price 250), Carrot (seed 20, price 35), Wheat (seed 10, price 25)
    seeds = private.get("seeds", {})
    money = me.get("money", 0)

    # Buy seeds dynamically based on remaining episode horizon
    max_steps = obs.get("configuration", {}).get("episodeSteps", 720)
    steps_left = max_steps - obs.get("step", 0)
    days_left = steps_left // 24

    total_seeds = sum(seeds.values())
    if total_seeds < 4 and money >= 30:
        if days_left >= 11 and prices.get("MELON", 250) >= 180 and money >= 200:
            market_orders.append(["BUY_SEED", "MELON", 1])
        elif days_left >= 4 and prices.get("CARROT", 35) >= 28 and money >= 100:
            market_orders.append(["BUY_SEED", "CARROT", 2])
        elif days_left >= 2 and money >= 30:
            market_orders.append(["BUY_SEED", "WHEAT", 2])


    # 3. Action on Current Tile
    # If plant is on tile
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = tile.get("crop", "WHEAT")
        crop_age = day - tile.get("planted_day", 0)
        yield_units = tile.get("yield_units", 0)

        # Harvest condition
        is_harvestable = False
        if crop == "WHEAT" and crop_age >= 2 and yield_units >= 2:
            is_harvestable = True
        elif crop == "CARROT" and crop_age >= 2 and yield_units >= 2:
            is_harvestable = True
        elif crop == "MELON" and crop_age >= 8 and yield_units >= 4:
            is_harvestable = True
        elif crop_age >= 10 and yield_units > 0:
            is_harvestable = True

        if is_harvestable:
            return {"farmer": ["HARVEST"], "hands": [], "market": market_orders}

        # Water condition
        if not tile.get("watered_today", False):
            return {"farmer": ["WATER"], "hands": [], "market": market_orders}

    # If weed on tile, clear it
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return {"farmer": ["DIG"], "hands": [], "market": market_orders}

    # If empty unlocked tile and we have seeds, plant highest value seed fitting the horizon
    if tile is None:
        plant_order = []
        if days_left >= 11:
            plant_order.append("MELON")
        if days_left >= 4:
            plant_order.append("CARROT")
        plant_order.append("WHEAT")
        for crop in plant_order:
            if seeds.get(crop, 0) > 0:
                return {"farmer": ["PLANT", crop], "hands": [], "market": market_orders}


    # If carrying items in farmer inventory, check shed drop
    farmer_inv = private["inventories"][0] if private.get("inventories") else {}
    carried_items = sum(farmer_inv.values())
    shed_adjacent_tiles = {(4, 4), (5, 4), (4, 5), (5, 5)}

    if carried_items >= 4:
        if (fx, fy) in shed_adjacent_tiles:
            return {"farmer": ["DROP"], "hands": [], "market": market_orders}
        # Navigate towards shed
        step_dir = find_nearest_target((fx, fy), shed_adjacent_tiles)
        if step_dir:
            return {"farmer": [step_dir], "hands": [], "market": market_orders}

    # 4. Target Search & Pathfinding
    # Collect coordinates needing action in NW quadrant (x in 0..4, y in 0..4)
    needing_water = set()
    needing_harvest = set()
    empty_plantable = set()

    for y in range(5):
        for x in range(5):
            t = tiles[y][x]
            if isinstance(t, dict) and t.get("kind") == "PLANT":
                if not t.get("watered_today", False):
                    needing_water.add((x, y))
                elif t.get("yield_units", 0) >= 2:
                    needing_harvest.add((x, y))
            elif t is None and sum(seeds.values()) > 0:
                empty_plantable.add((x, y))

    # Priority 1: Water unwatered plants (prevents decay into weeds)
    if needing_water:
        move_dir = find_nearest_target((fx, fy), needing_water)
        if move_dir:
            return {"farmer": [move_dir], "hands": [], "market": market_orders}

    # Priority 2: Harvest ripe crops
    if needing_harvest:
        move_dir = find_nearest_target((fx, fy), needing_harvest)
        if move_dir:
            return {"farmer": [move_dir], "hands": [], "market": market_orders}

    # Priority 3: Plant empty tiles
    if empty_plantable:
        move_dir = find_nearest_target((fx, fy), empty_plantable)
        if move_dir:
            return {"farmer": [move_dir], "hands": [], "market": market_orders}

    # Default fallback movement (center patrol)
    return {"farmer": ["PASS"], "hands": [], "market": market_orders}
