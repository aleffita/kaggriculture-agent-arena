"""Autonomously Mutated Agent: evolved_evolved_e2 (Epoch 2).

Mutated to resolve: SINGLE_QUADRANT_STALL: Farm never expanded to NE quadrant.
Synthesized via Dream-AGI evolutionary reflection loop.
"""
from __future__ import annotations

NW = sorted([(x, y) for x in range(5) for y in range(5)], key=lambda t: (abs(t[0]-4)+abs(t[1]-4), t))
NE = sorted([(x, y) for x in range(5, 10) for y in range(5)], key=lambda t: (abs(t[0]-5)+abs(t[1]-4), t))
SHED = (4, 4)

def step_toward(pos, target):
    (x, y), (tx, ty) = pos, target
    if x < tx: return ["EAST"]
    if x > tx: return ["WEST"]
    if y < ty: return ["SOUTH"]
    if y > ty: return ["NORTH"]
    return ["PASS"]

def tile_action(tile, preferred_crop, seeds, day):
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]
    if tile is None:
        if seeds.get(preferred_crop, 0) > 0:
            return ["PLANT", preferred_crop]
        for c in ["MELON", "CARROT", "WHEAT"]:
            if seeds.get(c, 0) > 0:
                return ["PLANT", c]
        return None
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        crop = tile.get("crop", "CARROT")
        crop_age = day - tile.get("planted_day", 0)
        yield_u = tile.get("yield_units", 0)
        if crop == "WHEAT" and crop_age >= 2 and yield_u > 0:
            return ["HARVEST"]
        elif crop == "CARROT" and crop_age >= 3 and yield_u > 0:
            return ["HARVEST"]
        elif crop == "MELON" and crop_age >= 10 and yield_u > 0:
            return ["HARVEST"]
        if not tile.get("watered_today", False):
            return ["WATER"]
    return None

def worker_step(pos, my_patch, me, preferred_crop, seeds, day, inventory):
    fx, fy = pos
    here = me["tiles"][fy][fx]
    if (fx, fy) in my_patch and here is not None:
        act = tile_action(here, preferred_crop, seeds, day)
        if act: return act
    for (x, y) in my_patch:
        t = me["tiles"][y][x]
        if (x, y) != (fx, fy) and t is not None:
            act = tile_action(t, preferred_crop, seeds, day)
            if act: return step_toward((fx, fy), (x, y))
    carried = sum(inventory.values())
    if carried >= 6:
        if (fx, fy) in {(4, 4), (5, 4), (4, 5), (5, 5)}:
            return ["DROP"]
        return step_toward((fx, fy), SHED)
    if (fx, fy) in my_patch and here is None:
        act = tile_action(here, preferred_crop, seeds, day)
        if act: return act
    for (x, y) in my_patch:
        if (x, y) != (fx, fy) and me["tiles"][y][x] is None:
            if any(seeds.get(c, 0) > 0 for c in ["MELON", "CARROT", "WHEAT"]):
                return step_toward((fx, fy), (x, y))
    if carried > 0:
        if (fx, fy) in {(4, 4), (5, 4), (4, 5), (5, 5)}:
            return ["DROP"]
        return step_toward((fx, fy), SHED)
    return ["PASS"]

def agent(obs: dict) -> dict:
    me = obs["farms"][obs["player"]]
    private = obs.get("private", {})
    seeds = private.get("seeds", {})
    invs = private.get("inventories") or [{}]
    hands = me.get("hands", [])
    day = obs.get("day", 0)
    hour = obs.get("hour", 0)
    money = me.get("money", 0)
    unlocked = me.get("unlocked_quadrants", ["NW"])

    max_steps = obs.get("configuration", {}).get("episodeSteps", 720)
    days_left = (max_steps - obs.get("step", 0)) // 24

    # Horizon crop schedule
    if days_left >= 11:
        preferred_crop = "MELON"
        seed_cost = 80
    elif days_left >= 3:
        preferred_crop = "CARROT"
        seed_cost = 20
    else:
        preferred_crop = "WHEAT"
        seed_cost = 10

    has_ne = "NE" in unlocked
    patch_pool = (NW + NE) if has_ne else NW
    market_orders = []

    # Market sell
    shed = private.get("shed", {})
    for item, count in shed.items():
        if count > 0 and item != "FERTILIZER":
            sell_amt = min(count, 8)
            market_orders.append(["SELL", item, sell_amt])

    # Land Expansion: unlock NE early once funds ready
    if not has_ne and day >= 4 and money >= 1050:
        market_orders.append(["BUY_LAND"])

    # Fibonacci crew expansion
    crew_target = 6 if has_ne else 3
    if len(hands) < crew_target and hour < 6 and money >= 200:
        market_orders.append(["HIRE"])

    # Replenish seeds
    total_seeds = sum(seeds.values())
    if total_seeds < 6 and money >= seed_cost * 3:
        market_orders.append(["BUY_SEED", preferred_crop, 3])

    units = 1 + len(hands)
    shares = [patch_pool[i::units] for i in range(units)]
    farmer_pos = tuple(me["farmer"])
    acts = [worker_step(farmer_pos, shares[0], me, preferred_crop, seeds, day, invs[0])]

    for i, h in enumerate(hands):
        pos = tuple(h["pos"]) if isinstance(h, dict) and "pos" in h else tuple(h)
        h_inv = invs[i + 1] if len(invs) > i + 1 else {}
        acts.append(worker_step(pos, shares[i + 1], me, preferred_crop, seeds, day, h_inv))

    return {"farmer": acts[0], "hands": acts[1:], "market": market_orders}
