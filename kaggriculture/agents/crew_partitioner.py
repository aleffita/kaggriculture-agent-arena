"""Crew Partitioner Agent for Kaggriculture (PDF Section 8 Implementation).

Farms 12 tiles in the NW quadrant with a crew of 3 hired hands, where every worker
strictly owns a non-overlapping slice of tiles (preventing job collision and idle loops).
"""

from __future__ import annotations

CARROT = "CARROT"
MAX_YIELD_DAY = 3
CREW_TARGET = 3
SHED = (4, 4)

# 12 nearest tiles to the shed in NW quadrant
PATCH = sorted(
    [(x, y) for x in range(5) for y in range(5)],
    key=lambda t: (abs(t[0] - 4) + abs(t[1] - 4), t)
)[:12]


def step_toward(pos, target):
    (x, y), (tx, ty) = pos, target
    if x < tx: return ["EAST"]
    if x > tx: return ["WEST"]
    if y < ty: return ["SOUTH"]
    if y > ty: return ["NORTH"]
    return ["PASS"]


def tile_needs(tile, seeds, day):
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        return ["DIG"]
    if tile is None:
        return ["PLANT", CARROT] if seeds.get(CARROT, 0) > 0 else None
    if isinstance(tile, dict) and tile.get("kind") == "PLANT":
        if not tile.get("watered_today"):
            return ["WATER"]
        if day - tile["planted_day"] >= MAX_YIELD_DAY and tile.get("yield_units", 0) > 0:
            return ["HARVEST"]
    return None


def unit_act(pos, my_tiles, me, seeds, day, held):
    fx, fy = pos
    here = me["tiles"][fy][fx]

    # If on my assigned tile and it needs action
    if (fx, fy) in my_tiles and here is not None:
        act = tile_needs(here, seeds, day)
        if act:
            return act

    # Check other assigned tiles needing care
    for (x, y) in my_tiles:
        t = me["tiles"][y][x]
        if (x, y) != (fx, fy) and t is not None and tile_needs(t, seeds, day):
            return step_toward((fx, fy), (x, y))

    # Deliver to shed if carrying harvest
    if held.get(CARROT, 0) >= 6:
        if (fx, fy) == SHED:
            return ["DROP"]
        return step_toward((fx, fy), SHED)

    # Planting on assigned tiles
    if (fx, fy) in my_tiles and here is None and seeds.get(CARROT, 0) > 0:
        return ["PLANT", CARROT]

    for (x, y) in my_tiles:
        if (x, y) != (fx, fy) and me["tiles"][y][x] is None and seeds.get(CARROT, 0) > 0:
            return step_toward((fx, fy), (x, y))

    if held.get(CARROT, 0) > 0:
        if (fx, fy) == SHED:
            return ["DROP"]
        return step_toward((fx, fy), SHED)

    return ["PASS"]


def agent(obs: dict) -> dict:
    me = obs["farms"][obs["player"]]
    private = obs.get("private", {})
    seeds = private.get("seeds", {})
    invs = private.get("inventories") or [{}]
    hands = me.get("hands", [])
    units = 1 + len(hands)

    market = []
    shed_stock = private.get("shed", {}).get(CARROT, 0)
    if shed_stock > 0:
        market.append(["SELL", CARROT, shed_stock])

    # Seed replenishment
    empty = sum(1 for (x, y) in PATCH if me["tiles"][y][x] is None)
    need = empty - seeds.get(CARROT, 0)
    if need > 0 and me["money"] >= 20 * need:
        market.append(["BUY_SEED", CARROT, need])

    # Hire hands in the morning (one hire per turn to manage Fibonacci cost)
    if len(hands) < CREW_TARGET and me["money"] >= 300:
        market.append(["HIRE"])

    # Disjoint partitioned shares of tiles for each unit
    shares = [PATCH[i::units] for i in range(units)]

    farmer_pos = me["farmer"]
    acts = [unit_act(farmer_pos, shares[0], me, seeds, obs["day"], invs[0])]

    for i, h in enumerate(hands):
        pos = tuple(h["pos"]) if isinstance(h, dict) and "pos" in h else tuple(h)
        h_inv = invs[i + 1] if len(invs) > i + 1 else {}
        acts.append(unit_act(pos, shares[i + 1], me, seeds, obs["day"], h_inv))

    return {"farmer": acts[0], "hands": acts[1:], "market": market}
