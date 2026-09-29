# AGENTS.md: Kaggriculture Agent Strategies (`kaggriculture/agents/`)

This directory contains standalone agents for 1v1 competition in the Kaggriculture environment.

---

## 1. Agent Specification Contract

Every agent module in this directory must export an `agent(obs)` callable that receives the Kaggle environment observation dictionary and returns the standard action dictionary:

```python
def agent(obs: dict) -> dict:
    return {
        "farmer": [op, *args],        # e.g. ["PLANT", "WHEAT"] or ["MOVE", "NORTH"]
        "hands":  [[op, *args], ...], # one action per hired hand
        "market": [[op, *args], ...], # up to 10 market orders per turn
    }
```

---

## 2. Invariants & Rules

1. **Submissions Compatibility**: Agents must be self-contained or easily bundleable for Kaggle competition submissions (`main.py`).
2. **Turn Time Limit**: Agents must compute decisions within the per-turn timeout (typically 1.0 second per step in Kaggle environments).
3. **Valid Operations**:
   - Movement: `NORTH`, `SOUTH`, `EAST`, `WEST`, `PASS`.
   - Shed/Center: `PICKUP <item> [n]`, `PLACE <item> [n]`, `DROP`. (Valid when adjacent to center shed at `(4,4)`, `(5,4)`, `(4,5)`, `(5,5)`).
   - Crops: `PLANT <crop>`, `WATER`, `HARVEST`, `FERTILIZE`.
   - Animals: `BUILD_COOP`, `BUILD_PASTURE`, `FEED`, `COLLECT_FERTILIZER`, `CARE`.
   - Terrain: `DIG`.
   - Market: `BUY_SEED <crop> <n>`, `BUY_PRODUCT <item> <n>`, `BUY_ANIMAL <animal> <n>`, `SELL <item> <n>`, `HIRE`, `BUY_LAND`.

---

## 3. Agent Catalog

- `wheat_looper.py`: Deterministic fast wheat cycling baseline.
- `market_arbitrage.py`: Dynamic pricing and yield optimization agent with pathfinding and crop rotation.
- `llm_strategist.py`: Hybrid strategist using local Gemma 4 on GTX 1050 Ti for daily macro planning.
