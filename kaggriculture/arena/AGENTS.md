# AGENTS.md: Kaggriculture 1v1 Arena Directives (`kaggriculture/arena/`)

This directory hosts the 1v1 evaluation engine, head-to-head match runner, and tournament analytics for Kaggriculture.

---

## 1. Arena Protocols

- **Match Execution**: Full season is 720 turns (30 days × 24 turns/day). Short testing matches can be run with `--steps 72` or `--steps 240`.
- **Fair Play**: In head-to-head evaluations, always execute two-leg ties where Player 0 and Player 1 roles are swapped to neutralize first-mover or quadrant initialization asymmetries.
- **Reporting**: Match results must report final coins, reward delta, peak market price realized, and total crops harvested.

---

## 2. Tools & Scripts

- `match_runner.py`: Runs a 1v1 match between any two agents (e.g. `market_arbitrage` vs `starter`, or `wheat_looper` vs `random`).
- `tournament.py`: Multi-agent round-robin league and Elo tracker.
