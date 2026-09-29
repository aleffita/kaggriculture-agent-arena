"""Horizon Stage Definitions for Kaggriculture Multi-Stage Arena.

Temporal stages decouple step horizons from skill/Elo ratings.
Each stage maintains its own independent Elo ladder starting at 600.0 baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class HorizonStage:
    id: str
    name: str
    step_horizon: int
    in_game_days: int
    baseline_elo: float = 600.0
    description: str = ""


STAGES: Dict[str, HorizonStage] = {
    "Sprint": HorizonStage(
        id="Sprint",
        name="Stage 1: Sprint",
        step_horizon=72,
        in_game_days=3,
        baseline_elo=600.0,
        description="Fast cash-crop turnover (Wheat/Carrot), zero weed penalties, immediate liquidity.",
    ),
    "Expansion": HorizonStage(
        id="Expansion",
        name="Stage 2: Expansion",
        step_horizon=144,
        in_game_days=6,
        baseline_elo=600.0,
        description="Early labor scaling (1–3 hands), Day 4 NE land gate ($1050), seed buffer management.",
    ),
    "Scaling": HorizonStage(
        id="Scaling",
        name="Stage 3: Scaling",
        step_horizon=240,
        in_game_days=10,
        baseline_elo=600.0,
        description="Disjoint spatial worker partitioning, first complete Melon harvest, market price damping.",
    ),
    "FullSeason": HorizonStage(
        id="FullSeason",
        name="Stage 4: Full Season",
        step_horizon=720,
        in_game_days=30,
        baseline_elo=600.0,
        description="Macroeconomic compounding, 8–12 workers, multi-quadrant rotation, end-game cash conversion.",
    ),
}

STAGE_ORDER: List[str] = ["Sprint", "Expansion", "Scaling", "FullSeason"]
