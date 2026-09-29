"""League Hierarchy and Elo-Gated Curriculum for Kaggriculture Arena."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class LeagueTier:
    name: str
    division: int
    step_horizon: int
    in_game_days: int
    min_elo: float
    promotion_elo: Optional[float]
    demotion_elo: Optional[float]
    description: str


LEAGUES: dict[str, LeagueTier] = {
    "Wood": LeagueTier(
        name="Wood",
        division=4,
        step_horizon=72,
        in_game_days=3,
        min_elo=600.0,
        promotion_elo=850.0,
        demotion_elo=None,
        description="Novice 3-day sprint: basic survival, cash-crop wheat/carrot turnover, zero weeds.",
    ),
    "Bronze": LeagueTier(
        name="Bronze",
        division=3,
        step_horizon=144,
        in_game_days=6,
        min_elo=850.0,
        promotion_elo=1200.0,
        demotion_elo=800.0,
        description="6-day early scale: modest crew hiring and Day 6 shop lottery observation.",
    ),
    "Silver": LeagueTier(
        name="Silver",
        division=2,
        step_horizon=240,
        in_game_days=10,
        min_elo=1200.0,
        promotion_elo=1600.0,
        demotion_elo=1150.0,
        description="10-day mid-game: first land purchase (NE quadrant), dedicated worker partitioning.",
    ),
    "Gold": LeagueTier(
        name="Gold",
        division=1,
        step_horizon=720,
        in_game_days=30,
        min_elo=1600.0,
        promotion_elo=None,
        demotion_elo=1550.0,
        description="Premier 30-day season: full 4-quadrant expansion, 12-15 crew, livestock compounding.",
    ),
}

LEAGUE_ORDER = ["Wood", "Bronze", "Silver", "Gold"]


def get_league_for_elo(elo: float) -> str:
    """Returns the default league assignment based on Elo rating."""
    if elo >= 1600.0:
        return "Gold"
    elif elo >= 1200.0:
        return "Silver"
    elif elo >= 850.0:
        return "Bronze"
    return "Wood"


def evaluate_promotion_demotion(
    agent_id: str,
    current_league: str,
    elo: float,
    matches_in_league: int,
    win_rate: float,
) -> Optional[Tuple[str, str]]:
    """Evaluates if an agent qualifies for promotion or demotion.
    
    Returns: (new_league, reason) or None.
    """
    tier = LEAGUES.get(current_league)
    if not tier:
        return None

    # Check Promotion
    if tier.promotion_elo is not None and elo >= tier.promotion_elo:
        if matches_in_league >= 3 and win_rate >= 0.50:
            curr_idx = LEAGUE_ORDER.index(current_league)
            next_league = LEAGUE_ORDER[curr_idx + 1]
            return next_league, f"Elo {elo:.1f} >= {tier.promotion_elo} with {win_rate:.0%} win rate."

    # Check Demotion
    if tier.demotion_elo is not None and elo < tier.demotion_elo:
        if matches_in_league >= 3:
            curr_idx = LEAGUE_ORDER.index(current_league)
            prev_league = LEAGUE_ORDER[curr_idx - 1]
            return prev_league, f"Elo {elo:.1f} < {tier.demotion_elo} demotion threshold."

    return None
