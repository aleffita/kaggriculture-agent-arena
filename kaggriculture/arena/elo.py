"""Elo Rating System tailored for Kaggriculture 1v1 Arena.

Starts at 600.0 baseline Elo with dynamic K-factor based on match maturity.
"""

from __future__ import annotations

import math
from typing import Tuple

DEFAULT_BASE_ELO = 600.0


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """Computes expected win probability for player A using logistic curve."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def get_k_factor(matches_played: int) -> float:
    """Dynamic K-factor: rapid calibration early, stabilizing over time."""
    if matches_played < 10:
        return 40.0
    elif matches_played < 30:
        return 24.0
    return 16.0


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    matches_a: int,
    matches_b: int,
) -> Tuple[float, float]:
    """Updates Elo ratings for two players following match resolution.
    
    score_a: 1.0 for win A, 0.5 for draw, 0.0 for loss A.
    Returns: (new_rating_a, new_rating_b).
    """
    exp_a = calculate_expected_score(rating_a, rating_b)
    exp_b = 1.0 - exp_a
    score_b = 1.0 - score_a

    k_a = get_k_factor(matches_a)
    k_b = get_k_factor(matches_b)

    new_rating_a = rating_a + k_a * (score_a - exp_a)
    new_rating_b = rating_b + k_b * (score_b - exp_b)

    # Floor at 100 Elo
    return max(100.0, round(new_rating_a, 1)), max(100.0, round(new_rating_b, 1))
