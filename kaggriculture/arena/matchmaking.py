"""Matchmaking & Tournament Pairing Engine for Kaggriculture Arena.

Implements:
1. Swiss Matchmaking: Pairs agents with nearest Elo ratings to maximize information gain.
2. Double Elimination Bracket Routing: Segregates Upper (Winners) and Lower (Losers/Redemption) brackets.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple


def swiss_pairings(
    agents: List[Dict[str, Any]],
    played_pairs: Optional[Set[Tuple[str, str]]] = None,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Generates Swiss pairings by sorting agents by Elo and pairing adjacent peers.
    
    agents: list of dicts with keys 'agent_id', 'name', 'elo'
    played_pairs: set of (agent_id_a, agent_id_b) tuples to avoid duplicate match repeats if possible.
    """
    if len(agents) < 2:
        return []

    # Sort descending by current stage Elo (with tiny noise to break exact ties)
    sorted_agents = sorted(
        agents,
        key=lambda a: (a.get("elo", 600.0), random.random()),
        reverse=True,
    )

    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    unpaired = list(sorted_agents)
    played = played_pairs or set()

    while len(unpaired) >= 2:
        p0 = unpaired.pop(0)
        p0_id = p0["agent_id"]

        # Find best peer: closest Elo not yet played, or fall back to adjacent
        best_idx = 0
        found_fresh = False
        for idx, candidate in enumerate(unpaired):
            p1_id = candidate["agent_id"]
            if (p0_id, p1_id) not in played and (p1_id, p0_id) not in played:
                best_idx = idx
                found_fresh = True
                break

        p1 = unpaired.pop(best_idx)
        pairs.append((p0, p1))

    return pairs


def split_upper_lower_brackets(
    agents: List[Dict[str, Any]],
    median_elo_threshold: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits an active stage roster into Upper (Winners) and Lower (Losers/Redemption) brackets."""
    if len(agents) < 4:
        # Too small to split into separate brackets; return all in upper
        return agents, []

    # Calculate median Elo
    elos = sorted([a.get("elo", 600.0) for a in agents])
    med = median_elo_threshold if median_elo_threshold is not None else elos[len(elos) // 2]

    upper = [a for a in agents if a.get("elo", 600.0) >= med]
    lower = [a for a in agents if a.get("elo", 600.0) < med]

    # Ensure at least 2 agents per bracket if possible
    if len(upper) < 2 and len(lower) >= 2:
        upper.append(lower.pop(0))
    elif len(lower) < 2 and len(upper) >= 2:
        lower.append(upper.pop(-1))

    return upper, lower
