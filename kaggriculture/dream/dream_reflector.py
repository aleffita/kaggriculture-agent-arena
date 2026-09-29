"""Dream Reflector: Introspects DuckDB Match Traces and Telemetry for Strategic Flaws."""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, List, Optional

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from kaggriculture.db.schema import get_connection


def introspect_epoch(epoch: int) -> List[Dict[str, Any]]:
    """Analyzes DuckDB match history and telemetry from an epoch to diagnose strategic bottlenecks."""
    con = get_connection(read_only=True)

    # 1. Query lowest-performing agents in the latest matches
    loss_stats = con.execute("""
        SELECT 
            agent_p0 AS agent, 
            COUNT(*) as games,
            AVG(p0_bank) as avg_bank,
            AVG(margin) as avg_margin
        FROM matches 
        WHERE epoch = ? AND winner != agent_p0 AND winner != 'Draw'
        GROUP BY agent_p0
        ORDER BY avg_bank ASC
    """, [epoch]).fetchall()

    diagnoses = []

    for row in loss_stats:
        agent_id, games, avg_bank, avg_margin = row

        # Inspect telemetry for this agent
        telem = con.execute("""
            SELECT 
                AVG(p0_crew) as avg_crew,
                MAX(p0_crew) as max_crew,
                MAX(p0_quadrants) as max_quadrants,
                MAX(p0_tiles_planted) as max_plants
            FROM match_telemetry mt
            JOIN matches m ON mt.match_id = m.match_id
            WHERE m.epoch = ? AND m.agent_p0 = ?
        """, [epoch, agent_id]).fetchone()

        if telem:
            avg_crew, max_crew, max_quads, max_plants = telem
            flaws = []

            if max_quads == 1:
                flaws.append("SINGLE_QUADRANT_STALL: Farm never expanded to NE quadrant.")

            if max_crew is not None and max_crew <= 1:
                flaws.append("CREW_STARVATION: Never utilized Fibonacci hired hands (actions constrained).")

            if max_plants is not None and max_plants < 8:
                flaws.append("GROUND_UNDERUTILIZATION: Less than 8 tiles actively cultivated.")

            if not flaws:
                flaws.append("TIMING_INEFFICIENCY: Bank compounded too slowly relative to opponent.")

            diagnoses.append({
                "agent_id": agent_id,
                "epoch": epoch,
                "avg_bank": float(avg_bank or 0.0),
                "avg_crew": float(avg_crew or 0.0),
                "max_quadrants": int(max_quads or 1),
                "flaws": flaws,
            })

    con.close()
    return diagnoses
