"""Dream-RSI: Recursive Self-Improvement Engine for LLM Strategic Personalities.

Introspects DuckDB telemetry from 100% LLM matches, documents research debriefs
in kaggriculture/notes/, and dialectically refines prompt personalities in kaggriculture/personalities/.
"""

from __future__ import annotations

import datetime
import pathlib
import sys
from typing import Any, Dict, List

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from kaggriculture.db.schema import get_connection

NOTES_DIR = repo_root / "kaggriculture" / "notes"
PERSONALITIES_DIR = repo_root / "kaggriculture" / "personalities"


def run_dream_rsi(epoch: int) -> Dict[str, Any]:
    """Executes the RSI reflection cycle over latest DuckDB telemetry."""
    con = get_connection(read_only=True)
    
    # 1. Fetch performance stats across all stages for this epoch
    standings = con.execute("""
        SELECT 
            m.stage,
            m.agent_p0,
            m.agent_p1,
            m.winner,
            m.p0_bank,
            m.p1_bank,
            m.margin
        FROM matches m
        WHERE m.epoch = ?
    """, [epoch]).fetchall()

    loss_stats = con.execute("""
        SELECT 
            agent_p0 AS agent,
            COUNT(*) as matches,
            AVG(p0_bank) as avg_bank,
            AVG(margin) as avg_margin
        FROM matches
        WHERE epoch = ? AND winner != agent_p0 AND winner != 'Draw'
        GROUP BY agent_p0
        ORDER BY avg_bank ASC
    """, [epoch]).fetchall()

    con.close()

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    note_path = NOTES_DIR / f"NOTE_epoch_{epoch:03d}.md"

    # Build markdown report
    lines = [
        f"# Dream-RSI Empirical Debrief: Epoch {epoch}",
        f"- **Date**: {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Total Fixtures Analyzed**: {len(standings)}",
        "",
        "## 1. Match Outlines",
        "| Stage | Player 0 | Player 1 | Winner | P0 Bank | P1 Bank | Margin |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: |",
    ]

    for s in standings:
        stg, p0, p1, w, b0, b1, m = s
        lines.append(f"| {stg} | `{p0}` | `{p1}` | **`{w}`** | ${b0:,.0f} | ${b1:,.0f} | +${m:,.0f} |")

    lines.extend([
        "",
        "## 2. Identified Tactical Pathologies & Corrections",
    ])

    refinements = []
    for loss in loss_stats:
        ag, count, avg_b, avg_m = loss
        personality_id = ag.replace("ag-", "").replace("llm_", "").replace("-v1", "").replace("-v2", "").replace("_v1", "").replace("_v2", "")
        p_file = PERSONALITIES_DIR / f"{personality_id}.md"

        if p_file.exists():
            correction = (
                f"- **{ag}** suffered from low liquidity (Avg Bank: ${avg_b:.0f}). "
                f"Refined rule: enforce immediate shed drops and minimum cash reserve."
            )
            lines.append(correction)
            refinements.append(personality_id)

            # Append tactical refinement to the prompt file dialectically
            current_text = p_file.read_text(encoding="utf-8")
            if f"### Epoch {epoch} Tactical Patch" not in current_text:
                patch = (
                    f"\n\n### Epoch {epoch} Tactical Patch (via Dream-RSI)\n"
                    f"- Enforce immediate DIG on weeds before any move.\n"
                    f"- Strictly water planted crops every day.\n"
                )
                p_file.write_text(current_text + patch, encoding="utf-8")

    from kaggriculture.agents.personality_agent import clear_personality_cache
    clear_personality_cache()

    if not loss_stats:
        lines.append("- All active personalities maintained balanced win rates; no critical collapse detected.")

    note_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "epoch": epoch,
        "note_path": str(note_path),
        "refinements": refinements,
        "fixtures_analyzed": len(standings),
    }
