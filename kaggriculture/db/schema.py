"""DuckDB Database Schema and Operations for Kaggriculture Arena."""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
from typing import Any, Dict, List, Optional
import duckdb

DB_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
DB_PATH = DB_DIR / "arena.duckdb"


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Returns a DuckDB connection, ensuring target directory exists."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def initialize_schema():
    """Initializes tables for agents, matches, telemetry, leagues, and dream logs."""
    con = get_connection()
    con.execute("""
    CREATE TABLE IF NOT EXISTS agents (
        agent_id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL,
        version VARCHAR NOT NULL,
        league VARCHAR NOT NULL DEFAULT 'Wood',
        elo DOUBLE NOT NULL DEFAULT 600.0,
        matches_played INTEGER NOT NULL DEFAULT 0,
        wins INTEGER NOT NULL DEFAULT 0,
        losses INTEGER NOT NULL DEFAULT 0,
        draws INTEGER NOT NULL DEFAULT 0,
        total_coins DOUBLE NOT NULL DEFAULT 0.0,
        created_at TIMESTAMP NOT NULL,
        description VARCHAR,
        code_path VARCHAR
    );

    CREATE TABLE IF NOT EXISTS matches (
        match_id VARCHAR PRIMARY KEY,
        epoch INTEGER NOT NULL,
        league VARCHAR NOT NULL,
        steps INTEGER NOT NULL,
        agent_p0 VARCHAR NOT NULL,
        agent_p1 VARCHAR NOT NULL,
        p0_bank DOUBLE NOT NULL,
        p1_bank DOUBLE NOT NULL,
        winner VARCHAR NOT NULL,
        margin DOUBLE NOT NULL,
        p0_elo_before DOUBLE NOT NULL,
        p0_elo_after DOUBLE NOT NULL,
        p1_elo_before DOUBLE NOT NULL,
        p1_elo_after DOUBLE NOT NULL,
        duration_s DOUBLE NOT NULL,
        timestamp TIMESTAMP NOT NULL
    );

    CREATE TABLE IF NOT EXISTS match_telemetry (
        telemetry_id VARCHAR PRIMARY KEY,
        match_id VARCHAR NOT NULL,
        turn INTEGER NOT NULL,
        day INTEGER NOT NULL,
        hour INTEGER NOT NULL,
        p0_money DOUBLE NOT NULL,
        p1_money DOUBLE NOT NULL,
        p0_crew INTEGER NOT NULL,
        p1_crew INTEGER NOT NULL,
        p0_tiles_planted INTEGER NOT NULL,
        p1_tiles_planted INTEGER NOT NULL,
        p0_quadrants INTEGER NOT NULL,
        p1_quadrants INTEGER NOT NULL,
        market_wheat DOUBLE,
        market_carrot DOUBLE,
        market_melon DOUBLE
    );

    CREATE TABLE IF NOT EXISTS elo_history (
        history_id VARCHAR PRIMARY KEY,
        agent_id VARCHAR NOT NULL,
        match_id VARCHAR NOT NULL,
        elo_before DOUBLE NOT NULL,
        elo_after DOUBLE NOT NULL,
        delta DOUBLE NOT NULL,
        timestamp TIMESTAMP NOT NULL
    );

    CREATE TABLE IF NOT EXISTS league_promotions (
        event_id VARCHAR PRIMARY KEY,
        agent_id VARCHAR NOT NULL,
        from_league VARCHAR NOT NULL,
        to_league VARCHAR NOT NULL,
        epoch INTEGER NOT NULL,
        reason VARCHAR NOT NULL,
        timestamp TIMESTAMP NOT NULL
    );

    CREATE TABLE IF NOT EXISTS dream_insights (
        insight_id VARCHAR PRIMARY KEY,
        epoch INTEGER NOT NULL,
        agent_id VARCHAR NOT NULL,
        flaw_diagnosed VARCHAR NOT NULL,
        mutation_applied VARCHAR NOT NULL,
        successor_agent_id VARCHAR,
        timestamp TIMESTAMP NOT NULL
    );
    """)
    con.close()


def register_agent(
    name: str,
    version: str = "v1",
    league: str = "Wood",
    initial_elo: float = 600.0,
    description: str = "",
    code_path: str = "",
) -> str:
    """Registers an agent if not already present, returns agent_id."""
    agent_id = f"{name}_{version}"
    con = get_connection()
    exists = con.execute("SELECT 1 FROM agents WHERE agent_id = ?", [agent_id]).fetchone()
    if not exists:
        con.execute(
            """
            INSERT INTO agents (agent_id, name, version, league, elo, created_at, description, code_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                agent_id,
                name,
                version,
                league,
                initial_elo,
                datetime.datetime.now(datetime.timezone.utc),
                description,
                code_path,
            ],
        )
    con.close()
    return agent_id


def record_match(
    match_id: str,
    epoch: int,
    league: str,
    steps: int,
    agent_p0: str,
    agent_p1: str,
    p0_bank: float,
    p1_bank: float,
    winner: str,
    margin: float,
    p0_elo_before: float,
    p0_elo_after: float,
    p1_elo_before: float,
    p1_elo_after: float,
    duration_s: float,
    telemetry_samples: Optional[List[Dict[str, Any]]] = None,
):
    """Persists match record, updates agent Elo & standings, and records telemetry."""
    con = get_connection()
    now = datetime.datetime.now(datetime.timezone.utc)

    # Insert match
    con.execute(
        """
        INSERT INTO matches (
            match_id, epoch, league, steps, agent_p0, agent_p1, p0_bank, p1_bank,
            winner, margin, p0_elo_before, p0_elo_after, p1_elo_before, p1_elo_after,
            duration_s, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            match_id, epoch, league, steps, agent_p0, agent_p1, p0_bank, p1_bank,
            winner, margin, p0_elo_before, p0_elo_after, p1_elo_before, p1_elo_after,
            duration_s, now,
        ],
    )

    # Record Elo history
    p0_delta = p0_elo_after - p0_elo_before
    p1_delta = p1_elo_after - p1_elo_before
    con.execute(
        "INSERT INTO elo_history VALUES (?, ?, ?, ?, ?, ?, ?)",
        [f"{match_id}_p0", agent_p0, match_id, p0_elo_before, p0_elo_after, p0_delta, now],
    )
    con.execute(
        "INSERT INTO elo_history VALUES (?, ?, ?, ?, ?, ?, ?)",
        [f"{match_id}_p1", agent_p1, match_id, p1_elo_before, p1_elo_after, p1_delta, now],
    )

    # Update agents table
    p0_win = 1 if winner == agent_p0 else 0
    p0_loss = 1 if (winner != agent_p0 and winner != "Draw") else 0
    p0_draw = 1 if winner == "Draw" else 0

    p1_win = 1 if winner == agent_p1 else 0
    p1_loss = 1 if (winner != agent_p1 and winner != "Draw") else 0
    p1_draw = 1 if winner == "Draw" else 0

    con.execute(
        """
        UPDATE agents
        SET elo = ?, matches_played = matches_played + 1,
            wins = wins + ?, losses = losses + ?, draws = draws + ?,
            total_coins = total_coins + ?
        WHERE agent_id = ?
        """,
        [p0_elo_after, p0_win, p0_loss, p0_draw, p0_bank, agent_p0],
    )

    con.execute(
        """
        UPDATE agents
        SET elo = ?, matches_played = matches_played + 1,
            wins = wins + ?, losses = losses + ?, draws = draws + ?,
            total_coins = total_coins + ?
        WHERE agent_id = ?
        """,
        [p1_elo_after, p1_win, p1_loss, p1_draw, p1_bank, agent_p1],
    )

    # Insert telemetry samples if provided
    if telemetry_samples:
        for idx, s in enumerate(telemetry_samples):
            tid = f"{match_id}_t{idx}"
            con.execute(
                """
                INSERT INTO match_telemetry VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    tid, match_id, s.get("turn", 0), s.get("day", 0), s.get("hour", 0),
                    s.get("p0_money", 0.0), s.get("p1_money", 0.0),
                    s.get("p0_crew", 0), s.get("p1_crew", 0),
                    s.get("p0_tiles_planted", 0), s.get("p1_tiles_planted", 0),
                    s.get("p0_quadrants", 1), s.get("p1_quadrants", 1),
                    s.get("market_wheat"), s.get("market_carrot"), s.get("market_melon"),
                ],
            )

    con.close()


def log_promotion(agent_id: str, from_league: str, to_league: str, epoch: int, reason: str):
    """Records a league promotion or demotion."""
    con = get_connection()
    now = datetime.datetime.now(datetime.timezone.utc)
    event_id = f"promo_{agent_id}_{epoch}_{now.strftime('%H%M%S')}"
    con.execute(
        """
        INSERT INTO league_promotions VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [event_id, agent_id, from_league, to_league, epoch, reason, now],
    )
    con.execute(
        "UPDATE agents SET league = ? WHERE agent_id = ?",
        [to_league, agent_id],
    )
    con.close()


def log_dream_insight(epoch: int, agent_id: str, flaw: str, mutation: str, successor_id: str = ""):
    """Logs an evolutionary reflection insight and code mutation."""
    con = get_connection()
    now = datetime.datetime.now(datetime.timezone.utc)
    insight_id = f"dream_{agent_id}_ep{epoch}_{now.strftime('%H%M%S')}"
    con.execute(
        """
        INSERT INTO dream_insights VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [insight_id, epoch, agent_id, flaw, mutation, successor_id, now],
    )
    con.close()
