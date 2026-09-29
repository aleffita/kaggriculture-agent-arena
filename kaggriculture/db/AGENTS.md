# AGENTS.md: DuckDB Arena Storage & Replay Subsystem (`kaggriculture/db/`)

This directory manages the persistent DuckDB analytical database for match replays, telemetry, Elo ratings, and league transitions.

---

## 1. Storage Invariants

- **Database Path**: All runs write to `kaggriculture/data/arena.duckdb` (persisted on disk).
- **Concurrency**: DuckDB connections should be initialized per thread or process, or pooled cleanly to avoid lock contention.
- **Schema Discipline**:
  - `agents`: Current registry of all competing agents and their active league.
  - `matches`: High-level match outcomes (p0, p1, banks, winner, Elo shifts).
  - `match_telemetry`: Turn-by-turn state samples (crew size, tiles planted, money curves).
  - `elo_history`: Full audit trail of rating shifts per match.
  - `league_promotions`: Historical promotion/demotion events across divisions.
  - `dream_insights`: Diagnosed flaws and proposed agent mutations.
