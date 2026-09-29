# AGENTS.md: Dream-AGI Recursive Self-Improvement Loop (`kaggriculture/dream/`)

This directory houses the recursive, gradient-free meta-agent evolution engine inspired by "Dream-AGI" reflection paradigms.

---

## 1. Architectural Philosophy

Rather than relying on heavy reinforcement learning gradients (e.g. fragile GRPO on weak baselines), this engine uses an inner/outer evolutionary reflection loop:

1. **Inner Loop (Match Arena & Ingestion)**:
   - Evaluates agents in 1v1 duels within their respective league tier.
   - Updates Elo ratings (starting at 600.0) and transitions agents across Wood, Bronze, Silver, and Gold leagues.
   - Persists all episode telemetry, money curves, and market price fluctuations directly into **DuckDB** (`kaggriculture/data/arena.duckdb`).
2. **Outer Loop (Dreaming & Introspection)**:
   - Introspects DuckDB match traces and failure modes (e.g. land gates locked, unspent worker-turns, crop decay, market price collapse).
   - Formulates algorithmic diagnoses without human intervention.
   - Synthesizes new agent mutations or parameter adaptations in `kaggriculture/agents/`.
   - Injects the newly synthesized agent into the Wood league at 600.0 Elo to challenge the incumbent hierarchy.

---

## 2. Invariants

- **No Premature Land Purchases**: Land without crew to work it is negative value.
- **No Infinite Cash Reserves**: Avoid gates like `money > 800 + LAND_COST_HEADROOM` that permanently trap agents in single-quadrant poverty.
- **All Data in DuckDB**: Replays and match logs must be queryable via standard SQL.
