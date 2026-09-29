# AGENTS.md: LLM Strategic Personalities (`kaggriculture/personalities/`)

This directory contains the prompt templates, strategic worldviews, and behavioral priors for **100% LLM agents** competing in the Kaggriculture arena on the NVIDIA GTX 1050 Ti.

---

## 1. Personality Specification Contract

Each personality file is a Markdown document (`<personality_name>.md`) exporting:
1. **Strategic Worldview & Priors**: Fundamental philosophy (e.g. liquidity-first, territorial expansion, workforce scaling).
2. **Horizon Affinity**: Primary suitability (Sprint, Expansion, Scaling, Full Season).
3. **Decision Rules & Prompt Core**: Instructions injected into the system prompt of Gemma 4 on the GTX 1050 Ti.
4. **Action Output Contract**: Strict requirement to emit valid action JSON:
   ```json
   {"farmer": ["ACTION", ...], "market": [["ORDER", ...]]}
   ```

---

## 2. Directory Invariants

- **Hardware Targeting**: All personalities execute inference against the shared in-memory `LiteRtModelRunner` bound to **GPU 1 (GTX 1050 Ti)**.
- **RSI Evolution**: Files in this directory are dialectically evolved by **Dream-RSI** based on DuckDB match telemetry, preserving aligned context while updating tactical rules.
- **No Procedural Overrides**: Strategic choices must originate from the model's neural token generation, not hardcoded Python overrides.
