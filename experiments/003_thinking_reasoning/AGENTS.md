# AGENTS.md: Thinking & Reasoning Configurations

## 1. Study Scope
Investigating Chain-of-Thought (CoT), thinking token budgets, and reasoning generation within LiteRT-LM on the **NVIDIA GeForce GTX 1050 Ti**.

## 2. Theoretical Mechanics
- **Thinking Tokens**: Models designed with reasoning capabilities generate internal rationale sequences before the final visible response.
- **Budget Allocation**: `litert_lm.ThinkingConfig(thinking_token_budget=...)` controls the maximum internal reasoning length.
- **Memory Impact**: Because reasoning tokens occupy KV cache slots during generation, allocating a large thinking budget (e.g. >2048 tokens) directly competes with the remaining VRAM on a 4GB card like the GTX 1050 Ti.

## 3. Execution Directives
- **Target Device**: ALWAYS enforce GTX 1050 Ti (`$env:LITERT_GPU_INDEX="1"`).
- **Probe Command**:
  ```powershell
  uv run python experiments/003_thinking_reasoning/run_probe.py
  ```
