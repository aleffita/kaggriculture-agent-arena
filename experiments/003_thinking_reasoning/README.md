# Experiment 003: Thinking & Chain-of-Thought (CoT) Reasoning

- **Target Device**: NVIDIA GeForce GTX 1050 Ti (4GB GDDR5)
- **Model**: `gemma-4-E2B-it.litertlm`
- **Runtime**: WebGPU via Google Dawn (Direct3D 12)
- **Date**: 2026-09-29

---

## 1. Architectural Foundation

Thinking models generate internal reasoning sequences prior to producing the user-facing response. In LiteRT-LM:
- `litert_lm.ThinkingConfig(enable_thinking=True, thinking_token_budget=N)` enables setting a hard ceiling on reasoning tokens.
- When thinking is active, tokens are generated inside specialized markers (e.g. `<thought> ... </thought>`) or internally managed token channels before emitting visible tokens.

---

## 2. VRAM Trade-offs on 4GB GPUs

1. **Context Window Contention**:
   - Every generated reasoning token remains in the KV-cache during generation.
   - On a 4GB card (where available cache memory is ~1.2 GB), setting an excessive thinking budget (e.g. 2048+ tokens) will either hit the KV-cache ringbuffer limit or trigger out-of-memory errors.
2. **Recommended Ceiling**:
   - For Gemma 4 E2B on GTX 1050 Ti, the optimal thinking budget is **256 to 512 tokens**, preserving sufficient VRAM headroom for multi-turn conversational history.

---

## 3. How to Run the Probe

```powershell
uv run python experiments/003_thinking_reasoning/run_probe.py
```
