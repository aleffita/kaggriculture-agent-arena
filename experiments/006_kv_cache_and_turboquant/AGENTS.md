# AGENTS.md: KV Cache Architecture & TurboQuant Research

## 1. Study Scope
Comprehensive analysis of **Key-Value (KV) Cache management** in LiteRT-LM and mathematical evaluation of **TurboQuant** (Google Research 2026) for constrained memory architectures (**NVIDIA GeForce GTX 1050 Ti**).

## 2. Theoretical Mechanics
- **The KV Cache Bottleneck**:
  - In autoregressive generation, keys and values of previous tokens must be stored to compute attention:
    $$\text{Memory}_{\text{KV}} = 2 \times L \times H_{\text{kv}} \times d_{\text{head}} \times \text{precision} \times S \times B$$
  - While compact for Gemma 4 E2B (~74 MB at 4096 tokens), large models (e.g. Gemma 4 26B or long 32K context windows) create a memory wall that instantly exceeds 4GB VRAM.
- **LiteRT-LM Native Mechanisms**:
  - `use_ringbuffers_local_attention`: Enforces rolling buffer windows, bounding KV allocation.
  - Disk serialization cache: Prevents recompilation overhead on startup.
- **TurboQuant (Google 2026)**:
  - Two-stage online vector quantization:
    1. **PolarQuant**: Random orthogonal rotation matrix $R$ spreads information across all dimensions, followed by optimal scalar quantization (3-bit / 4-bit).
    2. **QJL (Quantized Johnson-Lindenstrauss)**: 1-bit residual transform preserving inner-product geometry $\langle q, k \rangle$.
  - Data-oblivious, zero fine-tuning required.

## 3. Execution Directives
- **Target Device**: ALWAYS enforce GTX 1050 Ti (`$env:LITERT_GPU_INDEX="1"`).
- **Probe Command**:
  ```powershell
  uv run python experiments/006_kv_cache_and_turboquant/run_probe.py
  ```
