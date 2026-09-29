# AGENTS.md: MTP Drafter & Speculative Decoding

## 1. Study Scope
Investigation of **Multi-Token Prediction (MTP)** and **Speculative Decoding** within LiteRT-LM on the **NVIDIA GeForce GTX 1050 Ti**.

## 2. Theoretical Mechanics
- **The Bottleneck**: Autoregressive decode on GTX 1050 Ti is strictly memory-bandwidth bound (112 GB/s GDDR5). Each token generated requires reading ~2.4 GB of model weights from VRAM.
- **MTP / Speculative Decoding**: The model architecture includes an integrated verify subgraph (`signature=verify`). The drafter proposes $K$ tokens which are verified in parallel in a single matrix-multiplication forward pass. When acceptance rate $\alpha > 0.5$, tokens/sec scales dramatically without increasing DRAM bandwidth requirements.

## 3. Execution Directives
- **Target Device**: ALWAYS enforce GTX 1050 Ti (`$env:LITERT_GPU_INDEX="1"`).
- **Flag**: `enable_speculative_decoding=True` vs `False`.
- **Probe Command**:
  ```powershell
  uv run python experiments/002_mtp_speculative_decoding/run_probe.py
  ```
