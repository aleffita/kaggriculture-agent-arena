# AGENTS.md: Ablations & GPU Stress Testing

## 1. Study Scope
Ablation studies examining the sensitivity of the **NVIDIA GeForce GTX 1050 Ti** to LiteRT-LM runtime hyperparameters:
1. `gpu_decode_steps_per_sync` (Host-Device synchronization interval: 1, 2, 4, 8).
2. Context window length vs. Time to First Token (TTFT).
3. VRAM allocation limits (4096 MB boundary stress).

## 2. Theoretical Mechanics
- **Host-Device Synchronization Bottleneck**:
  - In WebGPU/D3D12, after dispatching a compute pass, the host CPU must synchronize with the GPU queue to receive the sampled token id.
  - Setting `gpu_decode_steps_per_sync > 1` allows the GPU to execute multiple autoregressive steps autonomously before blocking on a CPU fence, reducing PCIe latency on older PCIe 3.0 interfaces like the GTX 1050 Ti.

## 3. Execution Directives
- **Target Device**: ALWAYS enforce GTX 1050 Ti (`$env:LITERT_GPU_INDEX="1"`).
- **Probe Command**:
  ```powershell
  uv run python experiments/007_ablations_and_stress/run_probe.py
  ```
