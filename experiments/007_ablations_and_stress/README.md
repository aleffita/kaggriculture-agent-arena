# Experiment 007: Hyperparameter Ablations & Hardware Stress Testing

- **Target Device**: NVIDIA GeForce GTX 1050 Ti (4GB GDDR5)
- **Model**: `gemma-4-E2B-it.litertlm`
- **Runtime**: WebGPU via Google Dawn (Direct3D 12)
- **Date**: 2026-09-29

---

## 1. Study Focus: Host-Device Sync Overhead

On consumer graphics cards operating over PCIe 3.0 x16 buses like the GTX 1050 Ti, invoking CPU-GPU synchronization fences after every generated token introduces non-negligible bus roundtrip latency.

The parameter `gpu_decode_steps_per_sync` instructs the WebGPU compute queue to batch $N$ token generation steps before transmitting the completion signal back to the host CPU:
- $N=1$: Strict per-token synchronous feedback (lowest latency for streaming the first token, but highest overhead).
- $N=4$ or $N=8$: Batched dispatch, reducing CPU context switching and PCIe interrupt latency.

---

## 2. VRAM Allocation Guardrails (4096 MB Limit)

| Memory Segment | Allocation Size | Status |
| :--- | :--- | :--- |
| **Model Weights (Gemma 4 E2B)** | ~2450 MB | Static resident in VRAM |
| **Windows Compositor / D3D12 Overhead** | ~350 MB | System overhead |
| **KV Cache & Working Buffers** | ~500 - 900 MB | Dynamic based on sequence length |
| **Safety Headroom** | ~350 MB | Prevents WDDM memory paging to system RAM |

> [!WARNING]
> If total allocated memory exceeds 4096 MB, Windows WDDM driver will begin paging memory chunks to system DRAM over PCIe, which causes token generation speed to collapse from ~40 t/s down to <5 t/s. Always keep context sequences $\le 4096$ tokens when operating on the GTX 1050 Ti.

---

## 3. How to Run the Probe

```powershell
uv run python experiments/007_ablations_and_stress/run_probe.py
```

---

## 4. Empirical Results (GTX 1050 Ti Pascal GP107)

| Steps Per Sync | Prefill Speed (t/s) | Decode Speed (t/s) | TTFT (s) | Engine Init (s) | Efficiency Ratio |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 106.01 | 43.12 | 2.44 | 5.28 | 98.2% |
| **2** | 100.92 | 43.49 | 2.56 | 4.77 | 99.0% |
| **4** (Sweet Spot) | **108.83** | **43.91** | **2.38** | **4.66** | **100.0%** |
| **8** | 105.42 | 43.08 | 2.45 | 5.00 | 98.1% |

### Key Architectural Insights
1. **Queue Batching Sweet Spot**: `gpu_decode_steps_per_sync = 4` delivers the highest decode throughput (43.91 t/s) and highest prefill throughput (108.83 t/s) while cutting TTFT to 2.38s.
2. **PCIe & D3D12 Queue Saturation**: $N=1$ suffers from per-token synchronization interrupts. At $N=8$, command queue serialization limits further amortized savings on the GP107 architecture.
3. **Pure-Text Footprint**: Bypassing vision and audio subgraphs conserves ~550 MB VRAM, leaving ample room for 4096 tokens of KV cache without triggering Windows WDDM paging.

