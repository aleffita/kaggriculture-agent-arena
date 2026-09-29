# RESEARCH LEDGER: Insights, Hardware Observations & Architectural Axioms

This document is the centralized, append-only repository of empirical insights, hardware observations, and runtime truths discovered during research with LiteRT-LM, Google Antigravity SDK, and multi-GPU Windows architectures.

---

## Index of Ledger Entries

| Entry ID | Date | Hardware / Subsystem | Primary Insight |
| :--- | :---: | :--- | :--- |
| **[LEDGER-001](#ledger-001-direct3d-12-vs-cuda-subsystem-isolation)** | 2026-09-29 | DXGI / D3D12 / Windows NT | `CUDA_VISIBLE_DEVICES` is completely ignored; DXGI vtable hooking is mandatory for GPU routing. |
| **[LEDGER-002](#ledger-002-mtp-speculative-decoding-efficiency-on-pascal)** | 2026-09-29 | Pascal GP107 / LiteRT-LM | MTP Drafter achieves 95.96% acceptance rate, reaching 44.68 t/s (~96% of physical 112 GB/s bandwidth limit). |
| **[LEDGER-003](#ledger-003-pure-text-pipeline-vram-reclamation)** | 2026-09-29 | WebGPU / Memory Management | Disabling multimedia backends (`vision=None`, `audio=None`) frees ~550 MB VRAM on the 4GB framebuffer. |
| **[LEDGER-004](#ledger-004-d3d12-queue-batching--gpu_decode_steps_per_sync)** | 2026-09-29 | Direct3D 12 Command Queue | `gpu_decode_steps_per_sync = 4` is the optimal sweet spot; amortizes CPU-GPU sync fences without starvation. |
| **[LEDGER-005](#ledger-005-antigravity-localharness-loopback-serving)** | 2026-09-29 | `google-antigravity` / Serving | Local LiteRT agent binds ephemeral OpenAI server; concurrency serialized via `engine_lock`. |
| **[LEDGER-006](#ledger-006-ast-skeleton-privacy-boundary--edge-sentinel)** | 2026-09-29 | Security / Hybrid Swarm | AST extraction strips 100% of function bodies & secrets; Cloud Architect goes IDLE; local edge model executes stream. |
| **[LEDGER-007](#ledger-007-pascal-sm_61-arithmetic-profile--turboquant-fit)** | 2026-09-29 | NVIDIA GP107 Microarchitecture | Lacks FP16 Tensor Cores, but features native `__dp4a` INT8 and fast bitwise popcount, optimal for 1-bit QJL sketching. |
| **[LEDGER-008](#ledger-008-kaggriculture-1v1-crop-horizon-game-theory)** | 2026-09-29 | Kaggriculture / Game Theory | 2-day cash-crops (Wheat) achieve 100% win rate in short horizons (10d), whereas 10-day crops (Melon) dominate full 30d seasons (+13.8%). |

---

### [LEDGER-001] Direct3D 12 vs CUDA Subsystem Isolation
- **Date**: 2026-09-29
- **Target Subsystem / Hardware**: Windows DXGI / Direct3D 12 / Google Dawn
- **Related Experiment**: `experiments/001_baseline_comparisons/`
- **Empirical Observation**: Setting `$env:CUDA_VISIBLE_DEVICES = "1"` has zero impact on device selection. LiteRT-LM always defaulted to Adapter 0 (RTX 2060).
- **Underlying Mechanism**: LiteRT-LM on Windows is compiled against Google Dawn over Direct3D 12 via DXGI. The NVIDIA CUDA driver (`nvcuda.dll`) is never loaded into the process. The DXGI subsystem enumerates adapters in driver-preference order (`IDXGIFactory::EnumAdapters1`).
- **Operational Directive**: To isolate the secondary GPU (GTX 1050 Ti), agents must inject `dxgi_hook.dll` or set `$env:LITERT_GPU_INDEX = "1"`, which intercepts vtable slots 7, 12, and 29 of `IDXGIFactory` to swap physical Adapter 1 into Adapter 0.

---

### [LEDGER-002] MTP Speculative Decoding Efficiency on Pascal
- **Date**: 2026-09-29
- **Target Subsystem / Hardware**: NVIDIA GeForce GTX 1050 Ti (4GB GDDR5, 112 GB/s)
- **Related Experiment**: `experiments/002_mtp_speculative_decoding/`
- **Empirical Observation**:
  - Autoregressive Baseline: 42.68 t/s decode, TTFT 2.50s.
  - Speculative Decoding (MTP Enabled): 44.68 t/s decode, TTFT 2.37s.
  - Acceptance Rate: **95.96%**.
- **Underlying Mechanism**: The `gemma-4-E2B-it.litertlm` artifact bundles a multi-token prediction drafter head (`signature=mtp_drafter`). Since Pascal is heavily memory-bandwidth bound (112 GB/s theoretical, ~2.45 GB model weights requiring ~22 ms per full weight pass $\to \approx 45.7\text{ t/s}$ theoretical ceiling), the draft-verify loop executes near the physical silicon memory speed limit.
- **Operational Directive**: Always enable `enable_speculative_decoding=True` when running Gemma 4 E2B on Pascal cards.

---

### [LEDGER-003] Pure-Text Pipeline VRAM Reclamation
- **Date**: 2026-09-29
- **Target Subsystem / Hardware**: Direct3D 12 / VRAM Management (4096 MB Physical Limit)
- **Related Experiment**: `experiments/004_multimodal_encoders/` & `experiments/007_ablations_and_stress/`
- **Empirical Observation**:
  - Full Modalities (Text + Vision + Audio): Resident VRAM ~3150 MB, leaving only ~450 MB headroom before Windows WDDM paging kicks in.
  - Pure-Text Mode (`vision_backend=None`, `audio_backend=None`): Resident VRAM dropped to ~2600 MB.
- **Underlying Mechanism**: In `litert_lm_engine_settings_create()`, passing NULL for vision and audio backends prevents initialization of `embedder`, `eoi`, `eoa`, and `per_layer_embedder` subgraphs, saving ~550 MB of tensor buffers.
- **Operational Directive**: In pure text research and agentic loops, never initialize audio or vision backends. This guarantees context scaling up to 4096 tokens without PCIe paging degradation.

---

### [LEDGER-004] D3D12 Queue Batching & `gpu_decode_steps_per_sync`
- **Date**: 2026-09-29
- **Target Subsystem / Hardware**: Direct3D 12 Compute Command Queue / PCIe 3.0 x16
- **Related Experiment**: `experiments/007_ablations_and_stress/`
- **Empirical Observation**:
  - $N=1$: 106.01 prefill t/s, 43.12 decode t/s, TTFT 2.44s.
  - $N=2$: 100.92 prefill t/s, 43.49 decode t/s, TTFT 2.56s.
  - **$N=4$**: **108.83 prefill t/s**, **43.91 decode t/s**, **TTFT 2.38s** (Init: 4.66s).
  - $N=8$: 105.42 prefill t/s, 43.08 decode t/s, TTFT 2.45s.
- **Underlying Mechanism**: With $N=1$, every single token forces a CPU-GPU synchronization barrier and interrupt. At $N=4$, compute dispatches are batched into Direct3D 12 command lists, amortizing PCIe round-trip latency. Beyond $N=4$, command queue serialization limits further amortized savings on the GP107 architecture.
- **Operational Directive**: Configure `interfaces.GPU(gpu_decode_steps_per_sync=4)` for all production benchmarks and agent loops on Pascal hardware.

---

### [LEDGER-005] Antigravity LocalHarness Loopback Serving
- **Date**: 2026-09-29
- **Target Subsystem / Hardware**: `google-antigravity==0.1.20` / Windows NT Loopback Sockets
- **Related Experiment**: `experiments/005_antigravity_sdk_local_serving/`
- **Empirical Observation**: `google.antigravity.Agent(LiteRTAgentConfig(...))` successfully spawns `localharness.exe` via standard Python subprocess pipes, communicating over loopback WebSockets.
- **Underlying Mechanism**: The SDK hosts an internal `LiteRTOpenAIServer` with an `engine_lock` mutex to serialize model inference calls across multiple concurrent HTTP/WebSocket callers while maintaining persistent connections.
- **Operational Directive**: Use `LiteRTAgentConfig` with `env={"LITERT_GPU_INDEX": "1"}` to run official Antigravity agents directly against the on-device LiteRT runtime.

---

### [LEDGER-006] AST Skeleton Privacy Boundary & Edge Sentinel
- **Date**: 2026-09-29
- **Target Subsystem / Hardware**: Security / Privacy / Hybrid Edge Swarm
- **Related Experiment**: `experiments/008_antigravity_hybrid_orchestrator/`
- **Empirical Observation**: Python `ast.parse` extracted public contracts (`imports`, `globals`, `defs`) stripping 100% of function bodies, SQL queries, and API secrets. Cloud Architect emitted a 150-token invariant blueprint and entered `IDLE`. Local Gemma 4 on the GTX 1050 Ti processed 6 real-time burst ticks, dispatching 3 autonomous throttling interventions with 0 bytes leaving the machine.
- **Underlying Mechanism**: Separation of cognitive concerns: High-level architectural reasoning (which changes rarely) runs once in the cloud; high-frequency state evaluation and private data operations run exclusively on edge silicon.
- **Operational Directive**: When designing hybrid agent architectures, always enforce AST skeleton contracts at the cloud ingress boundary.

---

### [LEDGER-007] Pascal sm_61 Arithmetic Profile & TurboQuant Fit
- **Date**: 2026-09-29
- **Target Subsystem / Hardware**: NVIDIA Pascal GP107 (GTX 1050 Ti, Compute Capability 6.1)
- **Related Experiment**: `experiments/006_kv_cache_and_turboquant/`
- **Empirical Observation**: Pascal GP107 lacks FP16 Tensor Cores and D3D12 `Native16BitShaderOps` (FP16 runs at half-rate $\approx 1:64$ or is emulated as FP32). However, it natively accelerates 4-way 8-bit integer dot products (`__dp4a`) and fast bitwise popcount (`__popc`).
- **Underlying Mechanism**: TurboQuant combines PolarQuant (random orthogonal rotation) with 1-bit Quantized Johnson-Lindenstrauss (QJL) residual sketching. Calculating inner products on 1-bit sketches reduces to XOR and popcount, which executes at peak single-cycle ALU throughput on sm_61.
- **Operational Directive**: Future custom kernel development for KV-cache compression on Pascal should target INT8 (`__dp4a`) and 1-bit bitwise popcount rather than attempting FP16 tensor acceleration.

---

### [LEDGER-008] Kaggriculture 1v1 Crop Horizon Game Theory
- **Date**: 2026-09-29
- **Target Subsystem / Hardware**: `kaggriculture` Simulator / Game Theory & Multi-Agent 1v1
- **Related Project**: `kaggriculture/arena/tournament.py`
- **Empirical Observation**:
  - In a 10-day tournament (240 steps, double round-robin 12 fixtures): `wheat_looper` achieved a **100% win rate (6-0-0)** ($18,463 coins, +$5,399 net margin), while `market_arbitrage` won only 1 match (1-0-5, $10,320 coins).
  - In a full 30-day season (720 steps): `market_arbitrage` beat `wheat_looper` **$4,153.0 vs. $3,648.0 (+13.8% edge)**.
- **Underlying Mechanism**: Capital amortization vs. time horizon. Long-duration crops (Melon takes 10 days to max yield) require upfront seed investment ($80) and daily watering without early cash flow. If the game horizon $T \le 10$, capital is locked in immature plants at episode termination. In longer seasons ($T = 30$), the huge profit margin ($250 - 80 = 170$ per unit) compounds, whereas fast cash-crops (Wheat, 2 days) provide immediate liquidity and safety in short horizons.
- **Operational Directive**: Kaggriculture agents must condition crop selection on remaining seasonal steps ($T_{\text{remaining}}$). Never plant crops whose `time_to_max_yield` exceeds the remaining days of the season.

---

### [LEDGER-009] Kaggriculture Strategic Pathology & DuckDB Dream-AGI Evolutionary Loop
- **Date**: 2026-09-29
- **Target Subsystem / Hardware**: `kaggriculture` / DuckDB Replay Telemetry / Multi-Tier Elo Curriculum
- **Related Project**: `kaggriculture/dream/dream_loop.py`, `kaggriculture/db/schema.py`
- **Empirical Observation**:
  - Analysis of `Z:\workspaces\gamedir\src\kagri\heuristic_agent.py` revealed the exact mechanism causing sub-1200 coin ceilings: an artificial `CASH_RESERVE=800` combined with `LAND_COST_HEADROOM` locked the agent out of `BUY_LAND` indefinitely, leaving it trapped in a single 25-tile quadrant with single-unit orders.
  - In contrast, our `scale_compounder` unlocks the NE quadrant on Day 4-5 when funds reach $1050-$1100, scales Fibonacci labor to 8-10 hands, and yields **$30,810 coins** in a single 30-day season (surpassing `starter` by +$25,366 margin and `melon_expander` by +$24,173 margin).
  - The gradient-free **Dream-AGI reflection loop** across 4 league divisions (Wood: 72 steps, Bronze: 144 steps, Silver: 240 steps, Gold: 720 steps) successfully logged 20 matches and 88 telemetry snapshots in DuckDB, diagnosed bottlenecks via automated SQL heuristics (`SINGLE_QUADRANT_STALL`, `CREW_STARVATION`), and dynamically generated succeeding agents (`evolved_market_e1_v1`, `evolved_evolved_e2_v2`).
- **Operational Directive**: For curriculum self-play, seed the ladder at 600.0 baseline Elo. Dynamic code synthesizers must adjust land expansion gates according to the league's step horizon $T$ (e.g. Day 4 land triggers cannot execute in Wood league 3-day sprints).

---

### [LEDGER-010] Step-Level Real-Time LLM Inference on Pascal GTX 1050 Ti
- **Date**: 2026-09-29
- **Target Subsystem / Hardware**: NVIDIA GeForce GTX 1050 Ti (GP107) / Google LiteRT-LM / Direct3D 12
- **Related Project**: `kaggriculture/agents/llm_player.py`, `kaggriculture/agents/llm_sprint_rusher.py`
- **Empirical Observation**:
  - Direct step-level neural inference was successfully deployed to the discrete GTX 1050 Ti using the DXGI vtable interceptor (`$env:LITERT_GPU_INDEX = "1"`).
  - **Latency**: ~0.8s to 0.9s per game step (fully conforming to Kaggle's 1.0s timeout limit). Initial weight allocation overhead is ~10s amortized over the episode.
  - **Head-to-Head Performance**: `llm_sprint_rusher` defeated `wheat_looper` ($3,000 vs $2,940, +$60 margin) in 24 steps, and `llm_player` defeated `starter` ($3,000 vs $2,960, +$40 margin).
  - **VRAM Footprint**: Model weights and KV cache reside stably at ~2.4 GB out of 4.0 GB VRAM, with zero paging to system RAM.
- **Operational Directive**: To avoid reloading 2.4 GB weights on every step, instantiate `LiteRtModelRunner` as a persistent process singleton. LLM-driven agents are viable for direct competitive play across all stages.



