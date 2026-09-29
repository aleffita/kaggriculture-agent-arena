# AGENTS.md: Experiment 008 — Antigravity Hybrid Orchestrator & Local Telemetry Sentinel

Welcome to Sublevel 008. This directory investigates the architectural synergy between Cloud Architects and On-Device Edge Models using `google-antigravity` and `litert-lm` on the NVIDIA GeForce GTX 1050 Ti (GPU 1).

---

## 1. Architectural Principles

1. **Strict Hardware Target**:
   - All local model invocations must execute exclusively on the **GTX 1050 Ti (GPU 1)** via `select_gpu("1050ti")` or `$env:LITERT_GPU_INDEX = "1"`.
   - Never spill compute onto the RTX 2060 (GPU 0) or CPU unless comparative benchmarks are explicitly requested.
2. **Pure Text Mode**:
   - Disable vision and audio backends (`vision_backend=None`, `audio_backend=None`) to preserve ~550 MB of VRAM on the 4GB GDDR5 framebuffer.
3. **AST Skeleton Extraction**:
   - Zero private code bodies, strings, or secret literals may leave the machine.
   - Only `ast.parse` signatures (`imports`, `globals`, `defs`) are shared with the Cloud Architect contract layer.
4. **Slingshot / Bursty Telemetry Stream**:
   - The telemetry generator models bursty non-uniform packet/metric distributions ("estilingue na banda"), alternating between idle intervals and compressed burst clusters.
   - Local Gemma 4 on the 1050 Ti acts as the real-time sentinel, classifying spikes ($\ge 80$) and triggering throttle interventions via custom tools.

---

## 2. Directory Structure

```text
experiments/008_antigravity_hybrid_orchestrator/
├── AGENTS.md                 # This directive document
├── README.md                 # Deep architectural analysis and telemetry theory
├── slingshot_stream.py       # Stochastic bursty metric generator (0-100 with slingshot jitter)
├── hybrid_orchestrator.py    # Antigravity Agent pipeline with LiteRTAgentConfig on 1050 Ti
└── run_probe.py              # Automated probe execution and validation suite
```

---

## 3. Operational Protocols

- Execute with `uv run python experiments/008_antigravity_hybrid_orchestrator/run_probe.py`.
- Adhere strictly to single atomic commands and zero-polling reactive execution.
