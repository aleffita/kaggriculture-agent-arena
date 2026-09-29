# AGENTS.md: Experiments Hierarchy Master Directive

Welcome, Agent. This directory, `experiments/`, hosts empirical investigations, system probes, architectural studies, and ablation benchmarks on **Google LiteRT-LM** and local AI execution.

---

## 1. Prime Directives for Experiments

1. **Hardware Target**: 
   - **ALWAYS target the NVIDIA GeForce GTX 1050 Ti (GPU 1)** by default.
   - Do NOT route to the RTX 2060 or CPU unless the user explicitly requests a cross-device comparative study.
2. **GPU Routing Enforcement**:
   - LiteRT-LM uses Direct3D 12 / WebGPU via Google Dawn. **`CUDA_VISIBLE_DEVICES` has zero effect**.
   - Always enforce `$env:LITERT_GPU_INDEX = "1"` in shell or use `with select_gpu("1050ti"):` in Python scripts.
3. **4GB VRAM Budget Discipline**:
   - The GTX 1050 Ti (Pascal GP107) has **4096 MB GDDR5**.
   - With Gemma-4-E2B taking ~2450 MB for weights, available headroom for KV-cache, context windows, and framebuffers is **~1300 MB**.
   - Guard against unbounded context allocation. Monitor VRAM allocations using `nvidia-smi`.
4. **All Documentation in English**:
   - Every `AGENTS.md`, `README.md`, docstring, and code comment in the `experiments/` hierarchy must be written in **English**.

---

## 2. Directory Hierarchy and Sublevel Topics

```text
experiments/
├── AGENTS.md                                # This master directive
├── 001_baseline_comparisons/               # Baseline: RTX 2060 vs GTX 1050 Ti vs CPU
│   └── README.md
├── 002_mtp_speculative_decoding/            # Study 1: MTP Drafter and Speculative Decoding
│   ├── AGENTS.md
│   ├── README.md
│   └── run_probe.py
├── 003_thinking_reasoning/                  # Study 2: Thinking budgets, reasoning tokens & CoT
│   ├── AGENTS.md
│   ├── README.md
│   └── run_probe.py
├── 004_multimodal_encoders/                 # Study 3: Vision (SigLIP/PaliGemma) and Audio (ASR/TTS)
│   ├── AGENTS.md
│   ├── README.md
│   └── run_probe.py
├── 005_antigravity_sdk_local_serving/       # Study 4: Google Antigravity SDK & Local Serving
│   ├── AGENTS.md
│   ├── README.md
│   ├── local_server.py
│   └── test_client.py
├── 006_kv_cache_and_turboquant/             # Study 5: KV Cache (Ringbuffers) & TurboQuant research
│   ├── AGENTS.md
│   ├── README.md
│   └── run_probe.py
└── 007_ablations_and_stress/                # Study 6: Context length, batching & VRAM stress
    ├── AGENTS.md
    ├── README.md
    └── run_probe.py
```

---

## 3. Subfolder Protocol & Standards

Every experiment subdirectory must adhere to the following contract:
1. **`AGENTS.md`**: Outlines specific parameters, flags, theoretical constraints, and execution instructions for that experiment.
2. **`README.md`**: Contains:
   - Theoretical premise and hypothesis.
   - Exact model reference and quantization level.
   - Empirical results table (prefill speed, decode speed, TTFT, memory footprint).
   - Analysis of architectural findings.
3. **Executable Probes (`run_probe.py`)**:
   - Must be executable via `uv run python experiments/<folder>/run_probe.py`.
   - Must automatically bind to GTX 1050 Ti via `select_gpu("1050ti")`.
   - Must catch exceptions gracefully and report structured JSON / Rich outputs.
