# AGENTS.md: Operational Protocol and Repository Guide

Welcome, Agent. This repository, `litertlm-exploration`, is an experimental environment designed for researching, benchmarking, and developing around **Google LiteRT-LM** (LiteRT Large Model runtime) on multi-GPU Windows configurations (specifically NVIDIA GeForce RTX 2060 + GTX 1050 Ti + CPU).

---

## 1. Core Architectural Truths

1. **WebGPU / Direct3D 12 Execution (NOT CUDA)**:
   - LiteRT-LM on Windows uses **Google Dawn** over **Direct3D 12 (D3D12)**.
   - **DO NOT** use `CUDA_VISIBLE_DEVICES` to select GPUs. It is completely ignored by the D3D12 and DXGI subsystems.
2. **GPU Selection via DXGI Shim**:
   - To target the **GTX 1050 Ti (GPU 1)**, set `$env:LITERT_GPU_INDEX = "1"` or use the Python helper `select_gpu("1050ti")`.
   - To target the **RTX 2060 (GPU 0)**, unset `$env:LITERT_GPU_INDEX` (e.g., `Remove-Item env:LITERT_GPU_INDEX`).
   - The shim operates via `dxgi_hook.dll` (located in `shims/` and bundled in `src/litert_explore/`), which intercepts `IDXGIFactory` vtable calls (`EnumAdapters1` and `EnumAdapterByGpuPreference`).

---

## 2. Environment & Tooling Disciplines

- **Package Manager**: **Always use `uv`** (`uv run`, `uv add`, `uv sync`, `uv tool`). Never use raw `pip` or create manual virtual environments outside `uv`.
- **Command Execution**: Execute **single, atomic commands** synchronously. Never chain commands with semicolons (`;`) or logical operators (`&&`, `||`) in PowerShell.
- **Zero Polling Directive**: Never poll tasks in a loop or schedule artificial timers. Rely on reactive notification for background operations.
- **Communication Ceiling**: Keep conversational responses concise (2 to 4 sentences). Put deep technical documentation, logs, or analysis matrices in markdown artifacts or in `docs/`.

---

## 3. Repository Topology

```text
d:/workdir/litertlm-exploration/
├── pyproject.toml              # UV-managed Python project definition & entrypoints
├── README.md                   # Human-facing project overview & quickstart
├── AGENTS.md                   # This agent guidance document
├── .gitignore                  # Git exclusions for models, venvs, and build outputs
├── src/
│   └── litert_explore/         # Python package
│       ├── __init__.py         # Package exports
│       ├── cli.py              # CLI entrypoints (litert-explore, litert-bench, litert-gpu)
│       ├── gpu.py              # GPU listing, detection, and DXGI hook installation
│       ├── engine.py           # High-level LiteRT engine runner and streaming interface
│       ├── benchmark.py        # Automated benchmarking suite across devices
│       └── dxgi_hook.dll       # Bundled DXGI vtable hook binary
├── shims/
│   ├── dxgi_hook.cpp           # C++ source code for the DXGI vtable interceptor
│   └── dxgi_hook.dll           # Compiled 64-bit DLL
├── scripts/
│   ├── bench_1050ti.ps1        # Quick benchmark on GTX 1050 Ti
│   ├── bench_2060.ps1          # Quick benchmark on RTX 2060
│   ├── bench_cpu.ps1           # Quick benchmark on CPU
│   └── compile_dxgi_shim.ps1   # Recompiles dxgi_hook.cpp using local MSVC toolchain
├── docs/
│   ├── architecture.md         # Detailed runtime architecture & execution flow
│   ├── benchmarks.md           # Benchmark comparison tables (RTX 2060 vs 1050 Ti vs CPU)
│   └── gpu_routing.md          # Multi-GPU routing options and code examples
├── experiments/                # Research experiments, logs, custom prompts, and quantization trials
│   └── 001_baseline_comparisons/ # Baseline benchmarks for Gemma-4-E2B-it
└── upstream/
    └── LiteRT-LM/              # Cloned upstream Google LiteRT-LM repository (C++ source, Bazel configs)
```

---

## 4. Standard Operational Commands

### Running Benchmarks
```powershell
# Benchmark on GTX 1050 Ti
uv run litert-bench --target 1050ti

# Benchmark on RTX 2060
uv run litert-bench --target 2060

# Full comparative benchmark across all devices
uv run litert-bench --target all
```

### Interactive Model Execution (Chat)
```powershell
uv run litert-explore chat --target 1050ti
```

### Diagnosing Hardware & Shim Status
```powershell
uv run litert-gpu
```

### Recompiling the DXGI Shim (if modified)
```powershell
powershell -File scripts/compile_dxgi_shim.ps1
```

---

## 5. Experiment Protocol

When conducting new experiments:
1. Create a dedicated folder in `experiments/<00X_experiment_name>/`.
2. Document the hypothesis, parameters (quantization, context length, steps per sync), and findings in `experiments/<00X_experiment_name>/README.md`.
3. Save raw output logs in the experiment folder for reproducibility.
