# AGENTS.md: Core Package Directives (`src/litert_explore/`)

This directory contains the primary Python library supporting the LiteRT-LM exploration environment.

---

## 1. Architectural Invariants

- **Hardware Targeting**: Any function interacting with GPU initialization or adapter selection must route through `litert_explore.gpu.select_gpu()` or check `$env:LITERT_GPU_INDEX`.
- **Packaging Standard**: Managed via `uv`. All modules must maintain clean type annotations and backwards compatibility with Python 3.12.
- **Zero Raw CUDA Calls**: Never introduce direct CUDA runtime dependencies. All hardware acceleration runs over Google Dawn (Direct3D 12 on Windows).

---

## 2. Module Responsibilities

- `gpu.py`: DXGI vtable interceptor installation, hardware adapter enumeration, and GPU isolation contexts.
- `engine.py`: High-level wrappers around `litert_lm.Engine`, streaming generation, and model cache resolution.
- `benchmark.py`: Standardized cross-device benchmark suites.
- `cli.py`: Command-line entry points (`litert-explore`, `litert-bench`, `litert-gpu`).
