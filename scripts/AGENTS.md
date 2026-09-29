# AGENTS.md: Automation Scripts & Shell Utilities (`scripts/`)

This directory contains standalone PowerShell automation scripts and build utilities.

---

## 1. Execution Directives

- **Windows Shell Rigor**: All scripts must run cleanly in PowerShell 7 (`pwsh`) without unquoted expansions or unhandled exit codes.
- **Single Atomic Execution**: Scripts should focus on a single atomic operation (e.g. running a benchmark or compiling a native DLL).
- **GPU Targeting**: Scripts targeting the GTX 1050 Ti must explicitly set `$env:LITERT_GPU_INDEX = "1"`.

---

## 2. Script Index

- `bench_1050ti.ps1`: Automated benchmark on NVIDIA GeForce GTX 1050 Ti.
- `bench_2060.ps1`: Automated benchmark on NVIDIA GeForce RTX 2060.
- `bench_cpu.ps1`: Automated benchmark on Host CPU.
- `compile_dxgi_shim.ps1`: Recompiles `shims/dxgi_hook.cpp` using the MSVC toolchain.
