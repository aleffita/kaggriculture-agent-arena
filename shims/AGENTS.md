# AGENTS.md: Native Shims & C++ Vtable Interceptors (`shims/`)

This directory contains native 64-bit DLLs and C++ source code for intercepting Windows DXGI and Direct3D 12 adapter discovery.

---

## 1. Native Interceptor Axioms

- **Target Architecture**: Windows x64 (MSVC `cl.exe`).
- **Compilation Toolchain**: Visual Studio 2022/2026 MSVC HostX64/x64 compiler. Recompile using `powershell -File scripts/compile_dxgi_shim.ps1`.
- **Vtable Slots Intercepted**:
  - Slot 7: `IDXGIFactory::EnumAdapters`
  - Slot 12: `IDXGIFactory1::EnumAdapters1`
  - Slot 29: `IDXGIFactory6::EnumAdapterByGpuPreference`

---

## 2. Invariant Rules for Modifying Shims

1. Do NOT introduce external C++ runtime dependencies (`/MD` or `/MT` statically linked runtime is required to ensure portability across processes).
2. Always test compilation via `scripts/compile_dxgi_shim.ps1` before committing changes to `dxgi_hook.cpp`.
3. Synchronize `shims/dxgi_hook.dll` with `src/litert_explore/dxgi_hook.dll`.
