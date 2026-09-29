# Benchmark na NVIDIA GeForce GTX 1050 Ti (GPU 1)
$ErrorActionPreference = "Stop"

Write-Host "Configurando ambiente para GTX 1050 Ti via DXGI Shim..." -ForegroundColor Cyan
$env:LITERT_GPU_INDEX = "1"

uv run litert-bench --target 1050ti
