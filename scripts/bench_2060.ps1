# Benchmark na NVIDIA GeForce RTX 2060 (GPU 0)
$ErrorActionPreference = "Stop"

Write-Host "Configurando ambiente para RTX 2060..." -ForegroundColor Cyan
Remove-Item env:LITERT_GPU_INDEX -ErrorAction SilentlyContinue

uv run litert-bench --target 2060
