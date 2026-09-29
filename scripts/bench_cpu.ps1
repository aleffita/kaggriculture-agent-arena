# Benchmark no backend de CPU
$ErrorActionPreference = "Stop"

Write-Host "Configurando ambiente para benchmark de CPU..." -ForegroundColor Cyan
uv run litert-bench --target cpu
