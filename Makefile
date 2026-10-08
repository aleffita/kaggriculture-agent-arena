# ==============================================================================
# Unified Heterogeneous CED Runtime & Evaluation Suite Makefile (Windows Native)
# Hardware: NVIDIA GeForce RTX 2060 (Turing SM 7.5) + GTX 1050 Ti (Pascal SM 6.1)
# ==============================================================================

.DEFAULT_GOAL := help

.PHONY: help install setup build-engine server-start server-stop server-status \
        bench-smoke bench-full bench-ext-code bench-ext-humaneval bench-ext-mbpp \
        bench-ext-math bench-ext-context bench-ext-all bench-ext-wsl \
        probes gpu-info clean-cache plots clean

# ------------------------------------------------------------------------------
# 1. Ajuda e Documentação
# ------------------------------------------------------------------------------
help:
	@echo ==============================================================================
	@echo  Unified Heterogeneous CED Runtime - Comandos de Gerenciamento
	@echo ==============================================================================
	@echo  Ambiente:
	@echo    make install              - Sincroniza ambiente uv e instala pacotes em modo editavel
	@echo    make gpu-info             - Exibe topologia das GPUs e configuracao do DXGI Hook
	@echo    make build-engine         - Compila o unified_runtime.cu com NVCC (SM 7.5 + SM 6.1)
	@echo.
	@echo  Servidor Live Headless:
	@echo    make server-start         - Inicia o servidor litert-web (0.0.0.0:8765)
	@echo    make server-stop          - Encerra o processo ativo na porta 8765
	@echo    make server-status        - Verifica se a porta 8765 esta online
	@echo.
	@echo  Suite Interna de Auto-Research:
	@echo    make bench-smoke          - Roda todos os 13 avaliadores internos em modo SMOKE
	@echo    make bench-full           - Roda todos os 13 avaliadores internos em modo FULL
	@echo    make plots                - Gera graficos executivos dark mode dos relatorios
	@echo.
	@echo  Suite Externa (lm-eval / Meta CLM ContextBench / HumanEval / MBPP):
	@echo    make bench-ext-code       - Roda avaliacao de codigo (HumanEval + MBPP) via WSL2
	@echo    make bench-ext-humaneval  - Roda apenas HumanEval (pass@1) via WSL2
	@echo    make bench-ext-mbpp       - Roda apenas MBPP via WSL2
	@echo    make bench-ext-math       - Roda raciocinio matematico (GSM8K + Minerva Math)
	@echo    make bench-ext-context    - Roda Meta CLM ContextBench (Needle, Sudoku, KVStore, Log)
	@echo    make bench-ext-all        - Roda a suite externa completa (math, code, context)
	@echo.
	@echo  Manutencao e Silicio:
	@echo    make clean-cache          - Purgar KV-Cache persistido em NVMe (Z:\models)
	@echo    make probes               - Executa sondas de barramento PCIe e aceleracao BVH
	@echo    make clean                - Remove artefatos temporarios e caches de build
	@echo ==============================================================================


# ------------------------------------------------------------------------------
# 2. Instalacao e Ambiente
# ------------------------------------------------------------------------------
install: setup

setup:
	uv sync
	uv pip install -e .
	@if exist Z:\repos\context-language-models uv pip install -e Z:\repos\context-language-models

gpu-info:
	uv run litert-gpu

# ------------------------------------------------------------------------------
# 3. Compilacao do Motor C++/CUDA
# ------------------------------------------------------------------------------
build-engine:
	@echo [+] Compilando Unified CED Runtime em C++/CUDA (SM 7.5 + SM 6.1)...
	nvcc -O3 -std=c++17 \
		-gencode arch=compute_75,code=sm_75 \
		-gencode arch=compute_61,code=sm_61 \
		src/litert_explore/hpc_engine/unified_runtime.cu \
		-o src/litert_explore/hpc_engine/unified_runtime.exe \
		-ld3d12 -ldxgi
	@echo [OK] unified_runtime.exe compilado com sucesso!

# ------------------------------------------------------------------------------
# 4. Servidor Live Headless (Porta 8765)
# ------------------------------------------------------------------------------
server-start:
	@echo [+] Iniciando litert-web headless em segundo plano...
	@powershell -NoProfile -Command "Start-Process uv -ArgumentList 'run', 'litert-web', '--host', '0.0.0.0', '--port', '8765', '--headless' -WindowStyle Hidden"
	@echo [OK] Servidor despachado na porta 8765.

server-stop:
	@echo [+] Encerrando listeners na porta 8765...
	@powershell -NoProfile -Command "$$c = Get-NetTCPConnection -LocalPort 8765 -ErrorAction SilentlyContinue; if ($$c) { Stop-Process -Id ($$c.OwningProcess | Select-Object -Unique) -Force; Write-Host '[OK] Encerrado.' } else { Write-Host '[!] Nenhum ativo.' }"

server-status:
	@uv run python -c "import urllib.request, json; print('[OK] Online:', [m['id'] for m in json.loads(urllib.request.urlopen('http://127.0.0.1:8765/v1/models').read())['data']])"

# ------------------------------------------------------------------------------
# 5. Suíte Interna de Auto-Research
# ------------------------------------------------------------------------------
bench-smoke:
	uv run litert-autoresearch --all --mode smoke

bench-full:
	uv run litert-autoresearch --all --mode full

plots:
	uv run python benchmarks/plots.py

# ------------------------------------------------------------------------------
# 6. Suíte Externa (lm-eval + ContextBench via WSL POSIX Sandbox)
# ------------------------------------------------------------------------------
bench-ext-humaneval:
	wsl bash /mnt/d/workdir/litertlm-exploration/benchmarks/run_external_wsl.sh --tasks humaneval

bench-ext-mbpp:
	wsl bash /mnt/d/workdir/litertlm-exploration/benchmarks/run_external_wsl.sh --tasks mbpp

bench-ext-code:
	wsl bash /mnt/d/workdir/litertlm-exploration/benchmarks/run_external_wsl.sh --suite code

bench-ext-math:
	wsl bash /mnt/d/workdir/litertlm-exploration/benchmarks/run_external_wsl.sh --suite math

bench-ext-context:
	wsl bash /mnt/d/workdir/litertlm-exploration/benchmarks/run_external_wsl.sh --tasks contextbench

bench-ext-all:
	wsl bash /mnt/d/workdir/litertlm-exploration/benchmarks/run_external_wsl.sh --suite all

bench-ext-wsl: bench-ext-all

# ------------------------------------------------------------------------------
# 7. Manutenção de Silício e Limpeza
# ------------------------------------------------------------------------------
clean-cache:
	@uv run python -c "import urllib.request; urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8765/v1/runtime/reset', method='POST'), timeout=2); print('[OK] Memoria purgada via API.')"

probes:
	@if exist probes\bench_pcie_payload_curve.exe probes\bench_pcie_payload_curve.exe
	@if exist probes\bench_bvh_moe_and_asics.exe probes\bench_bvh_moe_and_asics.exe

clean:
	@uv run python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; print('[OK] Limpeza de caches concluida.')"
