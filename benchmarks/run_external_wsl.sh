#!/usr/bin/env bash
set -e

# Assegurar que o uv do usuário esteja no PATH do WSL
export PATH="$HOME/.local/bin:$PATH"
export HF_ALLOW_CODE_EVAL="1"
export PYTHONIOENCODING="utf-8"
export UV_PROJECT_ENVIRONMENT="/tmp/wsl_bench_venv"

# Resolução dinâmica do IP do host Windows a partir da rota default do WSL2
HOST_IP=$(ip route show default | awk '{print $3}')
if [ -z "$HOST_IP" ]; then
    HOST_IP="172.27.96.1"
fi

BASE_URL="http://${HOST_IP}:8765/v1/chat/completions"
echo "[WSL Benchmark] Conectando ao host Windows em: $BASE_URL"

ARGS=("$@")
if [ $# -eq 0 ]; then
    ARGS=(--suite all)
fi

# Execução limpa e isolada no WSL (sem alterar o .venv nativo do Windows)
uv run --python 3.12 \
       --no-project \
       --with "lm-eval[api,math]" \
       --with "antlr4-python3-runtime==4.11" \
       --with "math_verify" \
       --with "sympy" \
       --with "pillow" \
       --with "evaluate" \
       --with "rich" \
       --with "/mnt/z/repos/context-language-models" \
       --with "tiktoken" \
       --with "pyyaml" \
       python /mnt/d/workdir/litertlm-exploration/benchmarks/run_external.py \
       --base-url "$BASE_URL" \
       "${ARGS[@]}"

