# AGENTS.md: Operational Protocol and Repository Guide

Welcome, Agent. This repository, `litertlm-exploration`, is an experimental environment designed for researching, benchmarking, and developing around **Google LiteRT-LM** (LiteRT Large Model runtime) on multi-GPU Windows configurations (specifically NVIDIA GeForce RTX 2060 + GTX 1050 Ti + CPU).

---

## 1. Core Architectural Truths

1. **Substrato Unificado de Inferência Heterogênea (`src/litert_explore/hpc_engine/unified_runtime`)**:
   - Um único motor de produção em C++/CUDA/D3D12 que unifica:
     * **CED Ring Dual-GPU**: GPU 1 (GTX 1050 Ti) como Causal Encoder / Draft + GPU 0 (RTX 2060) como Generative Decoder / Main + Pinned Host DMA Ring para transmissão contínua do estado de fronteira $h_{boundary}$.
     * **Roteador MoE Espacial com Poda BVH (RT Cores Turing SM 7.5 / D3D12 Tier 1.1)**: Árvores AABB hierárquicas que podam 62.5% a 95% do espaço de especialistas em hardware gráfico antes da avaliação nos Tensor Cores.
     * **Tiered Engram Memory**: 256 MB Hot VRAM Ring + 512 MB Host Staging + Overlapped Unbuffered Direct NVMe I/O (`Z:\models`) com prefetch orientado por grafos de transição, garantindo **ZERO stalls de SSD**.
     * **Warp-Shuffle Adder Tree**: Decodificação ternária / 1.58-bit pura em registradores inteiros (PTQ1_0) sem dequantização em ponto flutuante.
     * **KV-Cache Persistido em Disco & Recomputação da Residual Stream** (estilo Chris Hay / DwarfStar4).
2. **WebGPU / Direct3D 12 Execution (NOT CUDA)** para LiteRT-LM:
   - LiteRT-LM oficial roda sobre Google Dawn via D3D12 (`d3d12.dll`). O direcionamento de adaptadores é gerenciado pelo DXGI Shim (`dxgi_hook.dll`).
   - `CUDA_VISIBLE_DEVICES` é ignorado pelo subsistema DirectX/D3D12.

---

## 2. Princípio SDLC Fundamental: Construção vs. Avaliação

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  CONSTRUÇÃO (Runtime Unificado de Produção)                              │
│  src/litert_explore/hpc_engine/unified_runtime.cu (.exe)                 │
│  -> Código canônico onde vivem todos os kernels, ASICs e teses.          │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ exercitado por
┌────────────────────────────────────▼─────────────────────────────────────┐
│  AVALIAÇÃO (Suíte de Benchmarks & Auto-Research)                         │
│  benchmarks/runner.py + benchmarks/plugins/*.py                          │
│  -> Avaliadores externos ("a prova"). Medem Prefill e Decode no runtime, │
│     geram JSONs em reports/ e anexam métricas ao results.csv.            │
└──────────────────────────────────────────────────────────────────────────┘
                                     │ diagnósticos isolados
┌────────────────────────────────────▼─────────────────────────────────────┐
│  SONDAS DE DIAGNÓSTICO FÍSICO                                            │
│  probes/*.cu, probes/*.cpp                                               │
│  -> Sondas atômicas de exploração de silício (curvas PCIe, D3D12 DXR).    │
└──────────────────────────────────────────────────────────────────────────┘
```

- **PROIBIÇÃO**: Nunca colocar implementações de modelos ou kernels de computação dentro de `benchmarks/plugins/`. Os plugins de benchmark são estritamente **avaliadores**.
- **PROMOÇÃO CONTÍNUA**: Toda nova otimização ou técnica validada deve ser promovida e integrada diretamente ao `unified_runtime`.
- **DIRETIVA OPERACIONAL: SMOKE MODE COMO PADRÃO**:
  * Durante o desenvolvimento e iteração contínua, o agente **DEVE SEMPRE** rodar os benchmarks no modo smoke (`--mode smoke`).
  * A baseline de progresso é sempre aferida sob o modo smoke para garantir ciclos ágeis de validação.
  * O modo completo (`--mode full`) é reservado para fechamentos de ciclo ou avaliações de release formal.

---

## 3. Environment & Tooling Disciplines

- **Package Manager**: **Always use `uv`** (`uv run`, `uv add`, `uv sync`, `uv tool`). Never use raw `pip` or create manual virtual environments outside `uv`.
- **Command Execution**: Execute **single, atomic commands** synchronously. Never chain commands with semicolons (`;`) or logical operators (`&&`, `||`) in PowerShell.
- **Zero Polling Directive**: Never poll tasks in a loop or schedule artificial timers. Rely on reactive notification for background operations.
- **Communication Ceiling**: Keep conversational responses concise (2 to 4 sentences, BLUF). Put deep technical documentation and matrices in markdown artifacts in the brain.

---

## 4. Repository Topology

```text
d:/workdir/litertlm-exploration/
├── pyproject.toml              # UV-managed Python project definition & entrypoints
├── README.md                   # Human-facing project overview & quickstart
├── AGENTS.md                   # This agent guidance document
├── .gitignore                  # Git exclusions for models, venvs, and build outputs
├── src/
│   └── litert_explore/         # Python package
│       ├── __init__.py         # Package exports
│       ├── cli.py              # CLI entrypoints (litert-explore, litert-bench, litert-gpu, litert-hpc)
│       ├── hpc_runtime.py      # Orquestrador do motor HPC unificado
│       ├── hpc_engine/         # Implementação C++/CUDA do Runtime de Produção
│       │   ├── unified_runtime.cu (.exe)  # Motor Unificado Heterogêneo
│       │   └── ...
│       ├── gpu.py              # GPU listing, detection, and DXGI hook installation
│       └── dxgi_hook.dll       # Bundled DXGI vtable hook binary
├── benchmarks/                 # Suíte Modular de Avaliação
│   ├── README.md               # Especificação dos benchmarks
│   ├── AGENTS.md               # Protocolo operacional da suíte
│   ├── runner.py               # Orquestrador CLI (uv run litert-autoresearch)
│   ├── results.csv             # Ledger histórico cumulativo
│   ├── reports/                # Relatórios JSON persistidos
│   └── plugins/                # Avaliadores externos desacoplados
│       ├── base.py             # Interfaces BaseBenchmarkPlugin e BenchmarkResult
│       ├── prefill_decode_eval.py # Avaliador de Prefill e Decode
│       ├── bvh_router_eval.py     # Avaliador do Roteador Espacial BVH
│       └── pcie_channel_eval.py   # Avaliador do Canal Pinned DMA
├── probes/                     # Sondas diagnósticas de baixo nível
│   ├── bench_pcie_payload_curve.cu
│   └── bench_bvh_moe_and_asics.cu
└── shims/
    ├── dxgi_hook.cpp           # C++ source code for the DXGI vtable interceptor
    └── dxgi_hook.dll           # Compiled 64-bit DLL
```

---

## 5. Standard Operational Commands

### Running Production Unified HPC Engine
```powershell
# Execução direta do motor unificado
uv run litert-hpc --mode unified --tokens 30

# Invocação direta do binário compilado nativo com telemetria JSON
src/litert_explore/hpc_engine/unified_runtime.exe --model moe --prompt-len 128 --tokens 30 --json
```

### Running Evaluation Benchmarks
```powershell
# Execução de todos os avaliadores registrados
uv run litert-autoresearch --all

# Listagem de avaliadores
uv run litert-autoresearch --list

# Execução individual de um avaliador
uv run litert-autoresearch --plugin prefill_decode_eval
```

### Hardware Diagnostics
```powershell
uv run litert-gpu
```
