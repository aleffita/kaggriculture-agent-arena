# 🚀 HPC Benchmarks & Auto-Research Suite

Bem-vindo ao subsistema de benchmarks modulares e pesquisa empírica automatizada de `litertlm-exploration`.

Inspirado na metodologia de **Auto-Research** de Andrej Karpathy e nas melhores práticas de Clean Code e SDLC, este subsistema foi projetado para permitir a execução, adição e expansão modular de sondas e benchmarks de silício heterogêneo (NVIDIA RTX 2060 + GTX 1050 Ti + Host CPU + NVMe Direct Storage).

---

## 📐 Filosofia Arquitetural

1. **Abordagem Plugin-Based (Modular)**:
   - Cada benchmark é um plugin desacoplado localizado em `benchmarks/plugins/`.
   - Novos benchmarks (ex: novos kernels de computação, codecs de compressão, schedulers) podem ser adicionados como plugins isolados sem alterar o runner principal.
2. **Medição Obrigatória Dupla (Prefill & Decode)**:
   - Toda medição sobre modelos de linguagem reporta explicitamente a taxa de **prefill** (processamento de prompt, tokens/s, TTFT) e a taxa de **decode** (geração autoregressiva token a token, tokens/s, ms/token).
3. **Persistência Estruturada**:
   - Cada execução gera um relatório estruturado detalhado em formato JSON em `benchmarks/reports/run_<timestamp>_<plugin>.json`.
   - O arquivo cumulativo `benchmarks/results.csv` na raiz atua como um *ledger* histórico de progresso, registrando métricas comparativas ao longo da evolução do projeto.

---

## 🛠️ Como Executar

Utilizando o gerenciador padrão do projeto (`uv`):

```powershell
# Executar todos os plugins de benchmark registrados via CLI do projeto
uv run litert-autoresearch --all

# Listar os plugins disponíveis
uv run litert-autoresearch --list

# Executar apenas o sweep da curva de latência vs payload PCIe
uv run litert-autoresearch --plugin pcie_payload_curve

# Executar o benchmark de MoE Esparso Dual-GPU em Ring Streaming
uv run litert-autoresearch --plugin moe_dual_gpu_ring

# Executar a sondagem D3D12 DXR e o benchmark de poda BVH espacial do MoE
uv run litert-autoresearch --plugin bvh_moe_router
```

---

## 📊 Estrutura do Diretório

```text
benchmarks/
├── README.md               # Este documento de especificação
├── AGENTS.md               # Diretrizes para agentes autônomos
├── results.csv             # Ledger tabular cumulativo de métricas
├── runner.py               # Orquestrador CLI modular
├── plugins/                # Diretório de plugins de benchmark
│   ├── base.py             # Classe base e dataclass BenchmarkResult
│   ├── pcie_payload_curve.py
│   ├── moe_dual_gpu_ring.py
│   └── bvh_moe_router.py
└── reports/                # Relatórios JSON persistidos por execução
```
