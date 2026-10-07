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
# Executar todos os avaliadores registrados contra o Unified Runtime
uv run litert-autoresearch --all

# Listar os avaliadores disponíveis
uv run litert-autoresearch --list

# Executar a avaliação completa de Prefill e Decode nos modelos MoE e Ternário
uv run litert-autoresearch --plugin prefill_decode_eval

# Executar a avaliação da aceleração de roteamento MoE via poda BVH (RT Cores)
uv run litert-autoresearch --plugin bvh_router_eval

# Executar a avaliação do canal de transporte DMA Pinned Memory inter-GPU
uv run litert-autoresearch --plugin pcie_channel_eval
```

---

## 📊 Estrutura do Diretório

```text
benchmarks/
├── README.md               # Este documento de especificação
├── AGENTS.md               # Protocolo operacional e regras de isolamento
├── results.csv             # Ledger tabular cumulativo de métricas
├── runner.py               # Orquestrador CLI modular
├── plugins/                # Avaliadores externos desacoplados
│   ├── base.py             # Classe base e dataclass BenchmarkResult
│   ├── prefill_decode_eval.py # Avaliador de Prefill e Decode
│   ├── bvh_router_eval.py     # Avaliador do Roteador Espacial BVH
│   └── pcie_channel_eval.py   # Avaliador do Canal Pinned DMA
└── reports/                # Relatórios JSON persistidos por execução
```
