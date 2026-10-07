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
# Executar todos os benchmarks em modo smoke (padrão de desenvolvimento rápido)
uv run litert-autoresearch --all --mode smoke

# Listar os benchmarks disponíveis
uv run litert-autoresearch --list

# Executar apenas o benchmark de Throughput & Latência em grade cartesiana
uv run litert-autoresearch --benchmark throughput --mode smoke

# Executar a avaliação de corretude de código (OpenAI HumanEval pass@1)
uv run litert-autoresearch --benchmark humaneval --mode smoke

# Executar a avaliação de Perplexidade e estabilidade de distribuição
uv run litert-autoresearch --benchmark perplexity --mode smoke

# Execução formal completa (modo full)
uv run litert-autoresearch --all --mode full
```

---

## 📊 Estrutura do Diretório

```text
benchmarks/
├── README.md               # Este documento de especificação
├── AGENTS.md               # Protocolo operacional e regras de isolamento
├── results.csv             # Ledger tabular cumulativo e granular de medições
├── runner.py               # Orquestrador CLI modular
├── plugins/                # Benchmarks e avaliadores agnósticos desacoplados
│   ├── base.py             # Interfaces BaseBenchmarkPlugin e BenchmarkSuiteResult
│   ├── throughput_bench.py # Throughput em grade (P x N x B) com stddev
│   ├── humaneval_bench.py  # Functional correctness (pass@1 em Python)
│   └── perplexity_bench.py # Perplexidade e cross-entropy loss
└── reports/                # Relatórios JSON estruturados por execução
```
