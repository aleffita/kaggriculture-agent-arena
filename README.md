# LiteRT-LM Exploration Suite

Ambiente de exploração, pesquisa e benchmarking para o **Google LiteRT-LM** (LiteRT Large Model Runtime) no Windows com suporte a configurações multi-GPU heterogêneas (**NVIDIA GeForce RTX 2060**, **NVIDIA GeForce GTX 1050 Ti** e **CPU**).

---

## 🌟 Principais Recursos

- **Controle Granular de GPU**: Seleção determinística de adaptador físico através do **DXGI Shim** (`dxgi_hook.dll`), contornando a limitação do Windows DXGI que entrega apenas o adaptador primário.
- **Suite de Benchmarks Automatizada**: Utilitário CLI para mensurar velocidade de prefill, decode (tokens/s), time-to-first-token e tempo de inicialização em todos os dispositivos.
- **Chat Interativo**: REPL de geração de texto com streaming em tempo real apontando para qualquer GPU ou CPU.
- **Gerenciamento Moderno com `uv`**: Ambiente isolado, reprodutível e com resolução ultrarrápida de dependências.
- **Agent-Friendly**: Estrutura documentada com [`AGENTS.md`](./AGENTS.md) e [`docs/`](./docs/) para desenvolvimento autônomo e colaborativo.

---

## 📊 Resultados de Benchmark (`gemma-4-E2B-it`)

| Dispositivo / Alvo | Arquitetura | VRAM | Prefill Speed | Decode Speed | Time to 1st Token |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA GeForce RTX 2060** | Turing (`sm_75`) | 6 GB | 149.35 t/s | **62.21 t/s** | 1.73 s |
| **NVIDIA GeForce GTX 1050 Ti** | Pascal (`sm_61`) | 4 GB | 98.02 t/s | **38.86 t/s** | 2.63 s |
| **CPU (Host System)** | x86_64 | 16 GB | 163.75 t/s | **14.88 t/s** | 1.63 s |

> A **GTX 1050 Ti** entrega **38.86 tokens/s**, sendo **2.61x mais rápida que a CPU** para inferência de linguagem via WebGPU/Direct3D 12 compute shaders.

---

## 🚀 Instalação e Inicialização Rápida

### 1. Clonar e Sincronizar o Ambiente via `uv`
```bash
uv sync
```

### 2. Verificar GPUs e Status do Shim
```bash
uv run litert-gpu
```

### 3. Executar Benchmarks
```bash
# Executar na GTX 1050 Ti
uv run litert-bench --target 1050ti

# Executar na RTX 2060
uv run litert-bench --target 2060

# Comparativo em todos os dispositivos
uv run litert-bench --target all
```

### 4. Chat Interativo com o Modelo
```bash
uv run litert-explore chat --target 1050ti
```

---

## 📁 Estrutura do Repositório

```text
litertlm-exploration/
├── pyproject.toml              # Definição do projeto e entrypoints CLI
├── README.md                   # Documentação principal
├── AGENTS.md                   # Protocolos de desenvolvimento para agentes de IA
├── src/litert_explore/         # Biblioteca Python principal
│   ├── cli.py                  # Entrypoints CLI (litert-explore, litert-bench, litert-gpu)
│   ├── gpu.py                  # Interceptor DXGI e listagem de GPUs
│   ├── engine.py               # Wrapper de alto nível do LiteRT-LM
│   ├── benchmark.py            # Suite de métricas e tabelas
│   └── dxgi_hook.dll           # Binário embutido do hook DXGI
├── shims/                      # Código C++ nativo do hook DXGI (dxgi_hook.cpp)
├── scripts/                    # Scripts PowerShell para automação rápida
├── docs/                       # Documentação técnica aprofundada
│   ├── architecture.md         # Análise da stack WebGPU/D3D12 vs CUDA
│   ├── benchmarks.md           # Relatórios e tabelas comparativas
│   └── gpu_routing.md          # Como chavear GPUs via CLI, env vars ou Python
├── experiments/                # Registros de experimentos empíricos
└── upstream/LiteRT-LM/         # Código-fonte oficial do Google LiteRT-LM
```

---

## 💡 Por que o DXGI Shim foi Necessário?

O LiteRT-LM no Windows utiliza **WebGPU via Google Dawn sobre Direct3D 12 (D3D12)** com compilação de shaders HLSL para DXIL via **DirectXShaderCompiler (DXC)**. 

Variáveis como `CUDA_VISIBLE_DEVICES` não afetam o DirectX. Para solucionar isso sem depender de compilação Bazel pesada, criamos um hook nativo de vtable (`shims/dxgi_hook.cpp`) que redireciona o `Adapter 0` para o índice desejado e esconde adaptadores secundários durante a inicialização do Dawn.

Consulte [`docs/architecture.md`](./docs/architecture.md) para detalhes completos da engenharia reversa e do funcionamento da vtable.
