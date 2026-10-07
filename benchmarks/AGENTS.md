# AGENTS.md: Protocolo Operacional para Benchmarks e Auto-Research

Este documento orienta agentes autônomos ao implementar, executar ou estender medições no diretório `benchmarks/`.

---

## 1. Regras Fundamentais de Engenharia

1. **Separação Rígida: Avaliador vs. Construção (A Prova vs. A Matéria)**:
   - Os plugins em `benchmarks/plugins/` são estritamente **avaliadores** desacoplados.
   - NUNCA colocar kernels CUDA/C++, lógica de modelo ou pipelines dentro de `benchmarks/plugins/`.
   - Toda implementação arquitetural vive no motor unificado (`src/litert_explore/hpc_engine/unified_runtime`). Os plugins apenas executam o runtime unificado com diferentes configurações, coletam métricas e validam o comportamento.
2. **Evidência Empírica Direta (Sem Simulações Vazias)**:
   - Todo benchmark deve interagir com o silício real e arquivos de pesos reais (`Z:\models` ou caches locais).
   - Nunca inventar métricas ou preencher tabelas sem execução fática.
3. **Medição Simultânea de Prefill e Decode**:
   - Sempre medir e registrar separadamente a vazão de prefill ($T_{prefill}$, tok/s prefill, TTFT) e a vazão de decode ($T_{decode}$, tok/s decode).
4. **Persistência via JSON Unificado (Sem Ruído)**:
   - Toda execução da suíte gera um relatório JSON estruturado e autocontido em `benchmarks/reports/run_<timestamp>_suite_<mode>.json`.
   - O JSON contém a totalidade dos dados das medições, metadados do ambiente de hardware e desvios padrão, sem compressão de dados.
5. **Formatação Numérica Padronizada (Zero Separadores de Milhar)**:
   - **NUNCA** usar pontos ou vírgulas para separar milhares (ex: `4969` ou `4969.76`, nunca `4.969` ou `4,969`).
   - O ponto (`.`) é exclusivamente o separador decimal.
6. **Tríade de Modelos e Comparação com Runtimes Originais**:
   - Os benchmarks devem sempre cobrir os 3 modelos do projeto:
     * `gpt-oss-20b` (MoE)
     * `bonsai-27b` (Ternário)
     * `gemma-4-E2B-it` (LiteRT Base)
   - Comparar o `unified-ced` com os runtimes originais:
     * `bonsai-27b` $\to$ `llama-cpp-prism` (`Z:\workspaces\llama-cpp-prism`)
     * `gpt-oss-20b` $\to$ `llama.cpp` stock
     * `gemma-4-E2B-it` $\to$ `litert-d3d12` stock
7. **Isolamento de Plugins**:
   - Cada plugin deve herdar de `BaseBenchmarkPlugin` e implementar a interface padrão `run(mode="smoke") -> BenchmarkSuiteResult`.
8. **Diretiva de Smoke Mode Obrigatório**:
   - Durante o trabalho e desenvolvimento contínuo, os benchmarks **DEVEM SEMPRE** ser executados no modo smoke (`--mode smoke`, padrão da CLI).
   - O modo smoke define a baseline canônica de iteração rápida do projeto.
   - O modo full (`--mode full`) só deve ser executado mediante solicitação explícita ou em releases.
9. **Comandos Atômicos e Zero Polling**:
   - Executar comandos via `run_command` de forma atômica e síncrona.
   - Nunca realizar polling ativo em loops; confiar no wakeup reativo.
