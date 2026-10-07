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
4. **Imutabilidade e Ledger Histórico**:
   - Toda execução bem-sucedida deve anexar uma nova linha ao `benchmarks/results.csv` e salvar o JSON correspondente em `benchmarks/reports/`.
   - O `results.csv` nunca deve ser apagado ou reescrito do zero.
5. **Isolamento de Plugins**:
   - Cada plugin deve herdar de `BaseBenchmarkPlugin` e implementar a interface padrão `run() -> BenchmarkResult`.
6. **Comandos Atômicos e Zero Polling**:
   - Executar comandos via `run_command` de forma atômica e síncrona.
   - Nunca realizar polling ativo em loops; confiar no wakeup reativo.
