# AGENTS.md: Protocolo Operacional para Benchmarks e Auto-Research

Este documento orienta agentes autônomos ao implementar, executar ou estender medições no diretório `benchmarks/`.

---

## 1. Regras Fundamentais de Engenharia

1. **Evidência Empírica Direta (Sem Simulações Vazias)**:
   - Todo benchmark deve interagir com o silício real ou arquivos de pesos reais (`Z:\models` ou caches locais).
   - Nunca gerar relatórios sem executar o binário nativo ou script correspondente.
2. **Medição Simultânea de Prefill e Decode**:
   - Sempre medir e registrar separadamente a vazão de prefill ($T_{prefill}$, tok/s prefill, TTFT) e a vazão de decode ($T_{decode}$, tok/s decode).
3. **Imutabilidade e Ledger Histórico**:
   - Toda execução bem-sucedida deve anexar uma nova linha ao `benchmarks/results.csv` e salvar o JSON correspondente em `benchmarks/reports/`.
   - O `results.csv` nunca deve ser apagado ou reescrito do zero.
4. **Isolamento de Plugins**:
   - Cada plugin deve herdar de `BaseBenchmarkPlugin` ou implementar a interface padrão `run() -> BenchmarkResult`.
   - Parâmetros de hardware (ex: `LITERT_GPU_INDEX`, CUDA Streams) devem ser restaurados ao estado inicial após a execução do plugin.
5. **Comandos Atômicos e Zero Polling**:
   - Executar comandos via `run_command` sem encadeamento de operadores (`;`, `&&`, `||`).
   - Confiar no wakeup reativo para compilações e execuções de longa duração.
