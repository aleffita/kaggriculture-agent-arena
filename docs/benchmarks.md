# Matriz de Desempenho e Benchmarks

Modelo avaliado: **`gemma-4-E2B-it.litertlm`** (2.4 GiB, quantizado).
Parâmetros de teste: Prefill: 256 tokens, Decode: 256 tokens, Max tokens: 4096.

## 1. Tabela Comparativa de Resultados

| Dispositivo / Alvo | Arquitetura | VRAM / RAM | Prefill Speed | Decode Speed | Time to 1st Token | Tempo de Inicialização |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA GeForce RTX 2060** | Turing (`sm_75`) | 6 GB GDDR6 | 149.35 t/s | **62.21 t/s** | 1.73 s | 6.82 s |
| **NVIDIA GeForce GTX 1050 Ti** | Pascal (`sm_61`) | 4 GB GDDR5 | 98.02 t/s | **38.86 t/s** | 2.63 s | 4.74 s |
| **CPU (Host System)** | x86_64 | 16 GB DDR4 | 163.75 t/s | **14.88 t/s** | 1.63 s | 0.68 s |

## 2. Análise Técnica dos Resultados

1. **Eficiência da GTX 1050 Ti**:
   - A GTX 1050 Ti atingiu **38.86 tokens/s** no decode, superando a CPU por um fator de **2.61x**.
   - Em comparação com a RTX 2060, a 1050 Ti entrega cerca de **62.5%** da velocidade de decode, o que é um resultado surpreendente considerando que a arquitetura Pascal não possui Tensor Cores e tem uma taxa de FP16 significativamente inferior.
2. **Tempo de Pré-preenchimento (Prefill)**:
   - Na CPU, o prefill é ligeiramente mais rápido para prompts pequenos (163 t/s) devido à largura de banda do cache do host, mas a geração subsequente (decode) colapsa para 14.88 t/s devido ao gargalo de memória DRAM.
   - Na GPU, o decode se mantém sustentado e estável durante toda a geração.
