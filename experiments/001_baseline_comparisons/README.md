# Experimento 001: Baseline Comparativo de Dispositivos (RTX 2060 vs GTX 1050 Ti vs CPU)

- **Data**: 2026-09-29
- **Modelo**: `gemma-4-E2B-it.litertlm`
- **Tamanho do Arquivo**: ~2.4 GiB
- **Objetivo**: Estabelecer a linha de base de velocidade de inferência (prefill e decode) comparando a GPU primária (RTX 2060), a GPU secundária (GTX 1050 Ti) e a CPU do sistema.

## Métricas Registradas

| Métrica | RTX 2060 | GTX 1050 Ti | CPU |
| :--- | :--- | :--- | :--- |
| **Prefill Speed** | 149.35 t/s | 98.02 t/s | 163.75 t/s |
| **Decode Speed** | 62.21 t/s | 38.86 t/s | 14.88 t/s |
| **Time to First Token (TTFT)** | 1.73 s | 2.63 s | 1.63 s |
| **Tempo de Inicialização** | 6.82 s | 4.74 s | 0.68 s |

## Principais Conclusões

1. O Direct3D 12 compute pipeline compilou e executou com sucesso na arquitetura Pascal (`sm_61`), mesmo com a ausência de Tensor Cores dedicados.
2. A GTX 1050 Ti atingiu **38.86 tokens/s**, viabilizando conversações interativas com latência imperceptível.
3. O DXGI Shim interceptou e direcionou o runtime de forma 100% determinística.
