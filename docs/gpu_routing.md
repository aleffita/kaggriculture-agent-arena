# Guia de Roteamento de GPU e Uso do DXGI Shim

## 1. Métodos de Seleção de Dispositivo

O projeto disponibiliza 3 formas de selecionar a GPU:

### Método A: Linha de Comando (CLI)
```bash
# Executar benchmark na GTX 1050 Ti
uv run litert-bench --target 1050ti

# Executar benchmark na RTX 2060
uv run litert-bench --target 2060

# Executar benchmark comparativo em todos os dispositivos
uv run litert-bench --target all
```

### Método B: Variável de Ambiente
Defina `$env:LITERT_GPU_INDEX` antes de rodar qualquer comando ou script:
```powershell
# Forçar GTX 1050 Ti
$env:LITERT_GPU_INDEX = "1"
litert-lm benchmark --from-huggingface-repo=litert-community/gemma-4-E2B-it-litert-lm gemma-4-E2B-it.litertlm --backend gpu

# Restaurar RTX 2060 (padrão)
Remove-Item env:LITERT_GPU_INDEX
```

### Método C: Python Context Manager
```python
from litert_explore import select_gpu, LiteRtModelRunner

# Executa na GTX 1050 Ti
with select_gpu("1050ti"):
    runner = LiteRtModelRunner(backend="gpu")
    print(runner.generate("Olá, quem é você?"))

# Executa na RTX 2060
with select_gpu("2060"):
    runner = LiteRtModelRunner(backend="gpu")
    print(runner.generate("Olá, quem é você?"))
```
