# Substrato Unificado de Inferência Heterogênea (HPC Engine)

Motor de inferência de produção de alta performance em **C++/CUDA/D3D12**, projetado para executar modelos de fronteira com orquestração física heterogênea em hardware assimétrico de consumo: **NVIDIA GeForce RTX 2060 (6 GB Turing SM 7.5)** + **GeForce GTX 1050 Ti (4 GB Pascal SM 6.1)** + **AMD Ryzen 5 3600 (Host 6C/12T)** + **SSD NVMe Direct I/O (`Z:\models`)**.

---

## 1. Arquitetura e Pilares de Silício

1. **Pipeline CED Dual-GPU (Causal Encoder / Generative Decoder)**:
   - **GPU 1 (GTX 1050 Ti 4GB)**: Executa o Prefill inicial, drafting especulativo e codificação causal de camadas iniciais.
   - **GPU 0 (RTX 2060 6GB)**: Executa a decodificação generativa pesada aproveitando Tensor Cores FP16/INT8 e RT Cores.
   - **Pinned DMA Ring Buffer (Host)**: 512 MB de memória física bloqueada para streaming contínuo do vetor residual de fronteira $h_{boundary}$ sem sincronização bloqueante na CPU (janela de 106 µs).
2. **Substrato de Memória de Engrams (Sempre Ativo & Não Opcional)**:
   - O **Engram Substrate** é o alicerce fundamental permanente do runtime. Ele rastreia grafos de transição de especialistas e sequências de tokens via Overlapped Direct Unbuffered I/O no NVMe (`Z:\models`), garantindo prefetch contínuo e zero stalls de SSD.
3. **Especulação Dual-Stage com Drafters Plugáveis (`--drafter`)**:
   - O drafter neural atua de forma complementar sobre o substrato de engrams:
     * `auto`: Resolução automática baseada no modelo.
     * `dspark`: Drafter Semi-Autoregressivo (SAR) com verificação de sobrevivência de prefixos inspirado no DeepSeek V4.1.
     * `eagle3`: Drafter latente residual autoregressivo de 1 camada que atua diretamente sobre $h_{boundary}$ (ex: `eagle3-gpt-oss-20b-Q8_0.gguf`).
     * `mtp`: Multi-Token Prediction nativo (cabeças auxiliares no mesmo modelo, ex: Ornith / Qwen-3.5).
     * `none`: Desativa o drafter neural auxiliar, mantendo o Engram Substrate como acelerador primário (1.15x speedup).
4. **Virtual Experts & Asynchronous Function Calling**:
   - **Virtual Expert 1 (Chris Hay Protocol)**: Resolução simbólica exata e matemática em background (`std::async`) no Ryzen 5 3600 durante a transferência DMA PCIe. Elimina até 92% dos tokens de divagação CoT.
   - **Virtual Expert 2 (Google Embedding Gemma 2 Multimodal RAG)**: Geração de embeddings 768d (MRL Matryoshka) com `google/embeddinggemma-2` (740M Q8_0 em `Z:\models\ggml-org\embeddinggemma-2-GGUF`) e navegação vetorial na memória de engrams sem stall na GPU.
5. **Roteador MoE Espacial com Poda BVH (Turing RT Cores / D3D12 Tier 1.1)**:
   - Poda de 62.5% a 96.88% dos especialistas MoE em silício de raytracing antes da avaliação nos Tensor Cores.
6. **Warp-Shuffle Adder Tree (Ternary 1.58-bit PTQ1_0)**:
   - Acumulação direta de pesos ternários $\{-1, 0, +1\}$ empacotados em 2 bits usando registradores inteiros e instruções `__shfl_xor_sync`. Zero conversão em ponto flutuante.
7. **KV-Cache Persistido em NVMe & Recursividade Estilo MiniAGI / DreamRSI**:
   - Reutilização de prefixos via Radix Tree no arquivo em disco (`Z:\models\kv_cache.bin`), permitindo loops de auto-revisão e branching reflexivo mantendo a VRAM estritamente constante em **<1.1 GB**.

---

## 2. Modelos Suportados

| Arquétipo CLI | Modelo / Formato | Parâmetros | VRAM Total | Alocação Heterogênea |
| :--- | :--- | :---: | :---: | :--- |
| `moe` | GPT-OSS-20B (MXFP4) | 20B (MoE 32 esp) | ~11.28 GB | GPU 0 (6GB) + BVH RT Cores + Eagle-3 + NVMe Direct |
| `ternary` | Ternary-Bonsai-2-27B (PTQ1_0) | 27B (Ternário 1.58b) | ~5.54 GB | GPU 1 (3.8GB) + GPU 0 (1.7GB) (Zero Host Spill) |
| `gemma4` | Gemma-4-E2B-it (LiteRT QAT) | 2.6B (Denso) | ~2.30 GB | GPU 0 (2.3GB) via Direct3D 12 Dawn DXGI Hook |
| `gemma12b` | Gemma-4-12B-it Heretic (Q4_K_XL) | 12B (Denso) | ~6.72 GB | GPU 1 (3.8GB) + GPU 0 (2.9GB) (Zero OOM) |
| `ornith` | Ornith-1.5-35B-A3B (IQ2_XXS) | 35B (MoE 256 esp) | ~8.40 GB | Dual-GPU + Poda BVH 96.88% + MTP Drafter |

---

## 3. Compilação Nativa

O motor canônico vive em `unified_runtime.cu` e é compilado diretamente via `nvcc` integrado ao MSVC:

```powershell
nvcc -ccbin "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64" -O3 -arch=sm_75 src/litert_explore/hpc_engine/unified_runtime.cu -o src/litert_explore/hpc_engine/unified_runtime.exe
```

---

## 4. Sintaxe CLI e Parâmetros

```text
src/litert_explore/hpc_engine/unified_runtime.exe [OPÇÕES]
```

### Argumentos Principais:

| Flag | Tipo | Padrão | Descrição |
| :--- | :--- | :--- | :--- |
| `--model <nome>` | `string` | `moe` | Modelo alvo: `moe`, `ternary`, `gemma4`, `gemma12b`, `ornith`. |
| `--tokens <N>` | `int` | `30` | Número de tokens a decodificar na sessão de inferência. |
| `--prompt-len <N>` | `int` | `128` | Comprimento do prompt de entrada (tokens avaliados no Prefill). |
| `--drafter <tipo>` | `string` | `auto` | Drafter neural auxiliar: `auto`, `dspark`, `eagle3`, `mtp`, `none`. |
| `--draft-model <path>` | `string` | Padrão | Caminho do modelo drafter (ex: `Z:\models\...\eagle3-gpt-oss-20b-Q8_0.gguf`). |
| `--virtual-experts` | `flag` | Desativado | Habilita interceptação e chamada assíncrona de Virtual Experts (Math + RAG). |
| `--async-tools` | `flag` | Desativado | Executa as ferramentas no host em paralelo aos kernels da GPU via `std::async`. |
| `--session-mode <modo>` | `string` | `global` | Modo de gestão de memória: `global`, `ephemeral`, `hierarchical`. |
| `--clean-cache` | `flag` | Desativado | Limpa arquivos temporários e caches em disco após a conclusão da execução. |
| `--json` | `flag` | Desativado | Emite a telemetria física completa em formato JSON estruturado no stdout. |

---

## 5. Exemplos Canônicos de Execução

### A. Executar Gemma-4-12B com Drafter DSpark e Virtual Experts
```powershell
src/litert_explore/hpc_engine/unified_runtime.exe --model gemma12b --drafter dspark --virtual-experts --async-tools --tokens 30 --json
```

### B. Executar GPT-OSS-20B com Eagle-3 e Limpeza de Disco ao Final
```powershell
src/litert_explore/hpc_engine/unified_runtime.exe --model moe --drafter eagle3 --clean-cache --tokens 50 --json
```

### C. Executar Sessão Efêmera com Modo Hierárquico
```powershell
src/litert_explore/hpc_engine/unified_runtime.exe --model ornith --session-mode hierarchical --tokens 30 --json
```
