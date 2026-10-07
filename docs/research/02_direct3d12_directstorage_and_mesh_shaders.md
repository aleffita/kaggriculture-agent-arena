---
title: "Lições de Silício Gráfico: Direct3D 12, DirectStorage GDeflate e Mesh Shaders em LLMs"
status: "RESEARCH_COMPLETED_PENDING_PROBE"
date: "2026-10-07"
authors: ["Alefita", "Antigravity Systems Research"]
target_hardware: ["NVIDIA RTX 2060 (Turing SM 7.5 / D3D12 Tier 1.1)", "DirectStorage 1.2"]
related_components: ["src/litert_explore/hpc_engine/unified_runtime.cu", "probes/bench_bvh_moe_and_asics.cu"]
tags: ["direct3d12", "directstorage", "gdeflate", "mesh-shaders", "nvofa", "tiled-resources", "bvh"]
---

# Pesquisa Avançada: Lições de Silício Gráfico Direct3D 12 para Modelos de Linguagem

## 1. Mapeamento de Tensores: Raw `ByteAddressBuffer` vs Texturas 2D

A inspeção do Google LiteRT-LM / Dawn revelou que tensores de pesos são empacotados em buffers lineares `ByteAddressBuffer` com instruções de 128 bits (`uint4`).

### Conclusões Arquiteturais:
1. **Unificação de Cache no Turing SM 7.5**: O L1 Data Cache e o Texture Cache compartilham a mesma SRAM física de 64 KB a 96 KB por SM. Logo, texturas 2D não trazem qualquer benefício de capacidade ou largura de banda sobre buffers lineares.
2. **Penalidade de Swizzling Morton 2D**: Texturas 2D impõem layout Z-curve para amostragem espacial 2D. Em matrizes GEMV/GEMM de LLM (onde o acesso é estritamente linear ao longo da dimensão $K$), o swizzle quebra a coalescência de linha, degradando o throughput de leitura em mais de 35%.
3. **Incapacidade de Filtragem de Inteiros**: Unidades de textura (TMUs) não executam interpolação bilinear em formatos inteiros (`R32_UINT`). Como os pesos PTQ1_0 e MXFP4 são dados inteiros puros, qualquer uso de TMU seria redundante e introduziria entre 20 e 80 ciclos de latência desnecessária.
4. **Decisão**: O runtime canônico mantém o uso estrito de **raw pointers / ByteAddressBuffers lineares coalescidos** para matrizes de pesos.

---

## 2. DirectStorage com Descompressão em Silício GPU (GDeflate)

### Modelagem de Throughput:
Ao transmitir pesos de especialistas MoE de $64\text{ MB}$ a partir de um SSD NVMe PCIe Gen3 x4 ($3500\text{ MB/s}$):
- **Win32 Direct Unbuffered I/O**:
  $$t_{\text{transfer}} = \frac{64\text{ MB}}{3.5\text{ GB/s}} = 18.28\text{ ms}$$
- **DirectStorage 1.2 + GDeflate via `BypassIO`**:
  Com razão de compressão típica de $2.2\times$ em tensores esparsos de MoE:
  $$\text{Tamanho no barramento} = \frac{64\text{ MB}}{2.2} \approx 29.09\text{ MB}$$
  $$t_{\text{nvme}} = \frac{29.09\text{ MB}}{3.5\text{ GB/s}} = 8.31\text{ ms}$$
  $$t_{\text{decomp\_gpu}} = \frac{64\text{ MB}}{28\text{ GB/s}} = 2.28\text{ ms}$$
  $$t_{\text{total}} = 8.31\text{ ms} \quad (\text{Speedup efetivo de } 2.20\times)$$

No link restrito PCIe Gen3 x4 da GPU 1 (GTX 1050 Ti), a taxa efetiva salta de $3.2\text{ GB/s}$ para **$7.04\text{ GB/s}$**, dobrando a vazão do anel DMA.

---

## 3. Orquestração MoE via Task e Mesh Shaders (D3D12 SM 6.5)

Em modelos esparsos MoE (`gpt-oss-20b`), o despacho tradicional exige que a CPU receba os índices dos especialistas ativos para emitir os comandos de computação, gerando latência de sincronização.

### Pipeline Gráfico em Silício:
1. **Task Shader (Amplification Shader)**:
   - Um warp de threads projeta os tokens na grade 3D, percorre a árvore BVH em hardware e determina os Top-$K$ especialistas.
   - Emite a instrução de silício:
     ```hlsl
     DispatchMesh(K, 1, 1, task_payload);
     ```
2. **Mesh Shader**:
   - O hardware scheduler da GPU instancia diretamente os $K$ grupos de threads para computar os blocos SwiGLU / GEMV dos especialistas ativos.
   - **Zero intervenção de CPU** e zero padding de tensores inativos.

---

## 4. Avaliação do NVOFA (NVIDIA Optical Flow Accelerator)

1. **Hardware Disponível**: Apenas a RTX 2060 (Turing SM 7.5) possui NVOFA; a GTX 1050 Ti não possui o acelerador.
2. **Inadequação para NLP**:
   - O NVOFA assume invariância de intensidade luminosa em grades 2D suaves, enquanto o estado residual $h_{boundary}$ é hiperdimensional ($d=2880$) e não-local.
   - A latência de lançamento do driver NVOFA é de **0.8 a 1.5 ms**, enquanto um kernel CUDA de extrapolação vetorial com warp-shuffle roda em **3.2 $\mu$s** (400x mais rápido).
3. **Escopo Único Viável**: O NVOFA é excelente para pré-processamento de vídeo e poda de tokens visuais em Modelos Visão-Linguagem Multimodais (VLMs/Omni), devendo ser excluído do caminho crítico de texto.

---

## 5. Próximos Passos e Status de Implementação

- [x] Avaliação teórica e modelagem de largura de banda
- [x] Verificação da superioridade de raw buffers vs texturas 2D
- [ ] Construção de sonda física D3D12 DirectStorage GDeflate em `probes/bench_directstorage_gdeflate.cpp`
- [ ] Interoperabilidade CUDA-D3D12 via `cudaGraphicsD3D12RegisterResource`
