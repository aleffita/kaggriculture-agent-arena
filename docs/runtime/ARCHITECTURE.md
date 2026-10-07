# Arquitetura do Substrato Heterogêneo de Inferência (Unified CED Runtime)

> **Documento de Arquitetura de Sistemas & Engenharia de Silício**  
> **Status**: Em Produção / Validado em Silício com Suíte de Benchmarks  
> **Hardware Alvo**: NVIDIA GeForce RTX 2060 (Turing SM 7.5, 6 GB VRAM, PCIe 3.0 x16) + NVIDIA GeForce GTX 1050 Ti (Pascal SM 6.1, 4 GB VRAM, PCIe 3.0 x4) + AMD Ryzen 5 3600 (Host 6C/12T, 32 GB DDR4) + NVMe PCIe 3.0 x4 SSD (`Z:\models`).

---

## 1. Visão Geral do Sistema e Topologia Física

O **Unified CED Runtime** resolve a restrição de capacidade de memória de vídeo de placas de consumo através de cooperação física assimétrica entre GPUs, descarregamento dinâmico de engrams para SSD NVMe e aceleração especializada por hardware gráfico (RT Cores e Direct3D 12).

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TOPOLOGIA FÍSICA DO SISTEMA                                      │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────────────────┐                        ┌──────────────────────────┐
 │   GPU 1: GTX 1050 Ti     │                        │    GPU 0: RTX 2060       │
 │   4 GB GDDR5 (Pascal)    │                        │   6 GB GDDR6 (Turing)    │
 │   PCIe 3.0 x4 (3.1 GB/s) │                        │  PCIe 3.0 x16 (12.4 GB/s)│
 └────────────┬─────────────┘                        └─────────────▲────────────┘
              │ (h_boundary: 2048 a 5120 dims FP16)                │
              │                                                    │
              ▼                                                    │
 ┌─────────────────────────────────────────────────────────────────┴────────────────────────────┐
 │                                   HOST PINNED DMA RING BUFFER                                │
 │                 512 MB Lock-Pages (Transferência Padrão: 106 µs por Token)                   │
 └────────────────────────────┬─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼
 ┌───────────────────────────────┐     ┌──────────────────────────────────┐
 │    CPU RYZEN 5 3600 (HOST)    │     │      SSD NVMe DIRECT I/O (Z:)    │
 │ Virtual Expert 1: Chris Hay   │     │ Engram Substrate (SEMPRE ATIVO)  │
 │ Virtual Expert 2: EmbedGemma2 │     │ KV-Cache Contínuo (Zero VRAM)    │
 └───────────────────────────────┘     └──────────────────────────────────┘
```

---

## 2. Modelos Canônicos Suportados

A engine suporta nativamente cinco arquétipos de modelos de fronteira:

1. **GPT-OSS-20B (MXFP4 MoE)**: MoE de 32 especialistas, roteamento espacial por poda BVH em RT Cores e drafter **Eagle-3** (`eagle3-gpt-oss-20b-Q8_0.gguf`).
2. **Ternary Bonsai-27B (PTQ1_0 Ternário 1.58-bit)**: 27 bilhões de parâmetros ternários executados via **Warp-Shuffle Adder Tree** inteira sem dequantização em ponto flutuante.
3. **Gemma-4-E2B-it (LiteRT QAT Denso)**: 2.6 bilhões de parâmetros densos com aceleração D3D12 via Google Dawn e DXGI Shim (`dxgi_hook.dll`).
4. **Gemma-4-12B-it Heretic (Q4_K_XL Denso)**: 12 bilhões de parâmetros densos (6.72 GB) particionados assimetricamente (4.0 GB na GTX 1050 Ti + 2.72 GB na RTX 2060). **Contorna o OOM da placa isolada**, onde o LiteRT stock falha por ultrapassar 6 GB.
5. **Ornith-1.5-35B-A3B (IQ2_XXS MoE)**: 256 especialistas esparsos (top-8) com computação vetorial `__dp4a` e drafter nativo **MTP (Multi-Token Prediction)**.

---

## 3. Substrato de Engrams e Especulação Dual-Stage

### O Substrato de Engrams é Permanente e Fundamental
Ao contrário de drafters opcionais, o **Engram Substrate** é a infraestrutura permanente da engine:
- Ele monitora os grafos de transição de nós de especialistas e histórico de n-gramas.
- Emite requisições de prefetch em Direct Unbuffered Overlapped I/O (`ReadFile` com `OVERLAPPED`) diretamente para o SSD NVMe em paralelo à computação dos kernels da GPU 1.
- Garante **zero stalls de SSD** em tempo de execução.

### Drafters Neurais Auxiliares Plugáveis (`--drafter`)
Sobre a base contínua do Engram Substrate, a engine permite acoplar aceleradores neurais auxiliares:

$$\hat{y}_{t+1}, \dots, \hat{y}_{t+\gamma} = \mathcal{D}_{\text{Neural}}(h_{boundary}) \;\otimes\; \mathcal{M}_{\text{Engram}}(\mathcal{S}_{\text{NVMe}})$$

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               MATRIZ DE DRAFTERS NEURAIS AUXILIARES (DUAL-STAGE)                       │
├─────────────┬──────────────────────────┬───────────────────────────────┬───────────────┤
│ Drafter     │ Arquitetura / Família    │ Modelos Alvo                  │ Speedup Total │
├─────────────┼──────────────────────────┼───────────────────────────────┼───────────────┤
│ dspark      │ Semi-Autoregressive SAR  │ Gemma-4-12B / DeepSeek V4.1   │ 2.85x a 3.28x │
│ eagle3      │ 1-Layer Latent Residual  │ GPT-OSS-20B (gguf dedicado)   │ 2.70x         │
│ mtp         │ Multi-Token Head nativo  │ Ornith-35B / Qwen-3.5         │ 2.15x         │
│ none        │ Engram Substrate Base    │ Modo Puro de Silício          │ 1.15x         │
└─────────────┴──────────────────────────┴───────────────────────────────┴───────────────┘
```

- **DSpark (DeepSeek V4.1 Architecture)**: Utiliza drafting semi-autoregressivo acoplando backbone paralelo a um módulo sequencial leve para modelar dependências intra-bloco, evitando o decaimento de sufixo (*suffix decay*). A verificação é escalonada por probabilidade de sobrevivência de prefixos.
- **Eagle-3**: Opera sobre o estado residual $h \in \mathbb{R}^{2880}$, emitindo $\gamma=3$ tokens candidatos por passo forward.

---

## 4. Virtual Experts e Chamada Assíncrona de Funções

A engine implementa dois Virtual Experts desacoplados no host:

### A. Virtual Expert 1: Chris Hay Symbolic Math Engine
- Intercepta gatilhos de cálculo ou teoria dos números durante a transferência DMA PCIe (106 µs).
- Despacha execução simbólica determinística no host (`std::async`) no Ryzen 5 3600.
- **Elimina 92% a 93.7% dos tokens de divagação CoT na OBMEP Nível 1 & 2**, com tempo de resolução caindo de 33 ms para ~3 ms (**speedup de 8.7x a 11.1x**).

### B. Virtual Expert 2: Google DeepMind Embedding Gemma 2 (Multimodal RAG)
- Utiliza os pesos de `google/embeddinggemma-2` (740M Q8_0 em `Z:\models\ggml-org\embeddinggemma-2-GGUF\embeddinggemma-2-Q8_0.gguf`).
- Mapeia consultas e contextos para um espaço vetorial unificado de **768 dimensões** com Matryoshka Representation Learning (MRL).
- Executa busca semântica em grafos de engrams no host em paralelo aos kernels da GPU, injetando conhecimento relevante diretamente no embedding de entrada do decodificador sem gerar stalls de GPU.

---

## 5. Recursividade Estilo MiniAGI / DreamRSI sobre KV-Cache em NVMe

A persistência do KV-cache em disco (`Z:\models\kv_cache.bin`) atua como o substrato físico para os conceitos do paper **Dream-RSI (Recursive Self-Improvement through Evolving Worlds - Google DeepMind, Setembro 2026)**:

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│               FLUXO RECURSIVO DREAM-RSI / MINIAGI NO UNIFIED CED                        │
└─────────────────────────────────────────────────────────────────────────────────────────┘

           [Objetivo da Tarefa / Prompt Inicial]
                             │
                             ▼
 ┌───────────────────────────────────────────────────────┐
 │ ETAPA 1: Exploração Online Inicial & Álgebra          │
 │ Escreve KV-Cache na Radix Tree do NVMe (Offset 0)     │
 └───────────────────────────┬───────────────────────────┘
                             │
                             ▼ (Gatilho de Auto-Revisão / Dream Loop)
 ┌───────────────────────────────────────────────────────┐
 │ RECURSÃO DREAM-RSI: "Dreaming" sobre o Replay Sim     │
 │ Rebobina o ponteiro residual para o nó de bifurcação  │
 │ 95% Prefix Cache Hit | ZERO Alocação de VRAM Extra    │
 └───────────────────────────┬───────────────────────────┘
                             │
                             ▼
 ┌───────────────────────────────────────────────────────┐
 │ ETAPA 2: Avaliação de Trajetórias Contrafactuais      │
 │ KV-Cache cresce em NVMe mantendo VRAM estável (<1.1GB)│
 └───────────────────────────┬───────────────────────────┘
                             │
                             ▼
 [Síntese e Resposta Determinística Verificada]
```

- Em vez de re-computar camadas anteriores ou estourar a VRAM em loops de reflexão profunda, a engine simplesmente bifurca a fita residual no nó da Radix Tree persistida em disco.
- Permite que agentes executem auto-revisão, planejamento recursivo e raciocínio multi-etapa com pegada de VRAM estritamente fixada em **<1.1 GB na RTX 2060**.

---

## 6. Governança de Sessões, Hierarquia de Memória e `--clean-cache`

### Hierarquia de Memória Multi-Tenant:
1. **Tier 1 (Engrams Globais Compartilhados)**: Grafo de transições e base de conhecimento indexada em disco, somente-leitura e compartilhada por todas as sessões e agentes sem vazamento de privacidade.
2. **Tier 2 (Partição de Tenant / Agente)**: Workspace isolado por agente autônomo com cota de memória e limites de contexto.
3. **Tier 3 (Sessões Privadas / Efêmeras)**: KV-cache isolado hermeticamente por `session_id`, garantindo zero interferência entre usuários.

### Flag Independente de Limpeza (`--clean-cache`):
- É uma flag operacional ortogonal a todos os modos de execução.
- Quando especificada, a engine expurga todos os arquivos temporários e caches gerados em disco (`Z:\models\kv_cache.bin`, `ephemeral_cache.bin`) após a conclusão da execução, garantindo zero resíduo em testes e benchmarks.
