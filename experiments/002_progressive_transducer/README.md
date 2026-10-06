# Experimento 002: Transdutor Progressivo & Probe P1 (Análise do Grafo e Topologia Gemma 4 E2B)

> Data: 2026-10-06  
> Modelo: `gemma-4-E2B-it.litertlm` (2,47 GB, LiteRT-LM 1.5.0)  
> Hardware: NVIDIA GTX 1050 Ti (Pascal SM 6.1) + AMD Ryzen 5  

---

## 1. Hipótese de Investigação (Probe P1)

Verificar se o modelo local `gemma-4-E2B-it.litertlm`:
1. Possui arquitetura de KV-sharing (quase-CED) com 20 camadas compartilhadas e 15 camadas de base.
2. Contém buffers de KV cache apenas para as 15 primeiras camadas.
3. Se o compilador e grafo do LiteRT-LM (`prefill_1024` vs `decode`) já pulam ou podam o cálculo das camadas 15–34 durante o prefill.
4. Se já possui suporte embutido a rascunho especulativo (MTP drafter e assinatura `verify`).

---

## 2. Metodologia e Execução

Executamos inspeção estática no flatbuffer `.litertlm` e inicialização direta da engine em C++ via bindings Python do `litert_lm` / `litert_lm_builder`.

### A. Anatomia das Seções do Arquivo `.litertlm`
O arquivo é um container flatbuffer com 12 seções:
* **Seção 0**: `LlmMetadataProto` (BOS=2, EOS=[1, 50, 106], gemma4 multimodal affixes).
* **Seção 1**: `SP_Tokenizer` (4,47 MB, SentencePiece).
* **Seção 2**: `tf_lite_embedder` (103 MB, 8 ops).
* **Seção 3**: `tf_lite_per_layer_embedder` (**1,28 GB**, 42 ops, assinatura `per_layer_embedder`).
* **Seções 4-6**: Encoders de áudio (94 MB + 9,4 MB + 6,7 KB).
* **Seções 7-9**: Encoders de visão SigLIP/ViT (224 MB + 4,7 MB + 6,7 KB).
* **Seção 10**: `tf_lite_prefill_decode` (**780,37 MB**, o modelo base de linguagem).
* **Seção 11**: `tf_lite_mtp_drafter` (**42,27 MB**, modelo drafter de 4 camadas para Multi-Token Prediction).

### B. Mapeamento de Tensores de KV Cache
Varredura direta dos tensores de estado na Seção 10:
* **Tensores K**: `kv_cache_k_0` até `kv_cache_k_14` (Exatamente 15 buffers).
* **Tensores V**: `kv_cache_v_0` até `kv_cache_v_14` (Exatamente 15 buffers).
* **Camadas 15 a 34**: **Nenhum tensor de KV cache existe no modelo compilado**.
* **Total de Camadas Detectadas no Grafo**: 35 camadas (0 a 34).

### C. Assinaturas e Contagem de Operadores no Grafo TFLite
A engine LiteRT compilou 4 subgrafos principais no modelo:

| Assinatura | Subgrafo | Nº de Inputs | Nº de Outputs | Nº de Operadores | Descrição |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `prefill_1024` | 1 | 35 | **30** | **1.107** | Prefill para sequências até 1024 tokens |
| `prefill_128` | 2 | 35 | **30** | **1.107** | Prefill para sequências curtas |
| `decode` | 0 | 35 | 32 | **2.068** | Decode autoregressivo (token único) |
| `verify` | 3 | 35 | 32 | **2.243** | **Verificação especulativa multi-token em lote** |

---

## 3. Conclusões Fáticas

1. **Pruning do Prefill é Real e Nativo**:
   * O grafo de `prefill_1024` tem apenas **30 saídas** — correspondendo exatamente aos 15 $K$ e 15 $V$ das primeiras 15 camadas.
   * `prefill_1024` executa apenas **1.107 operadores**, contra **2.068 operadores** no `decode`.
   * Isso prova que **o LiteRT-LM já descarta a computação das 20 camadas superiores (15–34) durante o prefill em lote**, calculando estritamente as camadas necessárias para popular o KV cache de base!
2. **Quase-CED Confirmado no Silício**:
   * O Gemma 4 E2B opera exatamente como um Causal Encoder-Decoder: as primeiras 15 camadas funcionam como o *Causal Encoder* (produzindo o KV cache global), enquanto as 20 camadas superiores operam sobre a projeção desse KV.
3. **Speculative Decoding / Transdutor Nativo**:
   * O modelo já vem empacotado de fábrica com a Seção 11 (`tf_lite_mtp_drafter`, 42 MB) e com o subgrafo compilado `verify` (2.243 ops), viabilizando verificação de árvore especulativa sem necessidade de adaptação manual de pesos.
