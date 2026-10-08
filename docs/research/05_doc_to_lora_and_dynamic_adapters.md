---
title: "Doc-to-LoRA (Sakana AI), Context Language Models (Meta CLM) e Adaptadores Dinâmicos On-Demand"
status: "VALIDATED_AND_INTEGRATED"
date: "2026-10-08"
authors: ["Alefita", "Antigravity Systems Research"]
target_hardware: ["NVIDIA RTX 2060 (Turing SM 7.5)", "NVIDIA GTX 1050 Ti (Pascal SM 6.1)", "DirectStorage 1.2"]
related_components: [
  "src/litert_explore/live/session.py",
  "src/litert_explore/live/code_synthesizer.py",
  "src/litert_explore/live/chat_template.py",
  "benchmarks/plugins/contextbench_eval.py"
]
tags: [
  "doc-to-lora",
  "sakana-ai",
  "hypernetworks",
  "context-language-models",
  "meta-clm",
  "suffix-cache-reuse",
  "dynamic-adapters",
  "cordis-plugins",
  "mini-agi",
  "virtual-experts"
]
---

# Pesquisa Avançada: Doc-to-LoRA, Meta Context Language Models e Adaptadores Dinâmicos em Silício

## 1. Doc-to-LoRA (Sakana AI): Internalização Instantânea de Contexto via Hiper-redes

A inspeção do repositório canônico `SakanaAI/doc-to-lora` (`ctx_to_lora`) revela um avanço conceitual de primeira grandeza na separação entre pesos congelados e condicionamento de contexto de longo prazo:

### Mecanismo Matemático e Arquitetura de Hiper-rede
Em vez de preencher o KV-Cache com milhões de tokens de contexto (o que impõe complexidade quadrática de atenção e esgota rapidamente os 6 GB de VRAM da RTX 2060), o **Doc-to-LoRA** sintetiza uma perturbação de baixo posto (low-rank adapter) diretamente do documento de entrada:

$$\Delta W = B \cdot A, \quad A \in \mathbb{R}^{r \times d_{in}}, \quad B \in \mathbb{R}^{d_{out} \times r}, \quad r \in \{8, 16\}$$

1. **Codificação do Documento**: O texto ou base de conhecimento $D$ é projetado através de um encoder de embeddings (`emb_model`) e agrupado (`pooling_fn`).
2. **Hiper-rede Geradora**: O `task_encoder` e `get_delta_weights` da classe `TextToLoRA` transformam o vetor latente do documento nos tensores $A$ e $B$ para cada camada alvo (`q_proj`, `v_proj`, `gate_up_proj`).
3. **Injeção Dinâmica em Forward**: As matrizes $\Delta W$ são aplicadas via `lora_forward` com escalonamento $\alpha / r$:
   $$y = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$
4. **Custo Zero de Reversão**: Ao finalizar a sessão ou mudar de tarefa, a chamada `model.reset()` descarrega os tensores $A$ e $B$, restaurando o backbone base imaculado sem jamais ter executado retropropagação (backpropagation) no modelo fundacional.

---

## 2. Context Language Models (Meta CLM, arXiv:2609.37725v1): O Contexto como Estado Editável

O paper da Meta introduz a formalização epistemológica dos **Context Language Models (CLMs)**, superando a visão ingênua de que a janela de contexto de um LLM deve ser uma fita append-only imutável.

### A Transformação Epistemológica do Contexto
Em LLMs autoregressivos padrão, o contexto cresce estritamente de forma monotônica:
$$c_{t+1} = c_t \circ y_t$$

No paradigma Meta CLM, o contexto é um estado dinâmico de computação manipulado por operadores de reflexão, poda, substituição e consolidação:
$$c_{t+1} = f_\theta^{\mathrm{CLM}}(c_t)$$

### Suffix Cache Reuse e Recomputação Residual
Quando um agente realiza chamadas de ferramentas (*tool calling*), busca na web ou sintetiza rascunhos em cadeia de pensamento (CoT), o histórico bruto acumula ruído descartável. O Meta CLM demonstra que substituir o payload prolixo de uma tool pelo resultado destilado, ou podar passos falhos de raciocínio, gera economia massiva se o runtime suportar **Suffix Cache Reuse**:
- **Prefix Reuse**: Tokens anteriores à mutação mantêm seus blocos de KV intactos no buffer circular.
- **Suffix Invalidation & Recomputation**: Apenas o delta de tokens alterado invalida blocos posteriores, que são recomputados em streaming aproveitando os estados de fronteira $h_{\mathrm{boundary}}$ da GPU 1 (GTX 1050 Ti) sem stalls de barramento PCIe.

---

## 3. Mini-AGI e Roteamento de Especialistas em Três Camadas

A análise do código-fonte do `volotat/mini-AGI` (`minagi/paged.py` e `minagi/store.py`) corrobora nossa arquitetura de tiers de memória:

> *"A pool larger than the card it runs on. The claim this exists to make true: disk holds every expert, RAM caches the ones recently wanted, and VRAM holds only the ones being worked with now. Three tiers, and only the last is scarce."*

### Delimitação de Escopo e Decisão Estratégica
- **Treinamento de Novo Modelo Deferido**: Conforme alinhado com a liderança técnica, **não** treinamos um novo modelo Mini-AGI no presente ciclo. O modelo base e os pesos existentes permanecem intocados.
- **Substrato de Especialistas Compartilhados / Especializados**: A técnica de paginação do Mini-AGI (armazenar especialistas MoE ou módulos de reflexão em disco e paginar para VRAM em slots fixos) é adotada como substrato agnóstico de expansão modular. O modelo pode solicitar e instanciar *virtual experts* em disco para matemática simbólica, parsing de código ou validação de esquemas de forma transparente.

---

## 4. Integração CORDIS: Plugins de Adaptadores Dinâmicos e Busca Web GPT-OSS

### A. Anatomia da Busca Web no GPT-OSS (`openai/gpt-oss`)
A inspeção do repositório oficial da OpenAI comprova empiricamente que o GPT-OSS **não** possui uma ferramenta genérica isolada de `web_search`.
- A capacidade de pesquisa está encapsulada dentro da ferramenta `SimpleBrowserTool` (`browser` namespace).
- O método `browser.search(query=...)` despacha a consulta para backends configurados como **Exa** (`https://api.exa.ai`) ou **You.com** (`https://api.ydc-index.io`).
- O resultado é devolvido em formato de página de navegação sintética (`PageContents`) com âncoras de citação estruturadas `【{idx}†{title}】` e identificadores de linha `L{idx}`, sobre as quais o modelo opera com comandos `browser.open` e `browser.find`.

### B. Módulos CORDIS no Unified CED Runtime
O sistema de extensões CORDIS unifica estas capacidades em um plano de execução único:
1. **Tool Streaming Interceptor**: Substitui chamadas brutas de ferramentas pelo resultado in-place no streaming, executando rollback e branch de sessões sem penalidade de latência.
2. **Dynamic LoRA Hot-Swapping**: Utiliza a abordagem Doc-to-LoRA para materializar pesos $A/B$ temporários em VRAM diretamente para tarefas especializadas (ex: síntese rigorosa de AST Python para HumanEval/MBPP).
3. **Persistência de Memória Hierárquica**: Mantém engrams em disco NVMe rápido (`Z:\models` e `Z:\workspaces`), permitindo que múltiplos modelos (ex: Gemma 4 12B denso + GPT-OSS 20B MoE) compartilhem o mesmo anel de DMA e barramento gráfico sem colisões.
