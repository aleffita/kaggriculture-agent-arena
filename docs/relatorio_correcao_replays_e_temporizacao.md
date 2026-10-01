# Relatório Técnico: Regeneração de Replays Integrais e Correção da Temporização (24 Turnos/Dia)

**Data**: 30 de Setembro de 2026  
**Ambiente**: Arena Kaggriculture + DuckDB (`kaggriculture/data/arena.duckdb`)  
**Servidor Ativo**: [http://127.0.0.1:8080/](http://127.0.0.1:8080/)

---

## 1. Resumo Executivo (BLUF)

Todos os **120 replays da arena** foram regenerados com seus horizontes completos em disco (`kaggriculture/data/replays/`) e devidamente sincronizados com o DuckDB:
- **FullSeason (30 partidas)**: **720 passos** completos (Dia 1 a 30, Turnos 1 a 24).
- **Scaling (30 partidas)**: **240 passos** completos (Dia 1 a 10, Turnos 1 a 24).
- **Expansion (30 partidas)**: **144 passos** completos (Dia 1 a 6, Turnos 1 a 24).
- **Sprint (30 partidas)**: **72 passos** completos (Dia 1 a 3, Turnos 1 a 24).

Zero truncamentos remanescentes (`0 de 120`). Todo o corpus de dados está íntegro para extração de trajetórias nos pipelines de RL e GRPO.

---

## 2. Diagnóstico das Causas Raízes

### A. Truncamento dos Replays em 72 Passos
- **Causa**: Um exportador anterior invocava o ambiente limitando o horizonte com `min(steps, 72)`. Como resultado, todas as 90 partidas das etapas de Expansion, Scaling e FullSeason continham apenas 72 passos nos arquivos JSON em disco, impedindo o aprendizado de dinâmicas avançadas (melancias, morangos, animais e expansão territorial).
- **Correção**: Implementado e executado o gerador paralelo [`regenerate_all_arena_replays.py`](../scripts/regenerate_all_arena_replays.py). O script utilizou as 6 políticas macro-estratégicas nativas com `actTimeout: 999999` e completou a simulação determinística dos 89 replays pendentes em apenas 64 segundos.

### B. Incongruência do Relógio (24 Turnos/Dia)
- **Causa**: No Kaggriculture, cada dia possui estritamente **24 turnos**. O visualizador oficial e o HUD anterior usavam convenções mistas de passo 0-indexado e 1-indexado sem explicitar o total de dias nem os 24 turnos por ciclo, além de sofrer conflito de `postMessage` com o listener interno do Vite player em nova janela.
- **Correção**:
  1. A fórmula do relógio foi unificada matematicamente:
     $$\text{Dia} = \lfloor \text{Passo} / 24 \rfloor + 1 \quad / \quad \lceil \text{TotalPassos} / 24 \rceil$$
     $$\text{Turno} = (\text{Passo} \pmod{24}) + 1 \quad / \quad 24$$
  2. As pílulas de status no HUD standalone e modal agora exibem:
     `📅 Dia 1 / 30 • ⏱️ Turno 1 / 24 • Passo 0 / 720`
  3. O despacho de telemetria isola eventos externos, impedindo que o visualizador resete seu frame para 0.

### C. Inventário e Dinheiro em Tempo Real
- **Causa**: O leitor de observações inspecionava apenas `obs.private.seeds` e `obs.private.shed`, ignorando a lista `obs.private.inventories` (itens transportados no bolso do fazendeiro e ajudantes).
- **Correção**: Implementada agregação completa de inventário carregado (`🎒`), sementes no barracão (`🌱`) e colheitas no armazém (`📦`), refletindo imediatamente compras e colheitas sem falso-positivo de "Bolsa vazia".

---

## 3. Matriz de Auditoria dos Replays

| Estágio | Partidas | Passos Alvo | Passos Reais em Disco | Status de Auditoria |
| :--- | :---: | :---: | :---: | :---: |
| **Sprint** | 30 | 72 | 72 | ✅ 100% Íntegro |
| **Expansion** | 30 | 144 | 144 | ✅ 100% Íntegro |
| **Scaling** | 30 | 240 | 240 | ✅ 100% Íntegro |
| **FullSeason** | 30 | 720 | 720 | ✅ 100% Íntegro |
| **Total Corpus** | **120** | **35.280** | **35.280** | ✅ **Pronto para Treino** |

---

## 4. Arquivos Modificados e Criados

- [`scripts/regenerate_all_arena_replays.py`](../scripts/regenerate_all_arena_replays.py): Gerador paralelo multiprocesso para regeneração e sincronização com DuckDB.
- [`kaggriculture/dashboard/app.py`](../kaggriculture/dashboard/app.py): Ajuste do HUD standalone, fórmulas do relógio (24 turnos/dia), suporte a inventários carregados e proteção contra loops no Vite player.
- [`kaggriculture/dashboard/static/app.js`](../kaggriculture/dashboard/static/app.js): Atualização do listener modal e renderização granular de inventários.
- [`kaggriculture/dashboard/templates/index.html`](../kaggriculture/dashboard/templates/index.html): Padrão visual do relógio `1 / 24` turnos.
