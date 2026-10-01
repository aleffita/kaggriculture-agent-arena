# Experiment 009: Macro-Planning Horizon Ablations & Concurrency

> - **Season 1 Report**: Documented below (Sprint Stage, 72 steps, H24/H12/H6/Hybrid).
> - **Season 2 Full Tournament Report**: See [`SEASON2.md`](file:///d:/workdir/litertlm-exploration/experiments/009_isolated_concurrency_and_macro_planning/SEASON2.md) (64 matches, 6 competitors, 144 steps Round-Robin + 720 steps Playoffs).
> - **Season 3 Scaling Curves & DeepSeek HCA-CSA**: See [`SEASON3.md`](file:///d:/workdir/litertlm-exploration/experiments/009_isolated_concurrency_and_macro_planning/SEASON3.md) (12 full-season matches at 720 steps, dual-engine VRAM, H48 wheat resonance champion, HCA-CSA).

## Executive Summary (Season 1)

Season 1 evaluated the fundamental runtime mechanics of Google LiteRT-LM (Gemma 4 E2B) on an **NVIDIA GeForce GTX 1050 Ti (4 GB)** via Direct3D 12 (Google Dawn + DXGI interceptor).

It proved that the historical bottleneck in Kaggriculture was not memory capacity (120 active sessions require only 2,084 MB / 4,096 MB VRAM), but **fine-grained per-turn invocation ping-pong (144 sequential round-trips per 72-step match)**, which caused single matches to take ~12.6 minutes.

By refactoring agents to use **Macro-Action Trajectory Planning** with native structured decoding (`LL_GUIDANCE`), matches completed in **40.37 seconds (18.8x speedup)** while producing functional farm economies ($2,880 vs $2,640).

---

## 1. Season 1 Mini-Tournament (Round-Robin with Seed 42)

A round-robin tournament was executed across 4 decision horizon policies under a controlled deterministic seed (`seed=42`) for a 72-step Sprint match:

### Final Leaderboard

| Rank | Variant | Description | Final Elo | W-D-L | Total Coins | Inferences | Efficiency ($/inf) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1st** | **H24** | Fixed Daily Horizon (1 call/day) | **645.9** | **3-0-0** | **$8,460** | **9** | **$940.0** |
| 🥈 **2nd** | **Hybrid** | H24 Base + Event-Driven Re-planning | **616.6** | **2-0-1** | **$8,400** | **10** | **$840.0** |
| 🥉 **3rd** | **H12** | Half-Day Horizon (2 calls/day) | **583.3** | **1-0-2** | **$7,920** | **18** | **$440.0** |
| 4th | **H6** | Quarter-Day Horizon (4 calls/day) | **554.2** | **0-0-3** | **$6,840** | **36** | **$190.0** |

---

## 2. Match-by-Match Outlines

All matches executed on the GTX 1050 Ti with `seed=42`:

| Match | P0 | P1 | Winner | Final Banks | Match Duration | Inferences (P0 vs P1) | Re-plan Events |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | H24 | H12 | **H24** | **$2,820** vs $2,640 | 62.0s | 3 vs 6 | 0 vs 0 |
| **2** | H24 | H6 | **H24** | **$2,820** vs $2,280 | 83.1s | 3 vs 12 | 0 vs 0 |
| **3** | H24 | Hybrid | **H24** | **$2,820** vs $2,760 | 53.6s | 3 vs 4 | 0 vs 1 |
| **4** | H12 | H6 | **H12** | **$2,640** vs $2,280 | 109.2s | 6 vs 12 | 0 vs 0 |
| **5** | H12 | Hybrid | **Hybrid** | $2,640 vs **$2,820** | 62.5s | 6 vs 3 | 0 vs 0 |
| **6** | H6 | Hybrid | **Hybrid** | $2,280 vs **$2,820** | 86.4s | 12 vs 3 | 0 vs 0 |

---

## 3. Key Scientific Insights

1. **Short-Horizon Tactical Myopia**:
   - In Kaggriculture, wheat requires 48 hours to mature.
   - Slicing planning into 6-step chunks blinds the model to the lifecycle of farming, causing erratic re-routing and unharvested crops. `H6` earned $1,620 less than `H24` despite consuming 4x more GPU compute.
2. **Coherent Routine Commitment**:
   - `H24` commits to a full-day agricultural loop (buy seeds $\to$ pathfind to field $\to$ plant $\to$ water $\to$ return to shed). This yielded maximum consistent earnings ($2,820 in every game).
3. **Hybrid Resilience**:
   - `Hybrid` combines 24-step strategic trajectories with reactive interrupts when weeds spawn or the opponent executes a large market transaction.
