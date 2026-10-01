# Dream-RSI Empirical Debrief: Epoch 8
- **Date**: 2026-09-30 11:42:21 UTC
- **Total Fixtures Analyzed**: 12

## 1. Match Outlines
| Stage | Player 0 | Player 1 | Winner | P0 Bank | P1 Bank | Margin |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| Sprint | `llm_sprint_rusher_v1` | `llm_land_baron_v1` | **`Draw`** | $2,880 | $2,880 | +$0 |
| Sprint | `llm_labor_magnate_v1` | `llm_cautious_farmer_v1` | **`llm_labor_magnate_v1`** | $2,880 | $2,640 | +$240 |
| Sprint | `llm_market_arbitrageur_v1` | `llm_melon_monopolist_v1` | **`llm_market_arbitrageur_v1`** | $2,820 | $2,040 | +$780 |
| Expansion | `llm_labor_magnate_v1` | `llm_sprint_rusher_v1` | **`llm_sprint_rusher_v1`** | $2,640 | $2,820 | +$180 |
| Expansion | `llm_land_baron_v1` | `llm_cautious_farmer_v1` | **`llm_land_baron_v1`** | $2,820 | $2,280 | +$540 |
| Expansion | `llm_market_arbitrageur_v1` | `llm_melon_monopolist_v1` | **`llm_market_arbitrageur_v1`** | $2,640 | $1,560 | +$1,080 |
| Scaling | `llm_sprint_rusher_v1` | `llm_land_baron_v1` | **`Draw`** | $2,700 | $2,700 | +$0 |
| Scaling | `llm_labor_magnate_v1` | `llm_cautious_farmer_v1` | **`llm_labor_magnate_v1`** | $2,700 | $1,800 | +$900 |
| Scaling | `llm_market_arbitrageur_v1` | `llm_melon_monopolist_v1` | **`llm_market_arbitrageur_v1`** | $2,400 | $600 | +$1,800 |
| FullSeason | `llm_land_baron_v1` | `llm_labor_magnate_v1` | **`llm_labor_magnate_v1`** | $1,380 | $2,100 | +$720 |
| FullSeason | `llm_sprint_rusher_v1` | `llm_market_arbitrageur_v1` | **`llm_sprint_rusher_v1`** | $2,100 | $1,200 | +$900 |
| FullSeason | `llm_melon_monopolist_v1` | `llm_cautious_farmer_v1` | **`Draw`** | $0 | $0 | +$0 |

## 2. Identified Tactical Pathologies & Corrections
- **llm_land_baron_v1** suffered from low liquidity (Avg Bank: $1380). Refined rule: enforce immediate shed drops and minimum cash reserve.
- **llm_labor_magnate_v1** suffered from low liquidity (Avg Bank: $2640). Refined rule: enforce immediate shed drops and minimum cash reserve.