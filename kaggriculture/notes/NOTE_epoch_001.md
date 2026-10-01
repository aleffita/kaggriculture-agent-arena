# Dream-RSI Empirical Debrief: Epoch 1
- **Date**: 2026-09-30 07:28:55 UTC
- **Total Fixtures Analyzed**: 12

## 1. Match Outlines
| Stage | Player 0 | Player 1 | Winner | P0 Bank | P1 Bank | Margin |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| Sprint | `llm_sprint_rusher_v1` | `llm_melon_monopolist_v1` | **`llm_sprint_rusher_v1`** | $2,880 | $2,040 | +$840 |
| Sprint | `llm_cautious_farmer_v1` | `llm_land_baron_v1` | **`llm_land_baron_v1`** | $2,640 | $2,820 | +$180 |
| Sprint | `llm_market_arbitrageur_v1` | `llm_labor_magnate_v1` | **`llm_labor_magnate_v1`** | $2,820 | $2,880 | +$60 |
| Expansion | `llm_land_baron_v1` | `llm_market_arbitrageur_v1` | **`llm_land_baron_v1`** | $2,820 | $2,640 | +$180 |
| Expansion | `llm_sprint_rusher_v1` | `llm_melon_monopolist_v1` | **`llm_sprint_rusher_v1`** | $2,820 | $1,560 | +$1,260 |
| Expansion | `llm_labor_magnate_v1` | `llm_cautious_farmer_v1` | **`llm_labor_magnate_v1`** | $2,820 | $2,280 | +$540 |
| Scaling | `llm_melon_monopolist_v1` | `llm_market_arbitrageur_v1` | **`llm_market_arbitrageur_v1`** | $600 | $2,400 | +$1,800 |
| Scaling | `llm_labor_magnate_v1` | `llm_land_baron_v1` | **`Draw`** | $2,700 | $2,700 | +$0 |
| Scaling | `llm_sprint_rusher_v1` | `llm_cautious_farmer_v1` | **`llm_sprint_rusher_v1`** | $1,980 | $1,800 | +$180 |
| FullSeason | `llm_land_baron_v1` | `llm_sprint_rusher_v1` | **`Draw`** | $2,100 | $2,100 | +$0 |
| FullSeason | `llm_cautious_farmer_v1` | `llm_labor_magnate_v1` | **`llm_labor_magnate_v1`** | $0 | $2,100 | +$2,100 |
| FullSeason | `llm_market_arbitrageur_v1` | `llm_melon_monopolist_v1` | **`llm_market_arbitrageur_v1`** | $1,200 | $0 | +$1,200 |

## 2. Identified Tactical Pathologies & Corrections
- **llm_melon_monopolist_v1** suffered from low liquidity (Avg Bank: $600). Refined rule: enforce immediate shed drops and minimum cash reserve.
- **llm_cautious_farmer_v1** suffered from low liquidity (Avg Bank: $1320). Refined rule: enforce immediate shed drops and minimum cash reserve.
- **llm_market_arbitrageur_v1** suffered from low liquidity (Avg Bank: $2820). Refined rule: enforce immediate shed drops and minimum cash reserve.