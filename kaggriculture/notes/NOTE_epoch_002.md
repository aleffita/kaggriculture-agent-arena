# Dream-RSI Empirical Debrief: Epoch 2
- **Date**: 2026-09-30 08:08:02 UTC
- **Total Fixtures Analyzed**: 12

## 1. Match Outlines
| Stage | Player 0 | Player 1 | Winner | P0 Bank | P1 Bank | Margin |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| Sprint | `llm_land_baron_v1` | `llm_sprint_rusher_v1` | **`Draw`** | $2,880 | $2,880 | +$0 |
| Sprint | `llm_labor_magnate_v1` | `llm_cautious_farmer_v1` | **`llm_labor_magnate_v1`** | $2,880 | $2,640 | +$240 |
| Sprint | `llm_market_arbitrageur_v1` | `llm_melon_monopolist_v1` | **`llm_market_arbitrageur_v1`** | $2,820 | $2,040 | +$780 |
| Expansion | `llm_labor_magnate_v1` | `llm_sprint_rusher_v1` | **`Draw`** | $2,820 | $2,820 | +$0 |
| Expansion | `llm_land_baron_v1` | `llm_cautious_farmer_v1` | **`llm_land_baron_v1`** | $2,820 | $2,280 | +$540 |
| Expansion | `llm_market_arbitrageur_v1` | `llm_melon_monopolist_v1` | **`llm_market_arbitrageur_v1`** | $2,640 | $1,560 | +$1,080 |
| Scaling | `llm_market_arbitrageur_v1` | `llm_sprint_rusher_v1` | **`llm_sprint_rusher_v1`** | $2,400 | $2,700 | +$300 |
| Scaling | `llm_land_baron_v1` | `llm_labor_magnate_v1` | **`llm_land_baron_v1`** | $2,700 | $2,520 | +$180 |
| Scaling | `llm_cautious_farmer_v1` | `llm_melon_monopolist_v1` | **`llm_cautious_farmer_v1`** | $1,800 | $600 | +$1,200 |
| FullSeason | `llm_labor_magnate_v1` | `llm_market_arbitrageur_v1` | **`llm_market_arbitrageur_v1`** | $0 | $1,200 | +$1,200 |
| FullSeason | `llm_land_baron_v1` | `llm_sprint_rusher_v1` | **`Draw`** | $2,100 | $2,100 | +$0 |
| FullSeason | `llm_cautious_farmer_v1` | `llm_melon_monopolist_v1` | **`Draw`** | $0 | $0 | +$0 |

## 2. Identified Tactical Pathologies & Corrections
- **llm_labor_magnate_v1** suffered from low liquidity (Avg Bank: $0). Refined rule: enforce immediate shed drops and minimum cash reserve.
- **llm_market_arbitrageur_v1** suffered from low liquidity (Avg Bank: $2400). Refined rule: enforce immediate shed drops and minimum cash reserve.