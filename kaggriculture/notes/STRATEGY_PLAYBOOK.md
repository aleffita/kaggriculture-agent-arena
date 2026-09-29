# Kaggriculture Master Strategy Playbook

- **Engine Target**: `kaggle-environments` (`kaggriculture`)
- **Neural Hardware**: NVIDIA GeForce GTX 1050 Ti (Google LiteRT-LM / Direct3D 12)
- **Status**: Living Knowledge Base maintained by Dream-RSI

---

## 1. Golden Tactical Axioms

1. **Immediate Weed Clearance (`DIG`)**:
   - Weeds yield zero value and block tile cultivation. Any turn spent leaving a weed un-dug costs opportunity yield.
2. **Strict Daily Watering (`WATER`)**:
   - Consecutive unwatered turns kill crops. Water must take absolute precedence over movement exploration.
3. **Horizon-Conditioned Crop Selection**:
   - **$T \le 3$ Days (Sprint)**: Pure Wheat (2-day harvest, $10 seed $\to$ $25 sell) or Carrot (3-day harvest, $20 seed $\to$ $35 sell). Never plant Melon.
   - **$T \ge 11$ Days (Scaling/Season)**: Melons compound capital ($80 seed $\to$ $250 sell).
4. **Market Price Damping Avoidance**:
   - Never dump 50+ units in a single market order; prices collapse to $1 floor. Sell in paced batches of 5–8 units per order.
5. **Land Gate Threshold**:
   - Unlocking the NE quadrant ($1,000 cost) requires having at least $1,050 to preserve seed working capital. Execute on Day 4–5.

---

## 2. Competitive Personality Archetypes

| Personality ID | Primary Strength | Known Vulnerability | Target Horizon |
| :--- | :--- | :--- | :---: |
| `sprint_rusher` | High coin velocity in early days | Low ceiling in 30-day macro games | Sprint (72s) |
| `land_baron` | 50-tile double-quadrant leverage | Vulnerable to cash lock before Day 4 | Expansion (144s) |
| `labor_magnate` | High action economy per day | Risk of wage starvation if crops stall | Scaling (240s) |
| `melon_monopolist` | Astronomical late-game coin yield | Fragile cash flow in first 10 days | Full Season (720s) |
| `market_arbitrageur` | Capitalizes on scarcity price spikes | Slower expansion pace | All Stages |
| `cautious_farmer` | Low variance, zero crop loss | Capped total revenue | Sprint / Expansion |
