"""Tournament Season 2: Multi-Horizon & Hybrid Macro-Planning on NVIDIA GTX 1050 Ti.

Architecture:
- 6 Competitors: H24, H24-Hybrid, H12, H12-Hybrid, H6, H6-Hybrid
- Event Triggers (Hybrid):
  1. Weed spawn on active tile (WEED_SPAWN)
  2. Crop ready for harvest (HARVEST_READY)
  3. Opponent cash jump >= $150 (OPPONENT_SPIKE)
- Phase 1: Round-Robin across 4 controlled seeds [42, 101, 777, 2026] at 144 steps (6 days)
  Total: 15 matchups * 4 seeds = 60 matches
- Phase 2: Top-4 Championship Playoffs at full 720 steps (30 days)
  Semifinals + Finals / 3rd Place Match
- Tracks Elo, Win/Loss/Draw, coin accumulation, inferences, and efficiency ($/inf).
"""

from __future__ import annotations

import collections
import itertools
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from kaggle_environments import make
from src.litert_explore.engine import LiteRtModelRunner
from kaggriculture.agents.llm_player import get_llm_runner
from kaggriculture.agents.personality_agent import load_personality_prompt


SEEDS = [42, 101, 777, 2026]


def get_vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--id=1", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            encoding="utf-8", errors="ignore"
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return f"{parts[0]}/{parts[1]} MB (Util: {parts[2]}%)"
    except Exception:
        return "N/A"


def parse_action_string(act_str: str) -> List[str]:
    act = act_str.strip().upper()
    if act.startswith("PLANT_"):
        crop = act.replace("PLANT_", "")
        return ["PLANT", crop]
    if act in ("NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "DIG", "DROP", "PASS"):
        return [act]
    if "PLANT" in act:
        return ["PLANT", "WHEAT"]
    return ["PASS"]


def compute_elo_update(elo_a: float, elo_b: float, score_a: float, k: float = 32.0) -> Tuple[float, float]:
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    expected_b = 1.0 - expected_a
    new_elo_a = elo_a + k * (score_a - expected_a)
    new_elo_b = elo_b + k * ((1.0 - score_a) - expected_b)
    return round(new_elo_a, 1), round(new_elo_b, 1)


def make_season2_agent(name: str, horizon: int, is_hybrid: bool = False):
    personality_core = load_personality_prompt("sprint_rusher")
    queue: collections.deque = collections.deque()
    stats = {
        "name": name,
        "horizon": horizon,
        "is_hybrid": is_hybrid,
        "inferences": 0,
        "replan_events": 0,
        "inference_latencies": [],
    }

    schema = {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": horizon,
                "maxItems": horizon,
            },
            "market_seed": {"type": "string"},
        },
        "required": ["actions"],
    }

    prev_opp_money = [100.0]

    def agent(obs: dict) -> dict:
        player = obs["player"]
        opp_player = 1 - player
        my_farm = obs["farms"][player]
        opp_farm = obs["farms"][opp_player]
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        step = obs.get("step", 0)
        money = my_farm.get("money", 0)
        opp_money = opp_farm.get("money", 0)
        fx, fy = my_farm["farmer"]
        current_tile = my_farm["tiles"][fy][fx]
        seeds = my_farm.get("seeds", {})
        total_seeds = sum(seeds.values())
        shed = my_farm.get("shed", {})
        market = obs.get("market", {})
        prices = market.get("prices", {})

        # Event triggers for Hybrid variants
        triggered_replan = False
        replan_reason = ""
        if is_hybrid and len(queue) > 0:
            if isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
                triggered_replan = True
                replan_reason = "WEED_SPAWN"
            elif isinstance(current_tile, dict) and current_tile.get("yield_units", 0) > 0:
                triggered_replan = True
                replan_reason = "HARVEST_READY"
            elif (opp_money - prev_opp_money[0]) >= 150:
                triggered_replan = True
                replan_reason = "OPPONENT_SPIKE"

        prev_opp_money[0] = opp_money

        if triggered_replan:
            stats["replan_events"] += 1
            queue.clear()

        if len(queue) > 0:
            return queue.popleft()

        stats["inferences"] += 1
        t_start = time.time()
        runner = get_llm_runner()

        reason_str = f" [EVENT RE-PLAN: {replan_reason}]" if triggered_replan else ""
        prompt = (
            f"{personality_core}\n\n"
            f"STRATEGIC BRIEFING (Day {day}, Hour {hour} | Step {step}){reason_str}:\n"
            f"- Your Cash: ${money:.0f} | Opponent Cash: ${opp_money:.0f}\n"
            f"- Farmer Position: ({fx},{fy})\n"
            f"- Seeds: {seeds} (Total: {total_seeds})\n"
            f"- Shed Inventory: {shed}\n"
            f"- Market Prices: Wheat=${prices.get('WHEAT', 25)}, Carrot=${prices.get('CARROT', 35)}, Melon=${prices.get('MELON', 250)}\n\n"
            f"DIRECTIVES FOR NEXT {horizon} ACTIONS:\n"
            f"1. Choose {horizon} actions from: NORTH, SOUTH, EAST, WEST, PLANT_WHEAT, WATER, HARVEST, DIG, DROP, PASS.\n"
            f"2. Indicate seed purchase in 'market_seed': 'WHEAT', 'CARROT', 'MELON', or 'NONE'.\n\n"
            f"Reply ONLY in JSON: {{\"actions\": [\"ACT_0\", ...], \"market_seed\": \"WHEAT\"}}"
        )

        try:
            raw = runner.generate_structured(prompt, schema)
            dur = time.time() - t_start
            stats["inference_latencies"].append(dur)
            parsed = json.loads(raw)
            actions_list = parsed.get("actions", [])
            seed_choice = parsed.get("market_seed", "WHEAT").upper()
        except Exception:
            dur = time.time() - t_start
            stats["inference_latencies"].append(dur)
            actions_list = ["PASS"] * horizon
            seed_choice = "WHEAT"

        market_orders = []
        if money >= 40 and seed_choice in ("WHEAT", "CARROT", "MELON"):
            market_orders.append(["BUY_SEED", seed_choice, 6])
        for crop_name in ("WHEAT", "CARROT", "MELON"):
            qty = shed.get(crop_name, 0)
            if qty > 0:
                market_orders.append(["SELL", crop_name, qty])

        for idx, act_name in enumerate(actions_list[:horizon]):
            parsed_farmer = parse_action_string(act_name)
            order_for_step = market_orders if idx == 0 else []
            queue.append({"farmer": parsed_farmer, "market": order_for_step})

        while len(queue) < 1:
            queue.append({"farmer": ["PASS"], "market": []})

        return queue.popleft()

    agent.stats = stats
    return agent


def execute_duel(p0_key: str, p1_key: str, competitors: dict, seed: int, steps: int) -> dict:
    c0 = competitors[p0_key]
    c1 = competitors[p1_key]

    ag0 = make_season2_agent(p0_key, horizon=c0["horizon"], is_hybrid=c0["is_hybrid"])
    ag1 = make_season2_agent(p1_key, horizon=c1["horizon"], is_hybrid=c1["is_hybrid"])

    env = make("kaggriculture", configuration={"episodeSteps": steps, "actTimeout": 60, "runTimeout": 3600, "randomSeed": seed})

    t0 = time.time()
    env.run([ag0, ag1])
    duration = time.time() - t0

    final_step = env.steps[-1]
    obs_last = final_step[0].get("observation", {})
    farms_last = obs_last.get("farms", [{}, {}])
    p0_bank = float(farms_last[0].get("money", 0.0))
    p1_bank = float(farms_last[1].get("money", 0.0))

    inf0 = ag0.stats["inferences"]
    inf1 = ag1.stats["inferences"]
    ev0 = ag0.stats["replan_events"]
    ev1 = ag1.stats["replan_events"]

    if p0_bank > p1_bank:
        score0, score1 = 1.0, 0.0
        winner = p0_key
    elif p1_bank > p0_bank:
        score0, score1 = 0.0, 1.0
        winner = p1_key
    else:
        score0, score1 = 0.5, 0.5
        winner = "Draw"

    return {
        "p0": p0_key,
        "p1": p1_key,
        "seed": seed,
        "steps": steps,
        "winner": winner,
        "p0_bank": p0_bank,
        "p1_bank": p1_bank,
        "score0": score0,
        "score1": score1,
        "inf0": inf0,
        "inf1": inf1,
        "ev0": ev0,
        "ev1": ev1,
        "duration_s": round(duration, 2),
    }


def run_season2():
    print("=" * 80)
    print("SEASON 2: MULTI-HORIZON & HYBRID TOURNAMENT ON NVIDIA GTX 1050 Ti")
    print("=" * 80)

    competitors = {
        "H24":        {"horizon": 24, "is_hybrid": False, "desc": "Macro-24 Fixo"},
        "H24-Hybrid": {"horizon": 24, "is_hybrid": True,  "desc": "Macro-24 Híbrido (Eventos)"},
        "H12":        {"horizon": 12, "is_hybrid": False, "desc": "Macro-12 Fixo"},
        "H12-Hybrid": {"horizon": 12, "is_hybrid": True,  "desc": "Macro-12 Híbrido (Eventos)"},
        "H6":         {"horizon": 6,  "is_hybrid": False, "desc": "Macro-6 Fixo"},
        "H6-Hybrid":  {"horizon": 6,  "is_hybrid": True,  "desc": "Macro-6 Híbrido (Eventos)"},
    }

    leaderboard = {
        name: {
            "name": name,
            "elo": 600.0,
            "matches": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "total_coins": 0.0,
            "total_inferences": 0,
            "replan_events": 0,
        }
        for name in competitors
    }

    # Pre-warm Engine
    print(f"[0] Pre-warming LiteRT-LM on GPU 1 (GTX 1050 Ti)... {get_vram()}")
    get_llm_runner()
    print(f"    Engine Ready! {get_vram()}\n")

    # =========================================================================
    # PHASE 1: ROUND-ROBIN (144 STEPS / 6 DAYS across 4 SEEDS)
    # =========================================================================
    pairs = list(itertools.combinations(competitors.keys(), 2))
    total_rr_matches = len(pairs) * len(SEEDS)
    print(f"--- FASE 1: ROUND-ROBIN (144 Passos | 6 Dias | 4 Seeds Fixas) ---")
    print(f"Confrontos: {len(pairs)} pares * {len(SEEDS)} seeds = {total_rr_matches} partidas\n")

    phase1_logs = []
    match_counter = 1

    t_phase1_start = time.time()
    for (p0_key, p1_key) in pairs:
        for seed in SEEDS:
            elo0_b = leaderboard[p0_key]["elo"]
            elo1_b = leaderboard[p1_key]["elo"]

            print(f"[RR {match_counter}/{total_rr_matches}] {p0_key} ({elo0_b:.0f}) vs {p1_key} ({elo1_b:.0f}) | Seed: {seed}...", flush=True)

            res = execute_duel(p0_key, p1_key, competitors, seed=seed, steps=144)

            # Update Leaderboard & Elo
            new_elo0, new_elo1 = compute_elo_update(elo0_b, elo1_b, res["score0"])
            leaderboard[p0_key]["elo"] = new_elo0
            leaderboard[p1_key]["elo"] = new_elo1
            leaderboard[p0_key]["matches"] += 1
            leaderboard[p1_key]["matches"] += 1
            leaderboard[p0_key]["total_coins"] += res["p0_bank"]
            leaderboard[p1_key]["total_coins"] += res["p1_bank"]
            leaderboard[p0_key]["total_inferences"] += res["inf0"]
            leaderboard[p1_key]["total_inferences"] += res["inf1"]
            leaderboard[p0_key]["replan_events"] += res["ev0"]
            leaderboard[p1_key]["replan_events"] += res["ev1"]

            if res["score0"] == 1.0:
                leaderboard[p0_key]["wins"] += 1
                leaderboard[p1_key]["losses"] += 1
            elif res["score1"] == 1.0:
                leaderboard[p1_key]["wins"] += 1
                leaderboard[p0_key]["losses"] += 1
            else:
                leaderboard[p0_key]["draws"] += 1
                leaderboard[p1_key]["draws"] += 1

            res["elo_p0_after"] = new_elo0
            res["elo_p1_after"] = new_elo1
            phase1_logs.append(res)

            print(f"    -> Vencedor: {res['winner']} (${res['p0_bank']:.0f} vs ${res['p1_bank']:.0f}) em {res['duration_s']}s | Infs: {res['inf0']} vs {res['inf1']} | Elos: {p0_key}={new_elo0:.0f}, {p1_key}={new_elo1:.0f}", flush=True)
            match_counter += 1

    t_phase1_total = time.time() - t_phase1_start

    print("\n" + "=" * 80)
    print("CLASSIFICAÇÃO FINAL DA FASE 1 (ROUND-ROBIN 144 PASSOS)")
    print("=" * 80)
    sorted_p1 = sorted(leaderboard.values(), key=lambda x: x["elo"], reverse=True)
    header = f"{'Pos':<3} | {'Estratégia':<12} | {'Elo':<6} | {'V-E-D':<9} | {'Moedas':<10} | {'Inferências':<11} | {'Eficiência ($/inf)':<18}"
    print(header)
    print("-" * len(header))
    for rank, b in enumerate(sorted_p1, start=1):
        ved = f"{b['wins']}-{b['draws']}-{b['losses']}"
        inf = b["total_inferences"]
        eff = (b["total_coins"] / inf) if inf > 0 else 0.0
        print(f"{rank:<3} | {b['name']:<12} | {b['elo']:<6.1f} | {ved:<9} | ${b['total_coins']:<9.0f} | {inf:<11} | ${eff:<17.1f}")
    print(f"\nTempo Total da Fase 1: {t_phase1_total:.1f}s ({t_phase1_total/60:.2f} min)\n")

    # =========================================================================
    # PHASE 2: TOP-4 PLAYOFFS (720 STEPS / 30 DAYS FULL SEASON)
    # =========================================================================
    top4 = [b["name"] for b in sorted_p1[:4]]
    print("=" * 80)
    print(f"--- FASE 2: PLAYOFFS DO CAMPEONATO (720 Passos | 30 Dias) ---")
    print(f"Qualificados para as Finais: {top4}")
    print("=" * 80)

    # Semifinal 1: 1º vs 4º | Semifinal 2: 2º vs 3º
    semi1 = (top4[0], top4[3])
    semi2 = (top4[1], top4[2])

    playoff_logs = []
    championship_seed = 42

    print(f"\n[SEMIFINAL 1 (720 steps)] {semi1[0]} vs {semi1[1]} (Seed {championship_seed})...", flush=True)
    res_semi1 = execute_duel(semi1[0], semi1[1], competitors, seed=championship_seed, steps=720)
    winner_semi1 = res_semi1["winner"]
    loser_semi1 = semi1[1] if winner_semi1 == semi1[0] else semi1[0]
    print(f"  -> Vencedor Semi 1: {winner_semi1} (${res_semi1['p0_bank']:.0f} vs ${res_semi1['p1_bank']:.0f}) em {res_semi1['duration_s']}s")
    playoff_logs.append({"round": "Semifinal 1", **res_semi1})

    print(f"\n[SEMIFINAL 2 (720 steps)] {semi2[0]} vs {semi2[1]} (Seed {championship_seed})...", flush=True)
    res_semi2 = execute_duel(semi2[0], semi2[1], competitors, seed=championship_seed, steps=720)
    winner_semi2 = res_semi2["winner"]
    loser_semi2 = semi2[1] if winner_semi2 == semi2[0] else semi2[0]
    print(f"  -> Vencedor Semi 2: {winner_semi2} (${res_semi2['p0_bank']:.0f} vs ${res_semi2['p1_bank']:.0f}) em {res_semi2['duration_s']}s")
    playoff_logs.append({"round": "Semifinal 2", **res_semi2})

    # Disputa do 3º Lugar
    print(f"\n[DISPUTA 3º LUGAR (720 steps)] {loser_semi1} vs {loser_semi2} (Seed {championship_seed})...", flush=True)
    res_3rd = execute_duel(loser_semi1, loser_semi2, competitors, seed=championship_seed, steps=720)
    third_place = res_3rd["winner"]
    fourth_place = loser_semi2 if third_place == loser_semi1 else loser_semi1
    print(f"  -> 3º Lugar: {third_place} (${res_3rd['p0_bank']:.0f} vs ${res_3rd['p1_bank']:.0f}) em {res_3rd['duration_s']}s")
    playoff_logs.append({"round": "3rd Place Match", **res_3rd})

    # Grande Final
    print(f"\n[GRANDE FINAL (720 steps)] {winner_semi1} vs {winner_semi2} (Seed {championship_seed})...", flush=True)
    res_final = execute_duel(winner_semi1, winner_semi2, competitors, seed=championship_seed, steps=720)
    champion = res_final["winner"]
    vice = winner_semi2 if champion == winner_semi1 else winner_semi1
    print(f"  -> CAMPEÃO DA SEASON 2: {champion} (${res_final['p0_bank']:.0f} vs ${res_final['p1_bank']:.0f}) em {res_final['duration_s']}s")
    playoff_logs.append({"round": "Grand Final", **res_final})

    # Update Elo for Playoff Matches
    for pl in playoff_logs:
        p0, p1 = pl["p0"], pl["p1"]
        e0, e1 = leaderboard[p0]["elo"], leaderboard[p1]["elo"]
        ne0, ne1 = compute_elo_update(e0, e1, pl["score0"])
        leaderboard[p0]["elo"] = ne0
        leaderboard[p1]["elo"] = ne1
        leaderboard[p0]["total_coins"] += pl["p0_bank"]
        leaderboard[p1]["total_coins"] += pl["p1_bank"]
        leaderboard[p0]["total_inferences"] += pl["inf0"]
        leaderboard[p1]["total_inferences"] += pl["inf1"]
        leaderboard[p0]["replan_events"] += pl["ev0"]
        leaderboard[p1]["replan_events"] += pl["ev1"]
        if pl["score0"] == 1.0:
            leaderboard[p0]["wins"] += 1
            leaderboard[p1]["losses"] += 1
        elif pl["score1"] == 1.0:
            leaderboard[p1]["wins"] += 1
            leaderboard[p0]["losses"] += 1
        else:
            leaderboard[p0]["draws"] += 1
            leaderboard[p1]["draws"] += 1

    # Persist Results
    out_file = pathlib.Path(__file__).parent / "season2_results.json"
    data = {
        "seeds": SEEDS,
        "phase1_matches_count": len(phase1_logs),
        "phase1_duration_s": round(t_phase1_total, 2),
        "leaderboard_phase1": sorted_p1,
        "playoffs": playoff_logs,
        "final_podium": {
            "champion": champion,
            "runner_up": vice,
            "third_place": third_place,
            "fourth_place": fourth_place,
        },
        "final_leaderboard": sorted(leaderboard.values(), key=lambda x: x["elo"], reverse=True),
    }
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n[OK] Resultados da Season 2 salvos em: {out_file}")


if __name__ == "__main__":
    run_season2()
