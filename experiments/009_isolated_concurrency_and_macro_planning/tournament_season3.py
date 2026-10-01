"""Tournament Season 3: Scaling Curves, Dual-Engine VRAM, and DeepSeek HCA-CSA Architecture.

Features:
1. Dual LiteRT-LM Engines in GTX 1050 Ti VRAM (Engine 0 for P0, Engine 1 for P1, ~2,060 MB total).
2. DeepSeek HCA-CSA Architecture:
   - HCA (Hierarchical Context Attention): Long-range macro campaign policy (96-step compressed strategic intent).
   - CSA (Cross-Scale Action): High-resolution 24-step local execution cross-attending to HCA intent.
3. Scaling Horizons: H24, H48 (wheat cycle), H96 (crop rotation), H256 (macro campaign via RLE).
4. Debounce Filter on all Hybrid variants: max(6, horizon // 4) to eliminate chattering.
5. Compact 3-Round Swiss Tournament at 720 steps (30 in-game days).
"""

from __future__ import annotations

import collections
import itertools
import json
import os
import pathlib
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from kaggle_environments import make
from src.litert_explore.engine import LiteRtModelRunner
from kaggriculture.agents.personality_agent import load_personality_prompt


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


def make_season3_agent(name: str, horizon: int, is_hybrid: bool = False, is_hca_csa: bool = False, engine_runner: Any = None):
    """Factory creating Season 3 agents with dual-engine binding, debounce, RLE, or HCA-CSA."""
    personality_core = load_personality_prompt("sprint_rusher")
    queue: collections.deque = collections.deque()
    debounce_threshold = max(6, horizon // 4)

    stats = {
        "name": name,
        "horizon": horizon,
        "is_hybrid": is_hybrid,
        "is_hca_csa": is_hca_csa,
        "inferences": 0,
        "replan_events": 0,
        "steps_since_replan": 999,
        "inference_latencies": [],
    }

    # HCA persistent strategic state
    hca_state = {
        "macro_objective": "MAXIMIZE_WHEAT_PRODUCTION",
        "primary_crop": "WHEAT",
        "last_hca_step": -999,
    }

    prev_opp_money = [100.0]

    # Schema for RLE (for horizons >= 96) vs direct actions (< 96)
    use_rle = (horizon >= 96 and not is_hca_csa)

    if use_rle:
        plan_schema = {
            "type": "object",
            "properties": {
                "rle_plan": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "minItems": 4,
                    "maxItems": 16,
                },
                "market_seed": {"type": "string"},
            },
            "required": ["rle_plan"],
        }
    elif is_hca_csa:
        # HCA-CSA dual-tier schema
        plan_schema = {
            "type": "object",
            "properties": {
                "macro_objective": {"type": "string"},
                "csa_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 24,
                    "maxItems": 24,
                },
                "market_seed": {"type": "string"},
            },
            "required": ["csa_actions"],
        }
    else:
        plan_schema = {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": min(horizon, 48),
                    "maxItems": min(horizon, 48),
                },
                "market_seed": {"type": "string"},
            },
            "required": ["actions"],
        }

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

        stats["steps_since_replan"] += 1

        # Check for event triggers in Hybrid or HCA-CSA
        triggered_replan = False
        replan_reason = ""
        can_trigger = (stats["steps_since_replan"] >= debounce_threshold)

        if (is_hybrid or is_hca_csa) and can_trigger and len(queue) > 0:
            if isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
                triggered_replan = True
                replan_reason = "WEED_SPAWN"
            elif isinstance(current_tile, dict) and current_tile.get("yield_units", 0) > 0:
                triggered_replan = True
                replan_reason = "HARVEST_READY"
            elif (opp_money - prev_opp_money[0]) >= 200:
                triggered_replan = True
                replan_reason = "OPPONENT_SPIKE"

        prev_opp_money[0] = opp_money

        if triggered_replan:
            stats["replan_events"] += 1
            stats["steps_since_replan"] = 0
            queue.clear()

        if len(queue) > 0:
            return queue.popleft()

        # Re-plan needed
        stats["inferences"] += 1
        stats["steps_since_replan"] = 0
        t_start = time.time()

        # HCA layer update (every 96 steps for HCA-CSA)
        if is_hca_csa and (step - hca_state["last_hca_step"] >= 96 or hca_state["last_hca_step"] < 0):
            hca_state["last_hca_step"] = step
            if money >= 1050:
                hca_state["macro_objective"] = "UNLOCK_NE_TERRITORY_AND_SCALE_MELON"
                hca_state["primary_crop"] = "MELON"
            elif money >= 400:
                hca_state["macro_objective"] = "EXPAND_FARM_PRODUCTION_WHEAT_AND_CARROT"
                hca_state["primary_crop"] = "CARROT"
            else:
                hca_state["macro_objective"] = "ACCUMULATE_CASH_AND_FAST_WHEAT_CYCLES"
                hca_state["primary_crop"] = "WHEAT"

        reason_str = f" [EVENT RE-PLAN: {replan_reason}]" if triggered_replan else ""
        hca_context = f"\n[HCA MACRO-INTENT]: {hca_state['macro_objective']} (Crop Focus: {hca_state['primary_crop']})" if is_hca_csa else ""

        prompt = (
            f"{personality_core}{hca_context}\n\n"
            f"STRATEGIC BRIEFING (Day {day}, Hour {hour} | Step {step}){reason_str}:\n"
            f"- Your Cash: ${money:.0f} | Opponent Cash: ${opp_money:.0f}\n"
            f"- Farmer Position: ({fx},{fy})\n"
            f"- Seeds: {seeds} (Total: {total_seeds})\n"
            f"- Shed Inventory: {shed}\n"
            f"- Market Prices: Wheat=${prices.get('WHEAT', 25)}, Carrot=${prices.get('CARROT', 35)}, Melon=${prices.get('MELON', 250)}\n\n"
        )

        if use_rle:
            prompt += (
                f"Plan horizon of {horizon} actions using Run-Length Encoding (RLE).\n"
                f"Format 'rle_plan': [['ACTION', 'REPEAT_COUNT'], ...] (e.g. [['NORTH', '4'], ['PLANT_WHEAT', '6'], ...]).\n"
                f"Reply ONLY in JSON: {{\"rle_plan\": [['ACTION', 'N'], ...], \"market_seed\": \"WHEAT\"}}"
            )
        elif is_hca_csa:
            prompt += (
                f"Align with HCA intent. Plan next 24 CSA actions.\n"
                f"Reply ONLY in JSON: {{\"macro_objective\": \"...\", \"csa_actions\": [\"ACT_0\", ... \"ACT_23\"], \"market_seed\": \"{hca_state['primary_crop']}\"}}"
            )
        else:
            prompt += (
                f"Plan next {min(horizon, 48)} actions.\n"
                f"Reply ONLY in JSON: {{\"actions\": [\"ACT_0\", ...], \"market_seed\": \"WHEAT\"}}"
            )

        try:
            raw = engine_runner.generate_structured(prompt, plan_schema)
            dur = time.time() - t_start
            stats["inference_latencies"].append(dur)
            parsed = json.loads(raw)

            if use_rle:
                rle_list = parsed.get("rle_plan", [])
                expanded = []
                for item in rle_list:
                    if isinstance(item, list) and len(item) >= 2:
                        act_name = str(item[0])
                        try:
                            count = min(int(item[1]), 32)
                        except Exception:
                            count = 1
                        expanded.extend([act_name] * max(1, count))
                actions_list = expanded if expanded else ["PASS"] * horizon
                seed_choice = parsed.get("market_seed", "WHEAT").upper()
            elif is_hca_csa:
                actions_list = parsed.get("csa_actions", ["PASS"] * 24)
                seed_choice = parsed.get("market_seed", hca_state["primary_crop"]).upper()
            else:
                actions_list = parsed.get("actions", ["PASS"] * horizon)
                seed_choice = parsed.get("market_seed", "WHEAT").upper()

        except Exception:
            dur = time.time() - t_start
            stats["inference_latencies"].append(dur)
            actions_list = ["PASS"] * (24 if is_hca_csa else horizon)
            seed_choice = "WHEAT"

        # Market orders
        market_orders = []
        if money >= 40 and seed_choice in ("WHEAT", "CARROT", "MELON"):
            market_orders.append(["BUY_SEED", seed_choice, 6])
        for crop_name in ("WHEAT", "CARROT", "MELON"):
            qty = shed.get(crop_name, 0)
            if qty > 0:
                market_orders.append(["SELL", crop_name, qty])

        for idx, act_name in enumerate(actions_list):
            parsed_farmer = parse_action_string(act_name)
            order_for_step = market_orders if idx == 0 else []
            queue.append({"farmer": parsed_farmer, "market": order_for_step})

        while len(queue) < 1:
            queue.append({"farmer": ["PASS"], "market": []})

        return queue.popleft()

    agent.stats = stats
    return agent


def execute_season3_match(
    p0_info: dict,
    p1_info: dict,
    engine_p0: Any,
    engine_p1: Any,
    seed: int,
    steps: int = 720,
) -> dict:
    """Executes a 720-step match binding Engine 0 to Player 0 and Engine 1 to Player 1."""
    ag0 = make_season3_agent(
        name=p0_info["name"],
        horizon=p0_info["horizon"],
        is_hybrid=p0_info.get("is_hybrid", False),
        is_hca_csa=p0_info.get("is_hca_csa", False),
        engine_runner=engine_p0,
    )
    ag1 = make_season3_agent(
        name=p1_info["name"],
        horizon=p1_info["horizon"],
        is_hybrid=p1_info.get("is_hybrid", False),
        is_hca_csa=p1_info.get("is_hca_csa", False),
        engine_runner=engine_p1,
    )

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
        winner = p0_info["name"]
    elif p1_bank > p0_bank:
        score0, score1 = 0.0, 1.0
        winner = p1_info["name"]
    else:
        score0, score1 = 0.5, 0.5
        winner = "Draw"

    return {
        "p0": p0_info["name"],
        "p1": p1_info["name"],
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


def run_season3():
    print("=" * 80)
    print("SEASON 3: SCALING TOURNAMENT (720 STEPS | DUAL ENGINES | DEEPSEEK HCA-CSA)")
    print("=" * 80)

    # 1. Initialize Dual LiteRT-LM Engines in GTX 1050 Ti VRAM
    print(f"[0] Initializing Dual LiteRT-LM Engines on GPU 1 (GTX 1050 Ti)...")
    t_load_0 = time.time()
    engine_0 = LiteRtModelRunner(backend="gpu", gpu_target="1050ti", max_num_tokens=512)
    print(f"    Engine 0 loaded in {time.time() - t_load_0:.2f}s | VRAM: {get_vram()}")

    t_load_1 = time.time()
    engine_1 = LiteRtModelRunner(backend="gpu", gpu_target="1050ti", max_num_tokens=512)
    print(f"    Engine 1 loaded in {time.time() - t_load_1:.2f}s | VRAM: {get_vram()}")
    print(f"    [OK] Dual Engines Ready! Both players evaluate simultaneously on GPU.\n")

    # 2. Competitors
    competitors = {
        "H24":         {"name": "H24",         "horizon": 24,  "is_hybrid": False, "is_hca_csa": False},
        "H24-Hybrid":  {"name": "H24-Hybrid",  "horizon": 24,  "is_hybrid": True,  "is_hca_csa": False},
        "H48":         {"name": "H48",         "horizon": 48,  "is_hybrid": False, "is_hca_csa": False},
        "H48-Hybrid":  {"name": "H48-Hybrid",  "horizon": 48,  "is_hybrid": True,  "is_hca_csa": False},
        "H96":         {"name": "H96",         "horizon": 96,  "is_hybrid": False, "is_hca_csa": False},
        "H96-Hybrid":  {"name": "H96-Hybrid",  "horizon": 96,  "is_hybrid": True,  "is_hca_csa": False},
        "H256-Hybrid": {"name": "H256-Hybrid", "horizon": 256, "is_hybrid": True,  "is_hca_csa": False},
        "HCA-CSA":     {"name": "HCA-CSA",     "horizon": 96,  "is_hybrid": True,  "is_hca_csa": True},
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

    # 3. Compact 3-Round Swiss Tournament (4 matches per round = 12 matches total)
    # Allows fast scaling validation at full 720 steps within ~35-40 min!
    num_rounds = 3
    swiss_seeds = [42, 101, 777]
    match_logs = []
    played_pairs = set()

    print(f"Torneio Suíço Compacto: {len(competitors)} concorrentes | {num_rounds} rodadas | 720 passos por partida\n")

    t_tourney_start = time.time()

    for r_num in range(1, num_rounds + 1):
        round_seed = swiss_seeds[r_num - 1]
        print(f"--- RODADA {r_num}/{num_rounds} (Seed: {round_seed} | Horizonte: 720 Passos / 30 Dias) ---")

        # Sort by current Elo
        sorted_keys = sorted(competitors.keys(), key=lambda k: leaderboard[k]["elo"], reverse=True)
        unpaired = list(sorted_keys)
        round_pairs = []

        while len(unpaired) >= 2:
            p0 = unpaired.pop(0)
            best_idx = 0
            for idx, cand in enumerate(unpaired):
                if (p0, cand) not in played_pairs and (cand, p0) not in played_pairs:
                    best_idx = idx
                    break
            p1 = unpaired.pop(best_idx)
            round_pairs.append((p0, p1))
            played_pairs.add((p0, p1))

        for m_idx, (p0_key, p1_key) in enumerate(round_pairs, start=1):
            e0_b = leaderboard[p0_key]["elo"]
            e1_b = leaderboard[p1_key]["elo"]

            print(f"  [R{r_num} Duelo {m_idx}/4] {p0_key} ({e0_b:.0f}) vs {p1_key} ({e1_b:.0f})...", flush=True)

            res = execute_season3_match(
                competitors[p0_key], competitors[p1_key],
                engine_0, engine_1, seed=round_seed, steps=720
            )

            ne0, ne1 = compute_elo_update(e0_b, e1_b, res["score0"])
            leaderboard[p0_key]["elo"] = ne0
            leaderboard[p1_key]["elo"] = ne1
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

            res["round"] = r_num
            res["elo_p0_after"] = ne0
            res["elo_p1_after"] = ne1
            match_logs.append(res)

            print(f"    -> Vencedor: {res['winner']} (${res['p0_bank']:.0f} vs ${res['p1_bank']:.0f}) em {res['duration_s']:.1f}s | Infs: {res['inf0']} vs {res['inf1']} (Evs: {res['ev0']}/{res['ev1']}) | Elos: {p0_key}={ne0:.0f}, {p1_key}={ne1:.0f}", flush=True)

        print()

    t_tourney_total = time.time() - t_tourney_start

    print("=" * 80)
    print("CLASSIFICAÇÃO FINAL DA SEASON 3 (HORIZONTE 720 PASSOS)")
    print("=" * 80)
    sorted_board = sorted(leaderboard.values(), key=lambda x: x["elo"], reverse=True)
    header = f"{'Pos':<3} | {'Estratégia':<14} | {'Elo':<6} | {'V-E-D':<7} | {'Moedas':<10} | {'Inferências':<11} | {'Eficiência ($/inf)':<18}"
    print(header)
    print("-" * len(header))
    for rank, b in enumerate(sorted_board, start=1):
        ved = f"{b['wins']}-{b['draws']}-{b['losses']}"
        inf = b["total_inferences"]
        eff = (b["total_coins"] / inf) if inf > 0 else 0.0
        print(f"{rank:<3} | {b['name']:<14} | {b['elo']:<6.1f} | {ved:<7} | ${b['total_coins']:<9.0f} | {inf:<11} | ${eff:<17.1f}")

    print(f"\nTempo Total da Season 3: {t_tourney_total:.1f}s ({t_tourney_total/60:.2f} min) | VRAM: {get_vram()}")

    # Persist Results
    out_file = pathlib.Path(__file__).parent / "season3_results.json"
    data = {
        "seeds": swiss_seeds,
        "matches_count": len(match_logs),
        "tournament_duration_s": round(t_tourney_total, 2),
        "leaderboard": sorted_board,
        "matches": match_logs,
    }
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n[OK] Resultados da Season 3 salvos em: {out_file}")


if __name__ == "__main__":
    run_season3()
