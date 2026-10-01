"""Mini Tournament: Horizon Ablations (H24 vs H12 vs H6 vs Hybrid) on NVIDIA GTX 1050 Ti.

Executes a full Round-Robin tournament with controlled deterministic seed (seed=42):
- Competitors:
  1. H24 (Macro-24: 1 call per day, 3 calls/match)
  2. H12 (Macro-12: 2 calls per day, 6 calls/match)
  3. H6  (Macro-6:  4 calls per day, 12 calls/match)
  4. Hybrid (Macro-24 base with event-driven re-planning on weed/harvest/opponent spikes)
- Pure LiteRT-LM Direct3D 12 on GTX 1050 Ti with LL_GUIDANCE structured JSON.
- Evaluates Elo impact, coin efficiency, and competitive responsiveness.
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
    """Standard Elo rating update."""
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    expected_b = 1.0 - expected_a
    new_elo_a = elo_a + k * (score_a - expected_a)
    new_elo_b = elo_b + k * ((1.0 - score_a) - expected_b)
    return round(new_elo_a, 1), round(new_elo_b, 1)


def make_ablation_agent(name: str, horizon: int, is_hybrid: bool = False):
    """Creates an ablation agent with parameterized horizon and optional event-driven re-planning."""
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

        # Check for event-driven interruption in Hybrid mode
        triggered_replan = False
        replan_reason = ""
        if is_hybrid and len(queue) > 0:
            # Event 1: Weed spawned on current tile
            if isinstance(current_tile, dict) and current_tile.get("kind") == "WEED":
                triggered_replan = True
                replan_reason = "WEED_SPAWN"
            # Event 2: Crop ready for harvest on current tile
            elif isinstance(current_tile, dict) and current_tile.get("yield_units", 0) > 0:
                triggered_replan = True
                replan_reason = "HARVEST_READY"
            # Event 3: Opponent sudden cash jump > $150 (sold crop)
            elif (opp_money - prev_opp_money[0]) >= 150:
                triggered_replan = True
                replan_reason = "OPPONENT_MARKET_DUMP"

        prev_opp_money[0] = opp_money

        if triggered_replan:
            stats["replan_events"] += 1
            queue.clear()

        # If queue still has actions, pop and execute
        if len(queue) > 0:
            return queue.popleft()

        # Re-planning triggered (horizon exhaustion or event interrupt)
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
            f"2. Indicate daily seed purchase in 'market_seed': 'WHEAT', 'CARROT', 'MELON', or 'NONE'.\n\n"
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


def run_ablation_tournament(seed: int = 42, steps: int = 72):
    print("=" * 80)
    print(f"MINI TORNEIO DE ABLAÇÕES: HORIZONTES DE MACRO-PLANEJAMENTO (SEED={seed})")
    print("=" * 80)

    competitors = {
        "H24": {"horizon": 24, "is_hybrid": False, "desc": "Macro-24 Diário Fixo"},
        "H12": {"horizon": 12, "is_hybrid": False, "desc": "Macro-12 Meio-Dia"},
        "H6":  {"horizon": 6,  "is_hybrid": False, "desc": "Macro-6 Quarto-Dia"},
        "Hybrid": {"horizon": 24, "is_hybrid": True, "desc": "Híbrido (H24 + Eventos)"},
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

    pairs = list(itertools.combinations(competitors.keys(), 2))
    print(f"Total de Duelos no Round-Robin: {len(pairs)} partidas (Passos por Partida: {steps})\n")

    match_logs = []

    for match_idx, (p0_key, p1_key) in enumerate(pairs, start=1):
        c0 = competitors[p0_key]
        c1 = competitors[p1_key]

        ag0 = make_ablation_agent(p0_key, horizon=c0["horizon"], is_hybrid=c0["is_hybrid"])
        ag1 = make_ablation_agent(p1_key, horizon=c1["horizon"], is_hybrid=c1["is_hybrid"])

        elo0_before = leaderboard[p0_key]["elo"]
        elo1_before = leaderboard[p1_key]["elo"]

        print(f"[{match_idx}/{len(pairs)}] DUELO: {p0_key} (Elo: {elo0_before:.0f}) vs {p1_key} (Elo: {elo1_before:.0f})...", flush=True)

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
            leaderboard[p0_key]["wins"] += 1
            leaderboard[p1_key]["losses"] += 1
        elif p1_bank > p0_bank:
            score0, score1 = 0.0, 1.0
            winner = p1_key
            leaderboard[p1_key]["wins"] += 1
            leaderboard[p0_key]["losses"] += 1
        else:
            score0, score1 = 0.5, 0.5
            winner = "Draw"
            leaderboard[p0_key]["draws"] += 1
            leaderboard[p1_key]["draws"] += 1

        new_elo0, new_elo1 = compute_elo_update(elo0_before, elo1_before, score0)
        leaderboard[p0_key]["elo"] = new_elo0
        leaderboard[p1_key]["elo"] = new_elo1
        leaderboard[p0_key]["matches"] += 1
        leaderboard[p1_key]["matches"] += 1
        leaderboard[p0_key]["total_coins"] += p0_bank
        leaderboard[p1_key]["total_coins"] += p1_bank
        leaderboard[p0_key]["total_inferences"] += inf0
        leaderboard[p1_key]["total_inferences"] += inf1
        leaderboard[p0_key]["replan_events"] += ev0
        leaderboard[p1_key]["replan_events"] += ev1

        print(f"    -> Vencedor: {winner} (${p0_bank:.0f} vs ${p1_bank:.0f}) em {duration:.1f}s")
        print(f"    -> Infs: {p0_key}={inf0} (Evs: {ev0}), {p1_key}={inf1} (Evs: {ev1}) | Novo Elo: {p0_key}={new_elo0:.0f}, {p1_key}={new_elo1:.0f}\n", flush=True)

        match_logs.append({
            "match": match_idx,
            "p0": p0_key,
            "p1": p1_key,
            "winner": winner,
            "p0_bank": p0_bank,
            "p1_bank": p1_bank,
            "inf_p0": inf0,
            "inf_p1": inf1,
            "events_p0": ev0,
            "events_p1": ev1,
            "duration_s": round(duration, 2),
            "elo_p0": new_elo0,
            "elo_p1": new_elo1,
        })

    print("=" * 80)
    print("CLASSIFICAÇÃO FINAL DO TORNEIO DE ABLAÇÕES")
    print("=" * 80)
    sorted_board = sorted(leaderboard.values(), key=lambda x: x["elo"], reverse=True)
    header = f"{'Estratégia':<10} | {'Elo':<6} | {'V-E-D':<7} | {'Moedas':<9} | {'Inferências':<11} | {'Eficiência ($/inf)':<18}"
    print(header)
    print("-" * len(header))

    for b in sorted_board:
        ved = f"{b['wins']}-{b['draws']}-{b['losses']}"
        inf = b["total_inferences"]
        eff = (b["total_coins"] / inf) if inf > 0 else 0.0
        print(f"{b['name']:<10} | {b['elo']:<6.1f} | {ved:<7} | ${b['total_coins']:<8.0f} | {inf:<11} | ${eff:<17.1f}")
    print("=" * 80)

    # Persist results
    out_file = pathlib.Path(__file__).parent / "ablation_tournament_results.json"
    data = {
        "seed": seed,
        "steps": steps,
        "leaderboard": sorted_board,
        "matches": match_logs,
    }
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nResultados persistidos em: {out_file}")


if __name__ == "__main__":
    run_ablation_tournament(seed=42, steps=72)
