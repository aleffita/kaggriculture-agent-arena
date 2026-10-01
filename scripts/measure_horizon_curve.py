"""Empirical measurement of the step horizon scaling curve on GTX 1050 Ti."""

import time
import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from kaggle_environments import make
from kaggriculture.agents.personality_agent import make_personality_agent

def run_horizon_test(horizon_steps: int):
    agent_a = make_personality_agent("sprint_rusher")
    agent_b = make_personality_agent("market_arbitrageur")

    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": horizon_steps,
            "actTimeout": 60,
            "runTimeout": 3600,
        },
        debug=True,
    )

    t0 = time.perf_counter()
    env.run([agent_a, agent_b])
    t1 = time.perf_counter()

    elapsed = t1 - t0
    final = env.steps[-1]
    p0_bank = final[0]["observation"]["farms"][0]["money"]
    p1_bank = final[0]["observation"]["farms"][1]["money"]
    p0_reward = final[0].reward
    p1_reward = final[1].reward

    return {
        "steps": horizon_steps,
        "days": horizon_steps / 24.0,
        "elapsed_s": elapsed,
        "sec_per_turn": elapsed / horizon_steps,
        "p0_bank": p0_bank,
        "p1_bank": p1_bank,
        "p0_reward": p0_reward,
        "p1_reward": p1_reward,
    }

def main():
    print("=" * 70)
    print("CURVA DE ESCALABILIDADE DE HORIZONTE NA GTX 1050 Ti")
    print("=" * 70)

    horizons = [12, 24, 48, 72]
    results = []

    for h in horizons:
        print(f"\n[*] Testando horizonte: {h} steps ({h / 24.0:.1f} dias in-game)...")
        res = run_horizon_test(h)
        results.append(res)
        print(f"    -> Tempo: {res['elapsed_s']:.2f}s ({res['elapsed_s']/60:.2f} min) | Latencia: {res['sec_per_turn']:.3f}s/turno")
        print(f"    -> Balancos: P0 (sprint)=${res['p0_bank']:.0f}, P1 (arbitrage)=${res['p1_bank']:.0f}")

    print("\n" + "=" * 70)
    print("TABELA CONSOLIDADA DA CURVA DE HORIZONTE:")
    print("=" * 70)
    print(f"{'Steps':<8} {'Dias':<6} {'Tempo Total':<14} {'Por Turno':<12} {'P0 Bank':<12} {'P1 Bank':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['steps']:<8} {r['days']:<6.1f} {r['elapsed_s']:<6.2f}s ({r['elapsed_s']/60:4.2f}m)  {r['sec_per_turn']:<6.3f}s/t    ${r['p0_bank']:<11.0f} ${r['p1_bank']:<11.0f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
