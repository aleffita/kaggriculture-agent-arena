"""Measures precise execution time of an isolated 72-step match on GTX 1050 Ti."""

import time
import os
import sys
import pathlib

# Ensure repo root is on sys.path
repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from kaggle_environments import make
from kaggriculture.agents.personality_agent import make_personality_agent

def main():
    print("=" * 60)
    print("INICIANDO MEDICAO ISOLADA DE PARTIDA NA GTX 1050 Ti")
    print("=" * 60)

    # Instantiate two distinct personality agents
    print("[1/3] Instanciando agentes...")
    agent_a = make_personality_agent("sprint_rusher")
    agent_b = make_personality_agent("market_arbitrageur")

    print("[2/3] Configurando ambiente (72 steps / 3 dias, actTimeout=60s)...")
    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": 72,
            "actTimeout": 60,
            "runTimeout": 3600,
        },
        debug=True,
    )

    print("[3/3] Executando partida completa (72 steps = 144 inferencias na 1050 Ti)...")
    t_start = time.perf_counter()
    env.run([agent_a, agent_b])
    t_end = time.perf_counter()

    elapsed = t_end - t_start
    total_steps = len(env.steps)
    avg_per_step = elapsed / max(total_steps, 1)

    final_step = env.steps[-1]
    p0_reward = final_step[0].reward
    p1_reward = final_step[1].reward
    p0_status = final_step[0].status
    p1_status = final_step[1].status

    p0_bank = final_step[0]["observation"]["farms"][0]["money"]
    p1_bank = final_step[0]["observation"]["farms"][1]["money"]

    print("\n" + "=" * 60)
    print("RESULTADO DA MEDICAO ISOLADA:")
    print("=" * 60)
    print(f"Tempo total decorrido : {elapsed:.2f} s ({elapsed / 60:.2f} min)")
    print(f"Total de turnos jogados: {total_steps} turnos")
    print(f"Tempo medio por turno : {avg_per_step:.3f} s/turno (2 inferencias por turno)")
    print(f"Tempo medio por inferencia: {avg_per_step / 2:.3f} s/inferencia na GTX 1050 Ti")
    print(f"Status Player 0       : {p0_status} (Reward: {p0_reward}, Bank: ${p0_bank:.0f})")
    print(f"Status Player 1       : {p1_status} (Reward: {p1_reward}, Bank: ${p1_bank:.0f})")
    print("=" * 60)

if __name__ == "__main__":
    main()
