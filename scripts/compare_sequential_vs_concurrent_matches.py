"""Empirical benchmark: 3 full matches executed Sequentially vs Concurrently on GTX 1050 Ti."""

import time
import concurrent.futures
import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from kaggle_environments import make
from kaggriculture.agents.personality_agent import make_personality_agent

def run_one_match(p0_name: str, p1_name: str, steps: int = 24, match_id: int = 1):
    agent_a = make_personality_agent(p0_name)
    agent_b = make_personality_agent(p1_name)

    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": steps,
            "actTimeout": 60,
            "runTimeout": 3600,
        },
        debug=False,
    )

    t0 = time.perf_counter()
    env.run([agent_a, agent_b])
    t1 = time.perf_counter()

    final = env.steps[-1]
    b0 = final[0]["observation"]["farms"][0]["money"]
    b1 = final[0]["observation"]["farms"][1]["money"]
    return {
        "match_id": match_id,
        "elapsed": t1 - t0,
        "p0": p0_name,
        "p1": p1_name,
        "p0_bank": b0,
        "p1_bank": b1,
        "winner": p0_name if b0 > b1 else (p1_name if b1 > b0 else "Draw"),
    }

def main():
    print("=" * 70)
    print("BENCHMARK: 3 PARTIDAS INTEIRAS (24 STEPS / 1 DIA IN-GAME)")
    print("SEQUENCIAL VS CONCORRENTE NA MESMA GPU (GTX 1050 Ti)")
    print("=" * 70)

    matchups = [
        ("sprint_rusher", "land_baron"),
        ("labor_magnate", "melon_monopolist"),
        ("market_arbitrageur", "cautious_farmer"),
    ]

    # --- TESTE A: SEQUENCIAL (Como a arena rodava) ---
    print("\n[FASE 1] Executando 3 partidas SEQUENCIALMENTE (uma apos a outra)...")
    t_seq_start = time.perf_counter()
    seq_results = []
    for i, (p0, p1) in enumerate(matchups, 1):
        print(f"  -> Rodando Partida {i}: {p0} vs {p1}...")
        r = run_one_match(p0, p1, steps=24, match_id=i)
        seq_results.append(r)
        print(f"     Concluida em {r['elapsed']:.2f}s | Vencedor: {r['winner']} (${r['p0_bank']:.0f} vs ${r['p1_bank']:.0f})")
    t_seq_total = time.perf_counter() - t_seq_start
    print(f"  [RESULTADO SEQUENCIAL] Tempo total para 3 partidas: {t_seq_total:.2f}s ({t_seq_total/60:.2f} min)")

    # --- TESTE B: CONCORRENTE (Em lote / paralelo) ---
    print("\n[FASE 2] Executando 3 partidas CONCORRENTEMENTE (ThreadPoolExecutor max_workers=3)...")
    t_par_start = time.perf_counter()
    par_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(run_one_match, p0, p1, 24, i)
            for i, (p0, p1) in enumerate(matchups, 1)
        ]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            par_results.append(res)
            print(f"     Partida {res['match_id']} ({res['p0']} vs {res['p1']}) concluida em {res['elapsed']:.2f}s")
    t_par_total = time.perf_counter() - t_par_start
    print(f"  [RESULTADO CONCORRENTE] Tempo total para 3 partidas: {t_par_total:.2f}s ({t_par_total/60:.2f} min)")

    print("\n" + "=" * 70)
    print("COMPARATIVO FINAL EMPIRICO:")
    print("=" * 70)
    print(f"Tempo Total 3 Partidas Sequenciais: {t_seq_total:.2f}s ({t_seq_total/60:.2f} min)")
    print(f"Tempo Total 3 Partidas Concorrentes: {t_par_total:.2f}s ({t_par_total/60:.2f} min)")
    speedup = t_seq_total / max(t_par_total, 0.001)
    print(f"Speedup de Throughput             : {speedup:.2f}x ({((1 - t_par_total/t_seq_total)*100):.1f}% de reducao de tempo)")
    print("=" * 70)

if __name__ == "__main__":
    main()
