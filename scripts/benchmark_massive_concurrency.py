"""Massive Concurrency Benchmark: Tests 10, 20, 40, and 60 simultaneous matches (up to 120 sessions) on GTX 1050 Ti."""

import time
import subprocess
import concurrent.futures
import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from kaggle_environments import make
from kaggriculture.agents.personality_agent import make_personality_agent

def get_vram_mb():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        if len(lines) >= 2:
            return float(lines[1])
    except Exception:
        pass
    return 0.0

def run_match_task(match_id: int, steps: int = 10):
    # Alternate pairings across personalities
    personalities = [
        ("sprint_rusher", "land_baron"),
        ("labor_magnate", "melon_monopolist"),
        ("market_arbitrageur", "cautious_farmer"),
    ]
    p0_name, p1_name = personalities[match_id % len(personalities)]
    a0 = make_personality_agent(p0_name)
    a1 = make_personality_agent(p1_name)

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": steps, "actTimeout": 60, "runTimeout": 3600},
        debug=False,
    )
    t0 = time.perf_counter()
    env.run([a0, a1])
    dt = time.perf_counter() - t0

    final = env.steps[-1]
    b0 = final[0]["observation"]["farms"][0]["money"]
    b1 = final[0]["observation"]["farms"][1]["money"]
    return match_id, dt, b0, b1

def run_batch_test(num_matches: int, steps: int = 10):
    num_sessions = num_matches * 2
    print(f"\n---> DISPARANDO BATCH DE {num_matches} PARTIDAS SIMULTANEAS ({num_sessions} SESSOES ATIVAS)...")
    v_start = get_vram_mb()
    t_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_matches) as pool:
        futures = [pool.submit(run_match_task, i, steps) for i in range(num_matches)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    t_total = time.perf_counter() - t_start
    v_peak = get_vram_mb()
    total_inferences = num_matches * steps * 2
    inf_per_sec = total_inferences / max(t_total, 0.001)
    matches_per_min = (num_matches / t_total) * 60

    print(f"  -> Concluido em: {t_total:.2f}s")
    print(f"  -> VRAM Utilizada: {v_peak:.1f} MB / 4096 MB (Delta: +{v_peak - v_start:.1f} MB)")
    print(f"  -> Throughput Global: {inf_per_sec:.1f} inferencias/s ({matches_per_min:.1f} partidas completas/minuto)")
    return {
        "matches": num_matches,
        "sessions": num_sessions,
        "time_s": t_total,
        "vram_mb": v_peak,
        "inf_per_sec": inf_per_sec,
        "matches_per_min": matches_per_min,
    }

def main():
    print("=" * 70)
    print("BENCHMARK DE CONCORRENCIA MASSIVA NA GTX 1050 Ti")
    print("ESCALA: 10, 20, 40 E 60 PARTIDAS SIMULTANEAS (ATE 120 SESSOES)")
    print("=" * 70)

    # Warmup singleton engine
    from kaggriculture.agents.llm_player import get_llm_runner
    runner = get_llm_runner()
    runner.generate("warmup")
    print(f"Engine aquecida na 1050 Ti. VRAM base: {get_vram_mb():.1f} MB\n")

    steps_per_match = 10
    concurrency_levels = [10, 20, 40, 60]
    bench_data = []

    for n_matches in concurrency_levels:
        res = run_batch_test(n_matches, steps=steps_per_match)
        bench_data.append(res)

    print("\n" + "=" * 70)
    print("CURVA CONSOLIDADA DE CONCORRENCIA MASSIVA (GTX 1050 Ti):")
    print("=" * 70)
    print(f"{'Partidas':<10} {'Sessoes':<10} {'Tempo Total':<14} {'VRAM':<12} {'Inferencias/s':<16} {'Partidas/min':<14}")
    print("-" * 70)
    for b in bench_data:
        print(f"{b['matches']:<10} {b['sessions']:<10} {b['time_s']:<6.2f}s        {b['vram_mb']:<6.1f} MB    {b['inf_per_sec']:<6.1f} inf/s     {b['matches_per_min']:<6.1f} part/m")
    print("=" * 70)

if __name__ == "__main__":
    main()
