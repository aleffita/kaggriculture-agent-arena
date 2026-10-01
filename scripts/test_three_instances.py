"""Tests loading 3 copies of LiteRT-LM simultaneously on GTX 1050 Ti."""

import time
import subprocess
import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.litert_explore.engine import LiteRtModelRunner

def get_1050ti_vram_mb():
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

def main():
    print("=" * 60)
    print("TESTE DE 3 INSTANCIAS SIMULTANEAS NA GTX 1050 Ti")
    print("=" * 60)

    v0 = get_1050ti_vram_mb()
    print(f"VRAM Inicial (Baseline): {v0:.1f} MB / 4096 MB")

    print("\n[1/3] Carregando Instancia 1...")
    t0 = time.perf_counter()
    r1 = LiteRtModelRunner(backend="gpu", gpu_target="1050ti", max_num_tokens=512)
    r1.generate("ok")
    t1 = time.perf_counter()
    v1 = get_1050ti_vram_mb()
    print(f"  -> Instancia 1 carregada em {t1-t0:.2f}s | VRAM: {v1:.1f} MB (Delta: +{v1-v0:.1f} MB)")

    print("\n[2/3] Carregando Instancia 2...")
    t2 = time.perf_counter()
    r2 = LiteRtModelRunner(backend="gpu", gpu_target="1050ti", max_num_tokens=512)
    r2.generate("ok")
    t3 = time.perf_counter()
    v2 = get_1050ti_vram_mb()
    print(f"  -> Instancia 2 carregada em {t3-t2:.2f}s | VRAM: {v2:.1f} MB (Delta: +{v2-v1:.1f} MB)")

    print("\n[3/3] Carregando Instancia 3...")
    t4 = time.perf_counter()
    r3 = LiteRtModelRunner(backend="gpu", gpu_target="1050ti", max_num_tokens=512)
    r3.generate("ok")
    t5 = time.perf_counter()
    v3 = get_1050ti_vram_mb()
    print(f"  -> Instancia 3 carregada em {t5-t4:.2f}s | VRAM: {v3:.1f} MB (Delta: +{v3-v2:.1f} MB)")

    print("\n[Verificacao de Inferencias]")
    latencies = []
    for i, r in enumerate([r1, r2, r3], 1):
        t_start = time.perf_counter()
        out = r.generate("Diga 'ativo'")
        t_end = time.perf_counter()
        lat = t_end - t_start
        latencies.append(lat)
        print(f"  -> Instancia {i}: {lat:.3f}s | Saida: {out.strip()[:20]}")

    v_final = get_1050ti_vram_mb()
    print("\n" + "=" * 60)
    print(f"VRAM Final Ocupada: {v_final:.1f} MB / 4096 MB ({v_final/4096*100:.1f}%)")
    print(f"VRAM Livre        : {4096 - v_final:.1f} MB")
    print(f"Latencia media    : {sum(latencies)/len(latencies):.3f}s por inferencia")
    print("=" * 60)

if __name__ == "__main__":
    main()
