"""Empirical test: loads 2 instances of LiteRT-LM simultaneously on GTX 1050 Ti."""

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
            return float(lines[1])  # GPU 1
    except Exception:
        pass
    return 0.0

def main():
    print("=" * 60)
    print("CURVA DE MEMORIA: TESTE DE DUAS INSTANCIAS NA GTX 1050 Ti")
    print("=" * 60)

    vram_0 = get_1050ti_vram_mb()
    print(f"[Baseline] VRAM GTX 1050 Ti inicial: {vram_0:.1f} MB")

    print("\n[Passo 1] Carregando Instancia 1 no Alvo 1050 Ti...")
    t0 = time.perf_counter()
    r1 = LiteRtModelRunner(backend="gpu", gpu_target="1050ti", max_num_tokens=512)
    out1 = r1.generate("Instancia 1 teste: diga ok")
    t1 = time.perf_counter()
    vram_1 = get_1050ti_vram_mb()
    print(f"  -> Instancia 1 pronta em {t1 - t0:.2f}s | Resposta: {out1[:30].strip()}")
    print(f"  -> VRAM apos Instancia 1: {vram_1:.1f} MB (Delta: +{vram_1 - vram_0:.1f} MB)")

    print("\n[Passo 2] Carregando Instancia 2 no Alvo 1050 Ti...")
    t2 = time.perf_counter()
    r2 = LiteRtModelRunner(backend="gpu", gpu_target="1050ti", max_num_tokens=512)
    out2 = r2.generate("Instancia 2 teste: diga ok")
    t3 = time.perf_counter()
    vram_2 = get_1050ti_vram_mb()
    print(f"  -> Instancia 2 pronta em {t3 - t2:.2f}s | Resposta: {out2[:30].strip()}")
    print(f"  -> VRAM apos Instancia 2: {vram_2:.1f} MB (Delta Instancia 2: +{vram_2 - vram_1:.1f} MB | Total: +{vram_2 - vram_0:.1f} MB)")

    print("\n[Passo 3] Testando inferencias alternadas:")
    t_gen_1 = time.perf_counter()
    r1.generate("Gere 1 palavra:")
    t_gen_1_end = time.perf_counter()

    t_gen_2 = time.perf_counter()
    r2.generate("Gere 1 palavra:")
    t_gen_2_end = time.perf_counter()

    print(f"  -> Tempo de inferencia Instancia 1: {t_gen_1_end - t_gen_1:.3f} s")
    print(f"  -> Tempo de inferencia Instancia 2: {t_gen_2_end - t_gen_2:.3f} s")
    print(f"  -> VRAM final: {get_1050ti_vram_mb():.1f} MB / 4096 MB")
    print("=" * 60)

if __name__ == "__main__":
    main()
