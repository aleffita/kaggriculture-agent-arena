"""Stress test: tests allocating up to 120 concurrent sessions on GTX 1050 Ti."""

import time
import subprocess
import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import litert_lm
from litert_lm import interfaces
from src.litert_explore.engine import resolve_model_path
from src.litert_explore.gpu import select_gpu

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

def main():
    print("=" * 65)
    print("STRESS TEST: ALOCACAO DE ATE 120 SESSOES NA GTX 1050 Ti")
    print("=" * 65)

    v0 = get_vram_mb()
    print(f"VRAM Inicial: {v0:.1f} MB")

    model_path = resolve_model_path()

    with select_gpu("1050ti"):
        backend = interfaces.GPU(gpu_decode_steps_per_sync=16)
        engine = litert_lm.Engine(
            model_path=model_path,
            backend=backend,
            max_num_tokens=512,
            max_num_images=0,
        )
        v_engine = get_vram_mb()
        print(f"VRAM com Engine: {v_engine:.1f} MB (Base: {v_engine - v0:.1f} MB)")

        convs = []
        target_counts = [10, 20, 40, 60, 80, 100, 120]
        
        for target in target_counts:
            print(f"\nTentando atingir {target} sessoes...")
            t0 = time.perf_counter()
            failed = False
            while len(convs) < target:
                try:
                    c = engine.create_conversation()
                    # Trigger allocation with a short send
                    c.send_message("ok")
                    convs.append(c)
                except Exception as exc:
                    print(f"  [FALHA / LIMITE ATINGIDO] Falha na sessao #{len(convs)+1}: {exc}")
                    failed = True
                    break
            t1 = time.perf_counter()
            current_v = get_vram_mb()
            print(f"  -> Sessoes ativas: {len(convs)} | VRAM: {current_v:.1f} MB / 4096 MB | Delta: +{current_v - v_engine:.1f} MB | Tempo: {t1-t0:.2f}s")
            if failed:
                break

        print("\n" + "=" * 65)
        print(f"TOTAL MAXIMO DE SESSOES CRIADAS: {len(convs)}")
        print(f"VRAM Final: {get_vram_mb():.1f} MB / 4096 MB")
        print("=" * 65)

        # Cleanly close all conversations before closing engine
        print("Fechando conversas...")
        for c in convs:
            try:
                c.close()
            except Exception:
                pass
        engine.close()

if __name__ == "__main__":
    main()
