"""Rigorous probe: measures KV-cache VRAM per conversation/session and concurrency scaling on GTX 1050 Ti."""

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

def get_1050ti_vram_mb() -> float:
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
    print("=" * 70)
    print("PROBE RIGOROSA: PESO BASE DO MODELO VS KV-CACHE POR CONVERSA / PARTIDA")
    print("=" * 70)

    v0 = get_1050ti_vram_mb()
    print(f"[0] Baseline VRAM GTX 1050 Ti (Ocioso): {v0:.1f} MB")

    model_path = resolve_model_path()
    
    with select_gpu("1050ti"):
        print("\n[1] Carregando 1 Instancia de Engine (max_num_tokens=512)...")
        backend = interfaces.GPU(gpu_decode_steps_per_sync=16)
        t0 = time.perf_counter()
        engine = litert_lm.Engine(
            model_path=model_path,
            backend=backend,
            max_num_tokens=512,
            max_num_images=0,
        )
        t1 = time.perf_counter()
        v_engine = get_1050ti_vram_mb()
        delta_engine = v_engine - v0
        print(f"  -> Engine carregada em {t1-t0:.2f}s")
        print(f"  -> VRAM com Engine: {v_engine:.1f} MB (Peso Base dos Pesos: {delta_engine:.1f} MB)")

        print("\n[2] Criando Conversacoes / Sessoes Ativas e Medindo VRAM:")
        convs = []
        checkpoints = [1, 2, 4, 8, 16]
        
        last_vram = v_engine
        for count in checkpoints:
            while len(convs) < count:
                c = engine.create_conversation()
                # Run an initial prompt to allocate internal KV cache buffers
                c.send_message("ok")
                convs.append(c)
            current_v = get_1050ti_vram_mb()
            delta_step = current_v - last_vram
            delta_total = current_v - v_engine
            per_conv = delta_total / count if count > 0 else 0
            print(f"  -> {count:<2} conversacoes ativas: VRAM = {current_v:.1f} MB | Delta: +{delta_total:.1f} MB ({per_conv:.2f} MB/conversacao)")
            last_vram = current_v

        v_final_convs = get_1050ti_vram_mb()
        print(f"\n[3] Resumo do Custo de Memoria:")
        print(f"  - Peso Fixo do Modelo (Pesos/Mmap/Shaders) : {delta_engine:.1f} MB")
        print(f"  - Custo por Sessao de Jogo / KV-Cache     : {(v_final_convs - v_engine) / len(convs):.2f} MB por conversa")
        print(f"  - VRAM Livre Restante na 1050 Ti          : {4096 - v_final_convs:.1f} MB")
        
        # Test concurrency latency across 1, 2, 4, 8 conversations
        print("\n[4] Teste de Throughput Concorrente (Interleaved Generation):")
        for n in [1, 2, 4, 8]:
            t_start = time.perf_counter()
            for i in range(n):
                convs[i].send_message("JSON: {'farmer': ['NORTH']}")
            t_elapsed = time.perf_counter() - t_start
            print(f"  -> {n} geracoes consecutivas: {t_elapsed:.3f}s total ({t_elapsed/n:.3f}s/inferencia | {n/t_elapsed:.2f} inferencias/s)")

        engine.close()

    print("=" * 70)

if __name__ == "__main__":
    main()
