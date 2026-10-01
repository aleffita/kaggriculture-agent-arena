"""Ablation benchmark for LiteRT-LM GPU optimizations on GTX 1050 Ti:
1. gpu_decode_steps_per_sync (1 vs 8 vs 16 vs 32)
2. activation_data_type (default vs FLOAT32 vs INT8)
3. Persistent KV-cache conversation vs recreation
4. Persistent 2-instance duel execution
"""

import time
import sys
import pathlib

repo_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import litert_lm
from litert_lm import interfaces
from src.litert_explore.engine import resolve_model_path
from src.litert_explore.gpu import select_gpu

def test_sync_steps():
    print("\n" + "=" * 60)
    print("ABLACAO 1: gpu_decode_steps_per_sync")
    print("=" * 60)
    model_path = resolve_model_path()
    
    for sync in [1, 8, 16, 32]:
        try:
            with select_gpu("1050ti"):
                backend = interfaces.GPU(gpu_decode_steps_per_sync=sync)
                engine = litert_lm.Engine(
                    model_path=model_path,
                    backend=backend,
                    max_num_tokens=512,
                    max_num_images=0,
                )
                conv = engine.create_conversation()
                # warmup
                conv.send_message("oi")
                # measure
                t0 = time.perf_counter()
                resp = conv.send_message("Gere uma lista de 5 culturas agricolas com precos:")
                t1 = time.perf_counter()
                text = resp.candidates[0].message.content[0].text if resp.candidates else ""
                print(f"  sync={sync:<2} | Latencia: {t1-t0:.3f}s | Tokens/Saida: {len(text.split())} palavras")
                engine.close()
        except Exception as exc:
            print(f"  sync={sync:<2} | Erro: {exc}")

def test_kv_cache_persistence():
    print("\n" + "=" * 60)
    print("ABLACAO 2: Re-criacao de Conversation vs KV-Cache Persistente")
    print("=" * 60)
    model_path = resolve_model_path()
    
    with select_gpu("1050ti"):
        backend = interfaces.GPU(gpu_decode_steps_per_sync=16)
        engine = litert_lm.Engine(
            model_path=model_path,
            backend=backend,
            max_num_tokens=512,
            max_num_images=0,
        )

        # Caso A: Re-criando conversation a cada step (como estava)
        t_recreate_start = time.perf_counter()
        for i in range(5):
            c = engine.create_conversation()
            c.send_message(f"Turno {i}: decida acao")
        t_recreate_total = time.perf_counter() - t_recreate_start

        # Caso B: Conversation persistente (KV cache mantido na GPU)
        t_persist_start = time.perf_counter()
        persisted_conv = engine.create_conversation()
        for i in range(5):
            persisted_conv.send_message(f"Turno {i}: decida acao")
        t_persist_total = time.perf_counter() - t_persist_start

        print(f"  5 turnos com Re-criacao (antigo)    : {t_recreate_total:.3f}s ({t_recreate_total/5:.3f}s/turno)")
        print(f"  5 turnos com KV-Cache Persistente   : {t_persist_total:.3f}s ({t_persist_total/5:.3f}s/turno)")
        print(f"  -> Ganho de velocidade do KV-Cache  : {t_recreate_total / max(t_persist_total, 0.001):.2f}x mais rapido!")
        engine.close()

def main():
    test_sync_steps()
    test_kv_cache_persistence()

if __name__ == "__main__":
    main()
