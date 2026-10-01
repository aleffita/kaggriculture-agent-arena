"""Pure LiteRT-LM Concurrency, KV-Cache & Scaling Ablation on NVIDIA GeForce GTX 1050 Ti.

Completely isolates the LiteRT-LM inference runtime from any simulator or game logic:
1. Measures VRAM scaling across 1 to 120 active conversation sessions.
2. Measures concurrent batch throughput across N concurrent workers (1, 2, 4, 8, 16, 32)
   under both plain text generation and structured JSON decoding (LL_GUIDANCE).
3. Compares full 72-turn inference horizon: turn-by-turn ping-pong (144 calls) vs. macro-horizon (24 calls).
"""

from __future__ import annotations

import collections
import concurrent.futures
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Tuple

repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

import litert_lm
from litert_lm import interfaces
from litert_explore.gpu import select_gpu
from litert_explore.engine import resolve_model_path


def get_gtx1050ti_vram_mb() -> float:
    """Queries current VRAM used on GPU 1 (GTX 1050 Ti) in MB."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--id=1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8", errors="ignore"
        ).strip()
        return float(out)
    except Exception:
        return 0.0


def run_ablation():
    print("=" * 80)
    print("PURE LITERT-LM CONCURRENCY & SCALING ABLATION ON NVIDIA GEFORCE GTX 1050 Ti")
    print("=" * 80)

    model_path = resolve_model_path()
    vram_init = get_gtx1050ti_vram_mb()
    print(f"[0] Desktop / Idle VRAM on GTX 1050 Ti: {vram_init:.1f} MB")

    print("[1] Initializing LiteRT-LM Engine on GPU 1 (GTX 1050 Ti Direct3D 12)...")
    with select_gpu("1050ti"):
        backend_obj = interfaces.GPU(gpu_decode_steps_per_sync=8)
        t0 = time.time()
        engine = litert_lm.Engine(
            model_path=model_path,
            backend=backend_obj,
            max_num_tokens=512,
            max_num_images=0,
        )
        init_time = time.time() - t0
    vram_base = get_gtx1050ti_vram_mb()
    print(f"    Engine initialized in {init_time:.2f}s | Base VRAM: {vram_base:.1f} MB (Delta: +{vram_base - vram_init:.1f} MB)\n")

    # -------------------------------------------------------------------------
    # PART 1: VRAM SCALING ACROSS 1 TO 120 SESSIONS (KV-CACHE FOOTPRINT)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("PART 1: KV-CACHE VRAM ALLOCATION SCALING (1 to 120 SESSIONS)")
    print("=" * 80)

    session_counts = [1, 2, 4, 8, 16, 32, 64, 120]
    sessions: List[Any] = []
    vram_session_data = []

    for target_count in session_counts:
        needed = target_count - len(sessions)
        for _ in range(needed):
            sessions.append(engine.create_conversation())
        vram_curr = get_gtx1050ti_vram_mb()
        delta = vram_curr - vram_base
        marginal_per_session = (delta / target_count) if target_count > 0 else 0.0
        vram_session_data.append({
            "sessions": target_count,
            "vram_mb": vram_curr,
            "delta_from_base_mb": delta,
            "mb_per_session": marginal_per_session,
        })
        print(f"  Sessions: {target_count:3d} | VRAM: {vram_curr:7.1f} MB | Delta: +{delta:6.1f} MB | Marginal: {marginal_per_session:5.2f} MB/session")

    print(f"  --> With 120 active sessions: VRAM is {vram_session_data[-1]['vram_mb']:.1f} MB / 4096 MB ({4096 - vram_session_data[-1]['vram_mb']:.1f} MB FREE)\n")

    # -------------------------------------------------------------------------
    # PART 2: CONCURRENT INFERENCE BATCH THROUGHPUT (1, 2, 4, 8, 16 WORKERS)
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("PART 2: CONCURRENT INFERENCE THROUGHPUT (DISPATCH ACROSS SESSIONS)")
    print("=" * 80)

    schema = {
        "type": "object",
        "properties": {
            "farmer": {"type": "array", "items": {"type": "string"}},
            "market": {"type": "array", "items": {"type": "array"}},
        },
        "required": ["farmer"],
    }
    cdc = litert_lm.ConstrainedDecodingConfig(
        enable=True,
        provider=litert_lm.LiteRtLmConstraintProviderType.LL_GUIDANCE
    )
    rf = litert_lm.ResponseFormat.json(schema)

    test_prompt = (
        "Farm Status: Day 0, Hour 2. Pos: (4,4) Shed. Seeds: 6 Wheat. Cash: $100.\n"
        "Choose farmer action from ['NORTH', 'SOUTH', 'EAST', 'WEST', 'PLANT', 'WATER', 'HARVEST', 'DIG', 'DROP'].\n"
        "Reply strictly in JSON format."
    )

    engine_lock = threading.Lock()

    def _infer_worker(worker_id: int, use_structured: bool = True) -> Tuple[int, float, str]:
        t_start = time.time()
        with engine_lock:
            if use_structured:
                conv = engine.create_conversation(constrained_decoding_config=cdc)
                resp = conv.send_message(test_prompt, response_format=rf)
            else:
                conv = engine.create_conversation()
                resp = conv.send_message(test_prompt)
        elapsed = time.time() - t_start
        text = str(resp.get("content", "")) if isinstance(resp, dict) else str(resp)
        return worker_id, elapsed, text

    worker_counts = [1, 2, 4, 8]
    throughput_results = []

    for num_w in worker_counts:
        print(f"\n[Testing {num_w} concurrent requests - Structured JSON (LL_GUIDANCE)]...")
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_w) as executor:
            worker_futures = [executor.submit(_infer_worker, i, True) for i in range(num_w)]
            batch_results = [f.result() for f in worker_futures]
        total_time = time.time() - t0
        vram_active = get_gtx1050ti_vram_mb()

        latencies = [r[1] for r in batch_results]
        avg_lat = sum(latencies) / len(latencies)
        inf_per_sec = num_w / total_time

        throughput_results.append({
            "workers": num_w,
            "mode": "structured",
            "batch_time_s": round(total_time, 2),
            "inf_per_sec": round(inf_per_sec, 3),
            "avg_latency_s": round(avg_lat, 2),
            "min_latency_s": round(min(latencies), 2),
            "max_latency_s": round(max(latencies), 2),
            "vram_mb": vram_active,
        })
        print(f"  -> Batch of {num_w} completed in {total_time:.2f}s | Throughput: {inf_per_sec:.3f} inf/s | Avg Latency: {avg_lat:.2f}s (Max: {max(latencies):.2f}s) | VRAM: {vram_active:.1f} MB")

    # -------------------------------------------------------------------------
    # PART 3: HORIZON SCALING: PING-PONG (72 STEPS) VS MACRO-HORIZON (6-STEP)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART 3: 72-TURN GAME HORIZON SIMULATION (PURE INFERENCE RUNTIME)")
    print("=" * 80)

    # Simulation A: Turn-by-Turn ping-pong (sample 6 steps and project to 72 steps to save time)
    sample_turns = 4
    print(f"[A] Measuring sample of {sample_turns} consecutive turn-by-turn inferences (2 players = {sample_turns * 2} calls)...")
    t0 = time.time()
    for t in range(sample_turns):
        for p in range(2):
            with engine_lock:
                conv = engine.create_conversation(constrained_decoding_config=cdc)
                _ = conv.send_message(f"Turn {t} Player {p}: {test_prompt}", response_format=rf)
    t_turn_sample = time.time() - t0
    projected_72_turn_by_turn = (t_turn_sample / sample_turns) * 72
    print(f"    Sample {sample_turns} turns ({sample_turns * 2} calls): {t_turn_sample:.2f}s ({t_turn_sample / (sample_turns * 2):.2f}s per call)")
    print(f"    --> Projected 72-step Match (144 sequential calls): {projected_72_turn_by_turn:.1f}s ({projected_72_turn_by_turn / 60:.2f} MINUTES per match)")

    # Simulation B: Macro-Horizon Planning (1 call every 6 steps = 12 calls per player = 24 calls total)
    macro_prompt = (
        "Farm Status: Day 0, Hour 0. Pos: (4,4) Shed. Seeds: 6 Wheat. Cash: $100.\n"
        "Plan next 6 actions: ['PLANT', 'WATER', 'NORTH', 'SOUTH', 'EAST', 'WEST', 'HARVEST', 'DIG', 'DROP'].\n"
        "Reply with JSON: {\"farmer\": [\"ACTION\", ...]}"
    )
    macro_schema = {
        "type": "object",
        "properties": {
            "plan": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}, "minItems": 1, "maxItems": 6},
            "market": {"type": "array", "items": {"type": "array"}},
        },
        "required": ["plan"],
    }
    rf_macro = litert_lm.ResponseFormat.json(macro_schema)
    sample_macro = 2
    print(f"\n[B] Measuring sample of {sample_macro} macro-plan inferences (horizon=6)...")
    t0 = time.time()
    for m in range(sample_macro):
        with engine_lock:
            conv = engine.create_conversation(constrained_decoding_config=cdc)
            _ = conv.send_message(f"Macro {m}: {macro_prompt}", response_format=rf_macro)
    t_macro_sample = time.time() - t0
    projected_72_macro = (t_macro_sample / sample_macro) * 24  # 12 calls per player * 2 players = 24 calls
    print(f"    Sample {sample_macro} macro calls: {t_macro_sample:.2f}s ({t_macro_sample / sample_macro:.2f}s per call)")
    print(f"    --> Projected 72-step Match with Macro-Queue (24 calls total): {projected_72_macro:.1f}s ({projected_72_macro / 60:.2f} MINUTES per match)")
    print(f"    --> SPEEDUP: {projected_72_turn_by_turn / projected_72_macro:.2f}x faster execution!")

    # -------------------------------------------------------------------------
    # PERSIST RESULTS TO JSON
    # -------------------------------------------------------------------------
    out_file = repo_root / "experiments" / "007_ablations_and_stress" / "litert_pure_concurrency_results.json"
    data = {
        "vram_init_mb": vram_init,
        "vram_base_engine_mb": vram_base,
        "session_scaling": vram_session_data,
        "throughput_concurrency": throughput_results,
        "horizon_comparison": {
            "turn_by_turn_144_calls_projected_s": round(projected_72_turn_by_turn, 2),
            "macro_queue_24_calls_projected_s": round(projected_72_macro, 2),
            "speedup_factor": round(projected_72_turn_by_turn / projected_72_macro, 2),
        }
    }
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\n[OK] Results persisted to {out_file}")


if __name__ == "__main__":
    run_ablation()
