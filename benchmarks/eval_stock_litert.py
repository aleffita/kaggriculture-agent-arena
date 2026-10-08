"""
Stock Original LiteRT-LM Direct Evaluator for Gemma-4-E2B-it.
Measures unmodified Google LiteRT-LM GPU performance (Direct3D 12 via Google Dawn)
running directly against C++ engine without HTTP server or virtual expert wrappers.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from litert_explore.engine import LiteRtModelRunner

BENCHMARK_PROMPTS = [
    {
        "id": "code_humaneval_0",
        "category": "code",
        "prompt": 'def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3)\n    True\n    >>> has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95)\n    False\n    """\n',
    },
    {
        "id": "code_humaneval_1",
        "category": "code",
        "prompt": 'def separate_paren_groups(paren_string: str) -> list[str]:\n    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to\n    separate those group into separate strings and return the list of those.\n    Separate groups are balanced (each open brace is properly closed) and not nested within each other\n    Ignore any spaces in the input string.\n    >>> separate_paren_groups(\'( ) (( )) (( )( ))\')\n    [\'()\', \'(())\', \'(()())\']\n    """\n',
    },
    {
        "id": "math_gsm8k_0",
        "category": "math",
        "prompt": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Let's solve this step by step.\n",
    },
    {
        "id": "math_gsm8k_1",
        "category": "math",
        "prompt": "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer: Let's calculate:\n",
    },
    {
        "id": "reasoning_general",
        "category": "reasoning",
        "prompt": "Explain the architectural difference between hardware-accelerated BVH spatial pruning and dense tensor evaluation in Mixture of Experts (MoE) LLMs.\n",
    }
]


def run_stock_evaluation(output_dir: Path) -> Dict[str, Any]:
    print("[*] Inicializando motor Stock Original LiteRT-LM (Direct3D 12 na RTX 2060)...")
    t_init_0 = time.perf_counter()
    runner = LiteRtModelRunner(backend="gpu", gpu_target="2060")
    t_init = time.perf_counter() - t_init_0
    print(f"✔ Motor stock carregado em {t_init:.2f}s")

    samples: List[Dict[str, Any]] = []
    total_tokens = 0
    total_time = 0.0

    print(f"\n[*] Executando {len(BENCHMARK_PROMPTS)} amostras na GPU (Stock Original)...")

    for i, item in enumerate(BENCHMARK_PROMPTS):
        p_id = item["id"]
        category = item["category"]
        prompt = item["prompt"]

        print(f"  [{i+1}/{len(BENCHMARK_PROMPTS)}] Prompt: {p_id} ({category})...", end="", flush=True)
        t0 = time.perf_counter()
        output_text = runner.generate(prompt)
        dt = time.perf_counter() - t0

        approx_tokens = len(output_text.split()) * 1.3
        tok_s = (approx_tokens / dt) if dt > 0 else 0.0
        total_tokens += approx_tokens
        total_time += dt

        print(f" Concluído em {dt:.2f}s (~{approx_tokens:.0f} tokens, {tok_s:.1f} tok/s)")

        samples.append({
            "id": p_id,
            "category": category,
            "prompt_length_chars": len(prompt),
            "output_length_chars": len(output_text),
            "estimated_tokens": round(approx_tokens, 1),
            "latency_seconds": round(dt, 3),
            "throughput_tokens_per_sec": round(tok_s, 2),
            "response_preview": output_text[:200]
        })

    avg_tok_s = (total_tokens / total_time) if total_time > 0 else 0.0
    report = {
        "benchmark": "stock_original_litert_eval",
        "model": "gemma-4-E2B-it",
        "backend": "Google LiteRT-LM Direct3D 12 (NVIDIA GeForce RTX 2060)",
        "hardware": "RTX 2060 (6 GB VRAM) + GTX 1050 Ti (4 GB VRAM)",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "init_time_seconds": round(t_init, 2),
        "total_eval_time_seconds": round(total_time, 2),
        "total_estimated_tokens": round(total_tokens, 1),
        "overall_average_throughput_tok_s": round(avg_tok_s, 2),
        "average_sample_latency_s": round(total_time / len(samples), 3),
        "samples": samples
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"stock_litert_eval_{report['timestamp']}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✔ Relatório Stock LiteRT-LM salvo em: {report_file}")
    print(f"  Vazão média: {avg_tok_s:.2f} tokens/s | Latência média: {total_time/len(samples):.2f}s")
    return report


def main():
    repo_root = Path(__file__).resolve().parent.parent
    output_dir = repo_root / "benchmarks" / "reports"
    run_stock_evaluation(output_dir)


if __name__ == "__main__":
    main()
