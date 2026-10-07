"""
Agnostic Benchmark & Evaluation Runner.
Executes standardized performance benchmarks (llama-bench style) and capability evals (HumanEval, Perplexity).
Defaults to --mode smoke for fast iterative development.
Saves a single unified JSON report per execution suite (run_<timestamp>_suite_<mode>.json).
Strict numeric formatting: NO thousand separators, decimal dot notation only.
"""
import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Type

# Garantir UTF-8 no stdout/stderr no Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BENCHMARKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARKS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.plugins.base import (
    BaseBenchmarkPlugin,
    BenchmarkMeasurement,
    BenchmarkSuiteResult,
)
from benchmarks.plugins.throughput_bench import ThroughputBenchmarkPlugin
from benchmarks.plugins.humaneval_bench import HumanEvalBenchmarkPlugin
from benchmarks.plugins.perplexity_bench import PerplexityBenchmarkPlugin

REGISTERED_BENCHMARKS: Dict[str, Type[BaseBenchmarkPlugin]] = {
    "throughput": ThroughputBenchmarkPlugin,
    "humaneval": HumanEvalBenchmarkPlugin,
    "perplexity": PerplexityBenchmarkPlugin,
}

REPORTS_DIR = BENCHMARKS_DIR / "reports"

def ensure_environment():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def save_unified_suite_run(results: List[BenchmarkSuiteResult], mode: str) -> Path:
    ensure_environment()
    now_iso = datetime.utcnow().isoformat() + "Z"
    safe_ts = now_iso.replace(":", "-").replace(".", "_")
    report_file = REPORTS_DIR / f"run_{safe_ts}_suite_{mode}.json"

    data = {
        "timestamp": now_iso,
        "mode": mode,
        "environment_metadata": {
            "os": "Windows NT",
            "gpus": ["NVIDIA GeForce RTX 2060 (6GB)", "NVIDIA GeForce GTX 1050 Ti (4GB)"],
            "runtime_primary": "Unified Heterogeneous CED Engine (CUDA/D3D12/Direct NVMe)",
            "models_triad": ["gpt-oss-20b", "bonsai-27b", "gemma-4-E2B-it"]
        },
        "suites": {},
        "all_measurements": []
    }

    for suite in results:
        data["suites"][suite.benchmark_name] = {
            "benchmark_name": suite.benchmark_name,
            "mode": suite.mode,
            "timestamp": suite.timestamp,
            "environment_metadata": suite.environment_metadata,
            "measurements": [m.__dict__ for m in suite.measurements],
        }
        for m in suite.measurements:
            data["all_measurements"].append(m.__dict__)

    with open(report_file, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return report_file

def print_suite_table(suite: BenchmarkSuiteResult):
    print("\n" + "=" * 116)
    print(f" 📊 RESULTADOS DO BENCHMARK: [{suite.benchmark_name.upper()}] (Modo: {suite.mode.upper()})")
    print("=" * 116)

    if suite.benchmark_name == "throughput":
        print(f" {'Backend':<24} | {'Modelo':<15} | {'P':<5} | {'N':<5} | {'Prefill (t/s)':<14} | {'TTFT (ms)':<10} | {'Decode (t/s)':<13} | {'ms/tok':<9} | {'Status':<12}")
        print("-" * 116)
        for m in suite.measurements:
            p_tok = f"{m.prefill_tok_s:.2f}"
            ttft = f"{m.prefill_ttft_ms:.2f}"
            d_tok = f"{m.decode_tok_s:.2f}"
            mpt = f"{m.decode_ms_per_tok:.4f}"
            print(f" {m.backend:<24} | {m.model:<15} | {m.prompt_tokens:<5} | {m.gen_tokens:<5} | {p_tok:>14} | {ttft:>10} | {d_tok:>13} | {mpt:>9} | {m.status:<12}")
    else:
        print(f" {'Benchmark':<15} | {'Backend':<24} | {'Metrica':<15} | {'Valor':<12} | {'Status':<10}")
        print("-" * 116)
        for m in suite.measurements:
            val_str = f"{m.metric_value:.4f}" if isinstance(m.metric_value, float) else str(m.metric_value)
            print(f" {m.benchmark:<15} | {m.backend:<24} | {m.metric_name:<15} | {val_str:<12} | {m.status:<10}")

    print("=" * 116)

def main():
    parser = argparse.ArgumentParser(description="Agnostic LLM Benchmark & Evaluation Suite")
    parser.add_argument("--benchmark", "-b", "--plugin", "-p", type=str, help="Nome do benchmark a executar")
    parser.add_argument("--all", "-a", action="store_true", help="Executar todos os benchmarks registrados")
    parser.add_argument("--list", "-l", action="store_true", help="Listar todos os benchmarks disponíveis")
    parser.add_argument("--mode", "-m", choices=["smoke", "full"], default="smoke",
                        help="Modo de execução: smoke (padrão de desenvolvimento rápido) ou full")

    args = parser.parse_args()

    if args.list:
        print("\n📋 Benchmarks Disponíveis na Suíte:")
        for name, cls in REGISTERED_BENCHMARKS.items():
            print(f"  - {name:<15}: {cls.description}")
        print()
        sys.exit(0)

    benchmarks_to_run: List[Type[BaseBenchmarkPlugin]] = []

    if args.all:
        benchmarks_to_run = list(REGISTERED_BENCHMARKS.values())
    elif args.benchmark:
        if args.benchmark not in REGISTERED_BENCHMARKS:
            print(f"[-] Erro: Benchmark '{args.benchmark}' não encontrado.")
            print(f"    Disponíveis: {list(REGISTERED_BENCHMARKS.keys())}")
            sys.exit(1)
        benchmarks_to_run = [REGISTERED_BENCHMARKS[args.benchmark]]
    else:
        parser.print_help()
        sys.exit(0)

    executed_suites: List[BenchmarkSuiteResult] = []
    any_failed = False

    for bench_cls in benchmarks_to_run:
        bench = bench_cls()
        print(f"\n[+] Executando benchmark: {bench.name} (Modo: {args.mode.upper()})...")
        try:
            suite_res = bench.run(mode=args.mode)
            executed_suites.append(suite_res)
            print_suite_table(suite_res)
            for m in suite_res.measurements:
                if m.status not in ("SUCCESS", "SKIPPED_OOM"):
                    any_failed = True
        except Exception as e:
            print(f"[-] Exceção durante execução de {bench.name}: {e}")
            any_failed = True

    if executed_suites:
        report_file = save_unified_suite_run(executed_suites, mode=args.mode)
        print(f"\n📁 [Relatório JSON Unificado Salvo]: {report_file.name}")
        print(f"   Caminho: {report_file}\n")

    if any_failed:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
