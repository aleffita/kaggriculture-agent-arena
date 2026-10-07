"""
Agnostic Benchmark & Evaluation Runner.
Executes standardized performance benchmarks (llama-bench style) and capability evals (HumanEval, Perplexity).
Defaults to --mode smoke for fast iterative development.
"""
import sys
import os
import argparse
import json
import csv
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

CSV_FILE = BENCHMARKS_DIR / "results.csv"
REPORTS_DIR = BENCHMARKS_DIR / "reports"

CSV_COLUMNS = [
    "timestamp",
    "benchmark",
    "backend",
    "model",
    "mode",
    "prompt_tokens",
    "gen_tokens",
    "batch_size",
    "prefill_tok_s",
    "prefill_ttft_ms",
    "decode_tok_s",
    "decode_ms_per_tok",
    "metric_name",
    "metric_value",
    "error_stddev",
    "status",
]

def ensure_environment():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_FILE.exists():
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

def save_suite_to_csv(suite: BenchmarkSuiteResult):
    ensure_environment()
    rows = suite.to_csv_rows()
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        for r in rows:
            writer.writerow(r)

def save_suite_to_json(suite: BenchmarkSuiteResult):
    ensure_environment()
    safe_ts = suite.timestamp.replace(":", "-").replace(".", "_")
    report_file = REPORTS_DIR / f"run_{safe_ts}_{suite.benchmark_name}_{suite.mode}.json"
    data = {
        "benchmark_name": suite.benchmark_name,
        "mode": suite.mode,
        "timestamp": suite.timestamp,
        "environment_metadata": suite.environment_metadata,
        "measurements": [m.__dict__ for m in suite.measurements],
    }
    with open(report_file, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return report_file

def print_suite_table(suite: BenchmarkSuiteResult, json_path: Path):
    print("\n" + "=" * 94)
    print(f" 📊 RESULTADOS DO BENCHMARK: [{suite.benchmark_name.upper()}] (Modo: {suite.mode.upper()})")
    print("=" * 94)

    if suite.benchmark_name == "throughput":
        print(f" {'Modelo':<10} | {'P':<5} | {'N':<5} | {'Prefill (t/s)':<15} | {'TTFT (ms)':<11} | {'Decode (t/s)':<14} | {'ms/tok':<9} | {'Desvio':<8}")
        print("-" * 94)
        for m in suite.measurements:
            print(f" {m.model:<10} | {m.prompt_tokens:<5} | {m.gen_tokens:<5} | {m.prefill_tok_s:>13,.2f}  | {m.prefill_ttft_ms:>9.2f}  | {m.decode_tok_s:>12,.2f}  | {m.decode_ms_per_tok:>7.4f}  | ±{m.error_stddev:>6.2f}")
    else:
        print(f" {'Benchmark':<15} | {'Backend':<15} | {'Métrica':<15} | {'Valor':<12} | {'Status':<10}")
        print("-" * 94)
        for m in suite.measurements:
            val_str = f"{m.metric_value:.4f}" if isinstance(m.metric_value, float) else str(m.metric_value)
            print(f" {m.benchmark:<15} | {m.backend:<15} | {m.metric_name:<15} | {val_str:<12} | {m.status:<10}")

    print("=" * 94)
    print(f" [Relatório JSON]: {json_path.name}")
    print(f" [Ledger Histórico]: benchmarks/results.csv\n")

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

    any_failed = False
    for bench_cls in benchmarks_to_run:
        bench = bench_cls()
        print(f"\n[+] Executando benchmark: {bench.name} (Modo: {args.mode.upper()})...")
        try:
            suite_res = bench.run(mode=args.mode)
            save_suite_to_csv(suite_res)
            json_path = save_suite_to_json(suite_res)
            print_suite_table(suite_res, json_path)
            for m in suite_res.measurements:
                if m.status != "SUCCESS":
                    any_failed = True
        except Exception as e:
            print(f"[-] Exceção durante execução de {bench.name}: {e}")
            any_failed = True

    if any_failed:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
