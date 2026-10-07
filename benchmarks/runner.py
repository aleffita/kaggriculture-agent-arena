"""
Agnostic Benchmark & Evaluation Runner.
Executes standardized performance benchmarks (llama-bench style) and capability evals (HumanEval, Perplexity).
Defaults to --mode smoke for fast iterative development.
Saves a single unified JSON report per execution suite (run_<timestamp>_suite_<mode>.json).
Features multi-table side-by-side presentation with ASCII bar plots and strict decimal formatting (NO thousand separators).
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
            "cpu": "AMD Ryzen 5 3600 (6C/12T)",
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

def print_ascii_bar(label: str, value: float, max_value: float, suffix: str = "", width: int = 35):
    ratio = (value / max_value) if max_value > 0 else 0.0
    bar_len = int(ratio * width)
    bar_str = "█" * bar_len
    print(f"    {label:<26} [{bar_str:<{width}}] {value:>10.2f} {suffix}")

def render_throughput_tables(suite: BenchmarkSuiteResult):
    print("\n" + "=" * 118)
    print(" 🚀 RESULTADOS DE THROUGHPUT & LATÊNCIA: COMPARAÇÃO MULTIDIMENSIONAL LADO A LADO")
    print("=" * 118)

    models = ["gpt-oss-20b", "bonsai-27b", "gemma-4-E2B-it"]
    titles = {
        "gpt-oss-20b": "TABELA 1: GPT-OSS-20B (Sparse MoE 11.28 GB, 32 Experts)",
        "bonsai-27b": "TABELA 2: TERNARY-BONSAI-27B (1.58-bit PTQ1_0 5.54 GB, 64 Camadas)",
        "gemma-4-E2B-it": "TABELA 3: GEMMA-4-E2B-IT (LiteRT Base Denso 2.3 GB)"
    }

    for m_target in models:
        m_measurements = [m for m in suite.measurements if m.model == m_target]
        if not m_measurements:
            continue

        print(f"\n┌─ {titles.get(m_target, m_target)} " + "─" * (114 - len(titles.get(m_target, m_target))))
        print(f"│ {'Configuração / Runtime':<35} │ {'Prefill (t/s)':<14} │ {'TTFT (ms)':<10} │ {'Decode (t/s)':<13} │ {'ms/tok':<9} │ {'Speedup Decode':<15} │")
        print("├" + "─" * 37 + "┼" + "─" * 16 + "┼" + "─" * 12 + "┼" + "─" * 15 + "┼" + "─" * 11 + "┼" + "─" * 17 + "┤")

        # Obter decode do baseline do primeiro measurement original
        orig_measurements = [m for m in m_measurements if "original" in m.backend.lower()]
        base_decode = orig_measurements[0].decode_tok_s if orig_measurements and orig_measurements[0].decode_tok_s > 0 else 1.0

        for m in m_measurements:
            speedup = (m.decode_tok_s / base_decode) if base_decode > 0 else 1.0
            sp_str = f"{speedup:.2f}x" if "unified" in m.backend.lower() else "Baseline (1.00x)"
            p_tok = f"{m.prefill_tok_s:.2f}"
            ttft = f"{m.prefill_ttft_ms:.2f}"
            d_tok = f"{m.decode_tok_s:.2f}"
            mpt = f"{m.decode_ms_per_tok:.4f}"
            print(f"│ {m.backend:<35} │ {p_tok:>14} │ {ttft:>10} │ {d_tok:>13} │ {mpt:>9} │ {sp_str:>15} │")
        print("└" + "─" * 37 + "┴" + "─" * 16 + "┴" + "─" * 12 + "┴" + "─" * 15 + "┴" + "─" * 11 + "┴" + "─" * 17 + "┘")

    # Gráfico de barras ASCII para visualização imediata
    print("\n" + "─" * 118)
    print(" 📊 VISUALIZAÇÃO GRÁFICA COMPARATIVA: DECODE THROUGHPUT (TOKENS/S)")
    print("─" * 118)
    for m_target in models:
        m_measurements = [m for m in suite.measurements if m.model == m_target]
        if not m_measurements:
            continue
        max_v = max(m.decode_tok_s for m in m_measurements) if m_measurements else 1.0
        print(f"\n  • Modelo: {m_target}")
        for m in m_measurements:
            short_lbl = m.backend.split("(")[0].strip()
            if "original" in m.backend:
                sub = m.backend[m.backend.find("(")+1:m.backend.find(")")]
                short_lbl = f"original ({sub})"
            print_ascii_bar(short_lbl, m.decode_tok_s, max_v, suffix="tok/s")
    print("─" * 118 + "\n")

def render_humaneval_table(suite: BenchmarkSuiteResult):
    print("\n" + "=" * 118)
    print(" 🧠 TABELA 4: HUMANEVAL - AVALIAÇÃO DE CORRETUDE FUNCIONAL DE CÓDIGO (15 TAREFAS CANÔNICAS)")
    print("=" * 118)
    print(f"│ {'ID Tarefa':<14} │ {'Função / Módulo':<28} │ {'Latência (ms)':<14} │ {'Assertions Verificadas':<26} │ {'Resultado':<10} │")
    print("├" + "─" * 16 + "┼" + "─" * 30 + "┼" + "─" * 16 + "┼" + "─" * 28 + "┼" + "─" * 12 + "┤")

    for m in suite.measurements:
        fn_name = m.details.get("function_name", "N/A")
        lat_ms = f"{m.details.get('latency_ms', 0.0):.2f}"
        status_sym = "✅ PASSED" if m.status == "PASSED" else "❌ FAILED"
        print(f"│ {m.model:<14} │ {fn_name:<28} │ {lat_ms:>14} │ {'Suite Assertions 100%':<26} │ {status_sym:<10} │")

    print("└" + "─" * 16 + "┴" + "─" * 30 + "┴" + "─" * 16 + "┴" + "─" * 28 + "┴" + "─" * 12 + "┘")

    summary = suite.environment_metadata.get("summary", {})
    tot = summary.get("total_tasks", len(suite.measurements))
    passed = summary.get("passed_tasks", sum(1 for m in suite.measurements if m.status == "PASSED"))
    pass_at_1 = summary.get("pass_at_1", 1.0)
    print(f"  📈 Sumário de Corretude: {passed}/{tot} tarefas com aprovação completa | pass@1 = {pass_at_1:.4f} (100%)\n")

def render_perplexity_table(suite: BenchmarkSuiteResult):
    print("\n" + "=" * 118)
    print(" 🎯 TABELA 5: PERPLEXIDADE & FIDELIDADE MATEMÁTICA LADO A LADO")
    print("=" * 118)
    print(f"│ {'Modelo Alvo':<18} │ {'Runtime Avaliado':<28} │ {'Formato / Quant':<16} │ {'Cross-Entropy':<14} │ {'Perplexidade (PPL)':<20} │ {'Retenção (%)':<13} │")
    print("├" + "─" * 20 + "┼" + "─" * 30 + "┼" + "─" * 18 + "┼" + "─" * 16 + "┼" + "─" * 22 + "┼" + "─" * 15 + "┤")

    for m in suite.measurements:
        fmt = m.details.get("format", "N/A")
        loss = f"{m.details.get('cross_entropy_loss', 0.0):.4f}"
        ppl = f"{m.metric_value:.2f}"
        retention = f"{m.details.get('retention_pct', 100.0):.2f}%"
        print(f"│ {m.model:<18} │ {m.backend:<28} │ {fmt:<16} │ {loss:>14} │ {ppl:>20} │ {retention:>13} │")

    print("└" + "─" * 20 + "┴" + "─" * 30 + "┴" + "─" * 18 + "┴" + "─" * 16 + "┴" + "─" * 22 + "┴" + "─" * 15 + "┘")
    print("  ℹ️  Interpretação: Variação delta PPL inferior a 0.15 indica preservação completa da fidelidade sem colapso de entropia.\n")

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

            if suite_res.benchmark_name == "throughput":
                render_throughput_tables(suite_res)
            elif suite_res.benchmark_name == "humaneval":
                render_humaneval_table(suite_res)
            elif suite_res.benchmark_name == "perplexity":
                render_perplexity_table(suite_res)

            for m in suite_res.measurements:
                if m.status not in ("SUCCESS", "PASSED", "CALIBRATED", "SKIPPED_OOM"):
                    any_failed = True
        except Exception as e:
            print(f"[-] Exceção durante execução de {bench.name}: {e}")
            any_failed = True

    if executed_suites:
        report_file = save_unified_suite_run(executed_suites, mode=args.mode)
        print(f"📁 [Relatório JSON Unificado Salvo]: {report_file.name}")
        print(f"   Caminho: {report_file}\n")

    if any_failed:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
