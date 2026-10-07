"""
HPC Benchmarks & Auto-Research Suite - Modular Runner
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

# Garantir importação local
BENCHMARKS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARKS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.plugins.base import BaseBenchmarkPlugin, BenchmarkResult
from benchmarks.plugins.pcie_payload_curve import PciePayloadCurvePlugin
from benchmarks.plugins.moe_dual_gpu_ring import MoeDualGpuRingPlugin
from benchmarks.plugins.bvh_moe_router import BvhMoeRouterPlugin

REGISTERED_PLUGINS: Dict[str, Type[BaseBenchmarkPlugin]] = {
    "pcie_payload_curve": PciePayloadCurvePlugin,
    "moe_dual_gpu_ring": MoeDualGpuRingPlugin,
    "bvh_moe_router": BvhMoeRouterPlugin,
}

CSV_FILE = BENCHMARKS_DIR / "results.csv"
REPORTS_DIR = BENCHMARKS_DIR / "reports"

CSV_COLUMNS = [
    "timestamp",
    "plugin",
    "hardware",
    "prefill_tok_s",
    "prefill_ttft_ms",
    "decode_tok_s",
    "decode_latency_ms",
    "stalls",
    "speedup",
    "status"
]

def ensure_environment():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_FILE.exists():
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

def save_result_to_csv(result: BenchmarkResult):
    ensure_environment()
    row = result.to_csv_row()
    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)

def save_result_to_json(result: BenchmarkResult):
    ensure_environment()
    safe_ts = result.timestamp.replace(":", "-").replace(".", "_")
    report_file = REPORTS_DIR / f"run_{safe_ts}_{result.plugin_name}.json"
    with open(report_file, mode="w", encoding="utf-8") as f:
        json.dump(result.__dict__, f, indent=2, ensure_ascii=False)
    return report_file

def print_result_card(result: BenchmarkResult, json_path: Path):
    print("\n" + "=" * 78)
    print(f" 🚀 BENCHMARK CONCLUÍDO: [{result.plugin_name}]")
    print("=" * 78)
    print(f" Hardware Alvo:       {result.target_hardware}")
    print(f" Status:              {result.status}")
    if result.prefill_tok_s > 0 or result.prefill_ttft_ms > 0:
        print(f" Prefill Throughput:  {result.prefill_tok_s:,.2f} tok/s (TTFT: {result.prefill_ttft_ms:.2f} ms)")
    if result.decode_tok_s > 0 or result.decode_latency_ms > 0:
        print(f" Decode Throughput:   {result.decode_tok_s:,.2f} tok/s (Latência: {result.decode_latency_ms:.2f} ms)")
    print(f" Stalls Registrados:  {result.stalls} stalls")
    print(f" Speedup / Ganho:     {result.speedup:.2f}x")
    print(f" Relatório JSON:      {json_path.name}")
    print(f" Ledger Atualizado:   benchmarks/results.csv")
    print("=" * 78 + "\n")

def main():
    parser = argparse.ArgumentParser(description="HPC Benchmarks & Auto-Research Suite")
    parser.add_argument("--plugin", "-p", type=str, help="Nome do plugin de benchmark a executar")
    parser.add_argument("--all", "-a", action="store_true", help="Executar todos os plugins registrados")
    parser.add_argument("--list", "-l", action="store_true", help="Listar todos os plugins registrados")

    args = parser.parse_args()

    if args.list:
        print("\n📋 Plugins de Benchmark Registrados:")
        for name, cls in REGISTERED_PLUGINS.items():
            print(f"  - {name:<22}: {cls.description}")
        print()
        sys.exit(0)

    plugins_to_run: List[Type[BaseBenchmarkPlugin]] = []

    if args.all:
        plugins_to_run = list(REGISTERED_PLUGINS.values())
    elif args.plugin:
        if args.plugin not in REGISTERED_PLUGINS:
            print(f"[-] Erro: Plugin '{args.plugin}' não encontrado.")
            print(f"    Disponíveis: {list(REGISTERED_PLUGINS.keys())}")
            sys.exit(1)
        plugins_to_run = [REGISTERED_PLUGINS[args.plugin]]
    else:
        parser.print_help()
        sys.exit(0)

    any_failed = False
    for plugin_cls in plugins_to_run:
        plugin = plugin_cls()
        print(f"\n[+] Executando plugin: {plugin.name} ({plugin.description})...")
        try:
            result = plugin.run()
            save_result_to_csv(result)
            json_path = save_result_to_json(result)
            print_result_card(result, json_path)
            if result.status != "SUCCESS":
                any_failed = True
        except Exception as e:
            print(f"[-] Exceção durante a execução do plugin {plugin.name}: {e}")
            any_failed = True

    if any_failed:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
