"""Probe for Multi-Token Prediction (MTP) Speculative Decoding on GTX 1050 Ti."""

import os
import sys
import pathlib

# Ensure repo src/ is importable
repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from litert_explore.gpu import select_gpu
from litert_explore.engine import resolve_model_path
from litert_lm.benchmark import Benchmark
from litert_lm import interfaces
from rich.console import Console
from rich.table import Table

console = Console()

def run_probe():
    console.print("[bold cyan]=== MTP Speculative Decoding Probe (Target: GTX 1050 Ti) ===[/bold cyan]\n")
    model_path = resolve_model_path()
    
    results = {}
    configs = [
        ("Autoregressive (MTP Disabled)", False),
        ("Speculative Decoding (MTP Enabled)", True),
    ]

    with select_gpu("1050ti"):
        for label, mtp_flag in configs:
            console.print(f"[yellow]Evaluating: {label}...[/yellow]")
            bench = Benchmark(
                model_path=model_path,
                backend=interfaces.GPU(),
                prefill_tokens=256,
                decode_tokens=256,
                max_num_tokens=4096,
                enable_speculative_decoding=mtp_flag,
            )
            info = bench.run()
            results[label] = {
                "prefill_tps": info.last_prefill_tokens_per_second,
                "decode_tps": info.last_decode_tokens_per_second,
                "ttft_s": info.time_to_first_token_in_second,
                "init_time_s": info.init_time_in_second,
            }
            console.print(f"[green]Done: {info.last_decode_tokens_per_second:.2f} tokens/s (decode)[/green]\n")

    # Render Summary Table
    table = Table(title="MTP Speculative Decoding Comparison on NVIDIA GeForce GTX 1050 Ti")
    table.add_column("Configuration", style="bold")
    table.add_column("Prefill Speed (t/s)", justify="right")
    table.add_column("Decode Speed (t/s)", justify="right", style="green")
    table.add_column("Speedup vs Baseline", justify="right", style="cyan")
    table.add_column("TTFT (s)", justify="right")

    base_decode = results["Autoregressive (MTP Disabled)"]["decode_tps"]

    for label, d in results.items():
        speedup = d["decode_tps"] / base_decode if base_decode > 0 else 1.0
        table.add_row(
            label,
            f"{d['prefill_tps']:.2f}",
            f"{d['decode_tps']:.2f}",
            f"{speedup:.2f}x",
            f"{d['ttft_s']:.2f}",
        )

    console.print(table)

if __name__ == "__main__":
    run_probe()
