"""Probe for Decode Steps Per Sync & Hyperparameter Ablations on GTX 1050 Ti."""

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
    console.print("[bold cyan]=== Hyperparameter Ablations Probe (Target: GTX 1050 Ti) ===[/bold cyan]\n")
    model_path = resolve_model_path()
    
    sync_steps = [1, 2, 4, 8]
    results = []

    with select_gpu("1050ti"):
        for step in sync_steps:
            console.print(f"[yellow]Evaluating gpu_decode_steps_per_sync = {step}...[/yellow]")
            try:
                gpu_backend = interfaces.GPU(gpu_decode_steps_per_sync=step)
                bench = Benchmark(
                    model_path=model_path,
                    backend=gpu_backend,
                    prefill_tokens=256,
                    decode_tokens=256,
                    max_num_tokens=4096,
                )
                info = bench.run()
                results.append({
                    "sync_steps": step,
                    "prefill_tps": info.last_prefill_tokens_per_second,
                    "decode_tps": info.last_decode_tokens_per_second,
                    "ttft_s": info.time_to_first_token_in_second,
                    "init_s": info.init_time_in_second,
                })
                console.print(f"[green]Done: {info.last_decode_tokens_per_second:.2f} t/s (decode)[/green]\n")
            except Exception as e:
                console.print(f"[red]Error at sync step {step}: {e}[/red]\n")

    # Render table
    table = Table(title="Ablation: gpu_decode_steps_per_sync on NVIDIA GeForce GTX 1050 Ti")
    table.add_column("Steps Per Sync", style="bold", justify="center")
    table.add_column("Prefill Speed (t/s)", justify="right")
    table.add_column("Decode Speed (t/s)", justify="right", style="green")
    table.add_column("TTFT (s)", justify="right")
    table.add_column("Init Time (s)", justify="right")

    for r in results:
        table.add_row(
            str(r["sync_steps"]),
            f"{r['prefill_tps']:.2f}",
            f"{r['decode_tps']:.2f}",
            f"{r['ttft_s']:.2f}",
            f"{r['init_s']:.2f}",
        )

    console.print(table)

if __name__ == "__main__":
    run_probe()
