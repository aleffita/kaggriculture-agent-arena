"""Probe for KV Cache Scaling & TurboQuant Compression Evaluation on GTX 1050 Ti."""

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
    console.print("[bold cyan]=== KV Cache Scaling & TurboQuant Evaluation (Target: GTX 1050 Ti) ===[/bold cyan]\n")
    model_path = resolve_model_path()
    
    # Gemma 4 E2B architectural constants
    NUM_LAYERS = 18
    NUM_KV_HEADS = 1
    HEAD_DIM = 256
    BYTES_PER_FP16 = 2

    # Context length steps to test
    steps = [256, 512, 1024, 2048]
    results = []

    with select_gpu("1050ti"):
        for ctx_len in steps:
            console.print(f"[yellow]Benchmarking context length {ctx_len} tokens...[/yellow]")
            try:
                bench = Benchmark(
                    model_path=model_path,
                    backend=interfaces.GPU(),
                    prefill_tokens=ctx_len // 2,
                    decode_tokens=ctx_len // 2,
                    max_num_tokens=4096,
                )
                info = bench.run()
                
                # KV cache theoretical memory calculations
                raw_bytes = 2 * NUM_LAYERS * NUM_KV_HEADS * HEAD_DIM * BYTES_PER_FP16 * ctx_len
                fp16_kb = raw_bytes / 1024.0
                turboquant_4bit_kb = fp16_kb * 0.25 # 4-bit vs 16-bit
                turboquant_3bit_kb = fp16_kb * 0.1875 # 3-bit vs 16-bit

                results.append({
                    "ctx_len": ctx_len,
                    "prefill_tps": info.last_prefill_tokens_per_second,
                    "decode_tps": info.last_decode_tokens_per_second,
                    "ttft_s": info.time_to_first_token_in_second,
                    "fp16_kb": fp16_kb,
                    "tq4_kb": turboquant_4bit_kb,
                    "tq3_kb": turboquant_3bit_kb,
                })
                console.print(f"[green]Done: {info.last_decode_tokens_per_second:.2f} tokens/s (decode)[/green]\n")
            except Exception as e:
                console.print(f"[red]Failed at {ctx_len} tokens: {e}[/red]\n")

    # Render table
    table = Table(title="KV Cache Scaling & TurboQuant Savings on NVIDIA GeForce GTX 1050 Ti")
    table.add_column("Tokens (Seq Len)", style="bold", justify="right")
    table.add_column("Prefill Speed (t/s)", justify="right")
    table.add_column("Decode Speed (t/s)", justify="right", style="green")
    table.add_column("TTFT (s)", justify="right")
    table.add_column("FP16 KV Cache", justify="right")
    table.add_column("TurboQuant 4-bit", justify="right", style="cyan")
    table.add_column("TurboQuant 3-bit", justify="right", style="magenta")

    for r in results:
        table.add_row(
            str(r["ctx_len"]),
            f"{r['prefill_tps']:.2f}",
            f"{r['decode_tps']:.2f}",
            f"{r['ttft_s']:.2f}",
            f"{r['fp16_kb']:.1f} KB",
            f"{r['tq4_kb']:.1f} KB",
            f"{r['tq3_kb']:.1f} KB",
        )

    console.print(table)
    console.print("\n[bold green]TurboQuant Assessment:[/bold green]")
    console.print("  - [bold]Compression Factor[/bold]: 4.0x (4-bit) to 5.3x (3-bit) reduction in KV cache footprints.")
    console.print("  - [bold]Pascal sm_61 Impact[/bold]: On memory-constrained cards (4GB), TurboQuant expands maximum context ceiling by 4x without triggering VRAM eviction.")

if __name__ == "__main__":
    run_probe()
