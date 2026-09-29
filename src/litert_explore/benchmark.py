"""Automated multi-device benchmarking suite for LiteRT-LM."""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table

from .engine import resolve_model_path
from .gpu import select_gpu

try:
    from litert_lm.benchmark import Benchmark
    from litert_lm import interfaces
except ImportError:
    Benchmark = None
    interfaces = None

@dataclasses.dataclass
class BenchmarkResult:
    target: str
    backend: str
    prefill_speed: float
    decode_speed: float
    init_time: float
    time_to_first_token: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def run_benchmark(
    target: str = "1050ti",
    model_path: Optional[str] = None,
    prefill_tokens: int = 256,
    decode_tokens: int = 256,
    max_num_tokens: int = 4096,
    runs: int = 1,
) -> BenchmarkResult:
    """Runs a single benchmark targeting a specific device: 'cpu', '2060', or '1050ti'."""
    path = resolve_model_path(model_path)
    target_clean = target.lower().strip()

    if target_clean == "cpu":
        backend_obj = interfaces.CPU()
        bench = Benchmark(
            model_path=path,
            backend=backend_obj,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            max_num_tokens=max_num_tokens,
        )
        info = bench.run()
        return BenchmarkResult(
            target="CPU (Host)",
            backend="cpu",
            prefill_speed=info.last_prefill_tokens_per_second,
            decode_speed=info.last_decode_tokens_per_second,
            init_time=info.init_time_in_second,
            time_to_first_token=info.time_to_first_token_in_second,
        )

    # GPU execution (2060 or 1050ti)
    gpu_idx = 1 if target_clean in ("1", "1050", "1050ti") else 0
    dev_name = "NVIDIA GeForce GTX 1050 Ti" if gpu_idx == 1 else "NVIDIA GeForce RTX 2060"

    with select_gpu(gpu_idx):
        backend_obj = interfaces.GPU()
        bench = Benchmark(
            model_path=path,
            backend=backend_obj,
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            max_num_tokens=max_num_tokens,
        )
        info = bench.run()

    return BenchmarkResult(
        target=dev_name,
        backend=f"gpu (adapter {gpu_idx})",
        prefill_speed=info.last_prefill_tokens_per_second,
        decode_speed=info.last_decode_tokens_per_second,
        init_time=info.init_time_in_second,
        time_to_first_token=info.time_to_first_token_in_second,
    )


def run_comparative_benchmark(model_path: Optional[str] = None) -> List[BenchmarkResult]:
    """Runs sequential benchmarks across all 3 targets: RTX 2060, GTX 1050 Ti, and CPU."""
    console = Console()
    console.print("[bold cyan]Iniciando suite de benchmarks comparativos no LiteRT-LM...[/bold cyan]\n")

    targets = ["2060", "1050ti", "cpu"]
    results = []

    for t in targets:
        console.print(f"[yellow]Executando teste no alvo: {t.upper()}...[/yellow]")
        try:
            res = run_benchmark(target=t, model_path=model_path)
            results.append(res)
            console.print(f"[green]Concluído: {res.target} -> Prefill: {res.prefill_speed:.2f} t/s, Decode: {res.decode_speed:.2f} t/s[/green]\n")
        except Exception as e:
            console.print(f"[red]Erro ao executar em {t}: {e}[/red]\n")

    # Render table
    table = Table(title="Resultados Comparativos de Benchmark (LiteRT-LM)")
    table.add_column("Dispositivo / Alvo", style="bold")
    table.add_column("Backend")
    table.add_column("Prefill Speed (t/s)", justify="right")
    table.add_column("Decode Speed (t/s)", justify="right", style="green")
    table.add_column("Time to 1st Token (s)", justify="right")
    table.add_column("Init Time (s)", justify="right")

    for r in results:
        table.add_row(
            r.target,
            r.backend,
            f"{r.prefill_speed:.2f}",
            f"{r.decode_speed:.2f}",
            f"{r.time_to_first_token:.2f}",
            f"{r.init_time:.2f}",
        )

    console.print(table)
    return results
