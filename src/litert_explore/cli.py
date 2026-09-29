"""CLI entrypoints for litert-explore."""

from __future__ import annotations

import argparse
import sys
from rich.console import Console

from .benchmark import run_benchmark, run_comparative_benchmark
from .gpu import list_gpus, get_shim_dll_path
from .engine import LiteRtModelRunner, resolve_model_path

console = Console()

def gpu_cli():
    """Lists GPUs and shows shim status."""
    console.print("[bold cyan]=== Diagnóstico de GPUs e DXGI Shim ===[/bold cyan]\n")
    gpus = list_gpus()
    if gpus:
        for idx, g in enumerate(gpus):
            console.print(f"  [bold]GPU {idx}[/bold]: {g.get('name')} | VRAM: {g.get('vram_mb')} MB | ID: {g.get('device_id')}")
    else:
        console.print("  [yellow]Nenhuma GPU enumerada via CIM.[/yellow]")

    try:
        shim = get_shim_dll_path()
        console.print(f"\n[green]DXGI Shim localizado:[/green] {shim}")
    except Exception as e:
        console.print(f"\n[red]DXGI Shim não encontrado:[/red] {e}")


def bench_cli():
    """Runs benchmark CLI."""
    parser = argparse.ArgumentParser(description="LiteRT-LM Benchmark Runner")
    parser.add_argument("--target", choices=["1050ti", "2060", "cpu", "all"], default="1050ti",
                        help="Target device to execute benchmark")
    parser.add_argument("--model", type=str, default=None, help="Path to .litertlm model file")
    args = parser.parse_args()

    if args.target == "all":
        run_comparative_benchmark(args.model)
    else:
        console.print(f"[cyan]Executando benchmark no alvo: {args.target}...[/cyan]")
        res = run_benchmark(target=args.target, model_path=args.model)
        console.print(f"[bold green]Resultado ({res.target}):[/bold green]")
        console.print(f"  Prefill Speed: {res.prefill_speed:.2f} t/s")
        console.print(f"  Decode Speed:  {res.decode_speed:.2f} t/s")
        console.print(f"  TTFT:          {res.time_to_first_token:.2f} s")
        console.print(f"  Init Time:     {res.init_time:.2f} s")


def chat_cli(target: str = "1050ti", model: str | None = None):
    """Starts interactive chat session."""
    console.print(f"[bold green]Iniciando sessão interativa no alvo: {target}...[/bold green]")
    runner = LiteRtModelRunner(model_path=model, backend="cpu" if target == "cpu" else "gpu", gpu_target=target)
    console.print("[cyan]Modelo carregado. Digite sua mensagem (ou 'exit' para sair):[/cyan]\n")

    while True:
        try:
            prompt = console.input("[bold yellow]Você:[/bold yellow] ").strip()
            if not prompt or prompt.lower() in ("exit", "quit"):
                break
            console.print("[bold green]Modelo:[/bold green] ", end="")
            for chunk in runner.stream(prompt):
                console.print(chunk, end="")
            console.print("\n")
        except (KeyboardInterrupt, EOFError):
            break


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(prog="litert-explore", description="LiteRT-LM Exploration & Multi-GPU Suite")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # bench
    bench_parser = subparsers.add_parser("bench", help="Run model benchmark")
    bench_parser.add_argument("--target", choices=["1050ti", "2060", "cpu", "all"], default="1050ti")
    bench_parser.add_argument("--model", type=str, default=None)

    # gpus
    subparsers.add_parser("gpus", help="List available GPUs and DXGI shim status")

    # chat
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat with model")
    chat_parser.add_argument("--target", choices=["1050ti", "2060", "cpu"], default="1050ti")
    chat_parser.add_argument("--model", type=str, default=None)

    args = parser.parse_args()

    if args.command == "bench":
        if args.target == "all":
            run_comparative_benchmark(args.model)
        else:
            res = run_benchmark(target=args.target, model_path=args.model)
            console.print(f"[bold green]Resultado ({res.target}):[/bold green]")
            console.print(f"  Prefill: {res.prefill_speed:.2f} t/s | Decode: {res.decode_speed:.2f} t/s")
    elif args.command == "gpus":
        gpu_cli()
    elif args.command == "chat":
        chat_cli(args.target, args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
