"""Probe for Thinking & Reasoning configurations on GTX 1050 Ti."""

import os
import sys
import time
import pathlib

# Ensure repo src/ is importable
repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from litert_explore.gpu import select_gpu
from litert_explore.engine import resolve_model_path
import litert_lm
from litert_lm import interfaces
from rich.console import Console
from rich.table import Table

console = Console()

def run_probe():
    console.print("[bold cyan]=== Thinking & Reasoning Probe (Target: GTX 1050 Ti) ===[/bold cyan]\n")
    model_path = resolve_model_path()
    
    test_prompt = "Solve step-by-step: A car travels 120 km at 60 km/h, and then 120 km at 40 km/h. What is the average speed of the entire trip?"

    budgets = [
        ("Standard / No Thinking Budget", None),
        ("Thinking Budget: 256 tokens", 256),
        ("Thinking Budget: 512 tokens", 512),
    ]

    results = []

    with select_gpu("1050ti"):
        console.print("[yellow]Initializing LiteRT-LM Engine on GTX 1050 Ti...[/yellow]")
        engine = litert_lm.Engine(
            model_path=model_path,
            backend=interfaces.GPU(),
            max_num_tokens=2048,
        )

        for label, budget in budgets:
            console.print(f"[cyan]Testing config: {label}...[/cyan]")
            thinking_cfg = None
            if budget is not None:
                thinking_cfg = litert_lm.ThinkingConfig(enable_thinking=True, thinking_token_budget=budget)

            start = time.perf_counter()
            conv = engine.create_conversation(
                thinking_config=thinking_cfg,
                max_output_tokens=512,
            )
            response = conv.send_message(test_prompt)
            elapsed = time.perf_counter() - start

            resp_text = ""
            if isinstance(response, dict):
                resp_text = "".join(
                    c.get("text", "") for c in response.get("content", []) if isinstance(c, dict)
                )
            elif isinstance(response, str):
                resp_text = response

            token_estimate = len(resp_text.split()) * 1.3
            tps = token_estimate / elapsed if elapsed > 0 else 0

            results.append({
                "label": label,
                "elapsed_s": elapsed,
                "char_len": len(resp_text),
                "est_tokens": int(token_estimate),
                "tps": tps,
                "preview": resp_text[:120].replace("\n", " "),
            })
            console.print(f"[green]Completed in {elapsed:.2f}s (~{tps:.1f} est. tokens/s)[/green]\n")

    # Render summary table
    table = Table(title="Thinking Budget Probes on NVIDIA GeForce GTX 1050 Ti")
    table.add_column("Configuration", style="bold")
    table.add_column("Elapsed (s)", justify="right")
    table.add_column("Output Size", justify="right")
    table.add_column("Est. Rate", justify="right", style="green")
    table.add_column("Output Preview", style="dim")

    for r in results:
        table.add_row(
            r["label"],
            f"{r['elapsed_s']:.2f}s",
            f"{r['est_tokens']} tokens",
            f"{r['tps']:.1f} t/s",
            f"{r['preview']}...",
        )

    console.print(table)

if __name__ == "__main__":
    run_probe()
