"""Probe Runner for Antigravity Hybrid Orchestrator on GTX 1050 Ti."""

import os
import sys
import pathlib
import time

# Ensure repo src/ is importable
repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from litert_explore.gpu import select_gpu

try:
    from .slingshot_stream import SlingshotTelemetryStream
    from .hybrid_orchestrator import (
        extract_public_skeleton,
        synthesize_cloud_contract,
        LocalGemmaSentinel,
    )
except ImportError:
    from slingshot_stream import SlingshotTelemetryStream
    from hybrid_orchestrator import (
        extract_public_skeleton,
        synthesize_cloud_contract,
        LocalGemmaSentinel,
    )


console = Console()

SAMPLE_TARGET_MODULE = '''\
"""Network Gateway Rate Limiting & Queue Orchestration."""
import os
import time

PRIVATE_GATEWAY_TOKEN = "Bearer secret_prod_live_8943729851"
INTERNAL_DATABASE_URL = "postgres://admin:supersecret@10.0.0.45:5432/gateway"

throttle_level = 0
queue_depth = 120

def process_event(event_id: str, payload_size: int) -> bool:
    """Processes incoming network packet with internal validation."""
    if payload_size > 1500:
        return False
    return True

def apply_throttle(level: int) -> int:
    """Modifies global throttle setting."""
    global throttle_level
    throttle_level = level
    return throttle_level
'''


def run_probe():
    console.print("\n[bold cyan]=== Antigravity Hybrid Orchestrator Probe (Target: GTX 1050 Ti) ===[/bold cyan]\n")

    # Step 1: AST Skeleton Extraction
    console.print("[bold yellow]Step 1: Extracting Public AST Skeleton (Privacy Invariant)...[/bold yellow]")
    skeleton = extract_public_skeleton(SAMPLE_TARGET_MODULE)
    console.print(Panel(
        f"[green]{skeleton}[/green]\n\n"
        "[dim]Notice: PRIVATE_GATEWAY_TOKEN, INTERNAL_DATABASE_URL, and function bodies were completely stripped.[/dim]",
        title="Zero-Body AST Skeleton (Sent to Cloud Architect)",
        border_style="cyan"
    ))

    # Step 2: Synthesize Cloud Architect Contract
    console.print("\n[bold yellow]Step 2: Cloud Architect Invariant Contract Synthesis...[/bold yellow]")
    contract = synthesize_cloud_contract("gateway_limiter.py", skeleton)
    console.print(Panel(
        f"[bold]Role:[/bold] {contract.role}\n"
        f"[bold]Strategy:[/bold] {contract.strategy}\n"
        f"[bold]Invariants:[/bold]\n" + "\n".join(f"  • {inv}" for inv in contract.invariants) + "\n"
        f"[bold]Allowed On-Device Tools:[/bold] {', '.join(contract.allowed_tools)}\n\n"
        "[dim]Status: Cloud Architect session transitioned to IDLE. Handing control to local GTX 1050 Ti.[/dim]",
        title="Cloud Architect Blueprint (Gemini 3.8 Flash Contract)",
        border_style="magenta"
    ))

    # Step 3: Initialize Local Gemma Sentinel on GTX 1050 Ti
    console.print("\n[bold yellow]Step 3: Initializing On-Device Gemma 4 on GTX 1050 Ti (Pure Text Mode)...[/bold yellow]")
    sentinel = LocalGemmaSentinel()
    
    with select_gpu("1050ti"):
        t0 = time.time()
        sentinel.initialize_on_1050ti()
        init_time = time.time() - t0
        console.print(f"[green]LiteRT-LM Engine successfully resident on GTX 1050 Ti (Init: {init_time:.2f}s).[/green]\n")

        # Step 4: Stream Slingshot Telemetry Events
        console.print("[bold yellow]Step 4: Processing Stochastic Slingshot Telemetry Stream ('Estilingue na Banda')...[/bold yellow]")
        stream = SlingshotTelemetryStream(base_interval_s=1.0, burst_probability=0.35, anomaly_threshold=80)
        
        events = list(stream.iter_events_fast(max_events=6))
        results = []

        for idx, event in enumerate(events, start=1):
            console.print(f"  Evaluating tick #{idx}: Value={event.value}/100 [{event.label}] (Burst: {event.is_burst})...")
            turn_result = sentinel.evaluate_telemetry_turn(event, contract)
            results.append(turn_result)

    # Step 5: Render Results Table
    table = Table(title="Autonomous On-Device Telemetry Evaluations (GTX 1050 Ti)")
    table.add_column("Tick", justify="center")
    table.add_column("Metric Value", justify="right", style="bold")
    table.add_column("Phase", justify="center")
    table.add_column("Action Taken", justify="center", style="cyan")
    table.add_column("Throttle Level", justify="center", style="magenta")
    table.add_column("Model Explanation", style="dim")

    for idx, r in enumerate(results, start=1):
        ev = r["event"]
        dec = r["decision"]
        action = dec.get("action", "NOMINAL")
        action_style = "[bold red]THROTTLE_APPLIED[/bold red]" if "THROTTLE" in action else "[green]NOMINAL[/green]"
        table.add_row(
            str(idx),
            f"{ev.value}/100",
            f"[bold red]BURST[/bold red]" if ev.is_burst else "Nominal",
            action_style,
            f"Level {r['active_throttle']}",
            dec.get("summary", ""),
        )

    console.print("\n", table)

    # Step 6: Telemetry Anomaly Log
    console.print(f"\n[bold green]Local On-Device Audit Register ({len(sentinel.anomaly_log)} entries recorded):[/bold green]")
    for item in sentinel.anomaly_log:
        console.print(f"  • [red]ANOMALY[/red] Metric: {item['metric']} | Active Throttle: Level {item['throttle_active']} | Reason: {item['reason']}")

    console.print("\n[bold cyan]Hybrid Orchestrator Probe completed successfully on GPU 1 (GTX 1050 Ti).[/bold cyan]")


if __name__ == "__main__":
    run_probe()
