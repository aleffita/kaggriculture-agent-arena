"""Probe for Multimodal Encoders (Vision, Audio, Text) in LiteRT-LM."""

import sys
import pathlib

# Ensure repo src/ is importable
repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from litert_explore.engine import resolve_model_path
from litert_explore.gpu import select_gpu
from litert_lm import capabilities
from rich.console import Console
from rich.table import Table

console = Console()

def run_probe():
    console.print("[bold cyan]=== Multimodal Encoders Probe (Target: GTX 1050 Ti) ===[/bold cyan]\n")
    model_path = resolve_model_path()
    
    console.print(f"[yellow]Inspecting model package:[/yellow] {pathlib.Path(model_path).name}")
    caps = capabilities.Capabilities(model_path)
    
    table = Table(title="Model Multimodal & Subsystem Capabilities")
    table.add_column("Capability / Modality", style="bold")
    table.add_column("Supported Status", justify="center")
    table.add_column("Technical Specification")

    # Modalities
    mods = caps.input_modalities
    table.add_row("Text Input", "[green]YES[/green]" if mods.text else "[red]NO[/red]", "Tokenized text embeddings")
    table.add_row("Vision Input", "[green]YES[/green]" if mods.vision else "[red]NO[/red]", f"SigLIP Vision Encoder (Max tokens: {caps.max_vision_token_budget})")
    table.add_row("Audio Input", "[green]YES[/green]" if mods.audio else "[red]NO[/red]", "ASR Audio Encoder / Spectrogram features")
    table.add_row("Video Input", "[green]YES[/green]" if mods.video else "[red]NO[/red]", "Multi-frame spatio-temporal features")
    table.add_row("Speculative Decoding", "[green]YES[/green]" if caps.has_speculative_decoding_support() else "[red]NO[/red]", "Multi-Token Prediction (MTP) Drafter Head")
    table.add_row("Thinking / CoT", "[green]YES[/green]" if caps.supports_thinking() else "[yellow]OFF[/yellow]", "Internal reasoning token channel")
    table.add_row("Tool / Function Calling", "[green]YES[/green]" if caps.supports_function_calling() else "[yellow]OFF[/yellow]", "OpenAI-compatible function schemas")

    console.print(table)
    
    console.print("\n[bold green]Heterogeneous Routing Strategy for 4GB VRAM (GTX 1050 Ti):[/bold green]")
    console.print("  - [bold]Language Decoder[/bold]: GPU (GTX 1050 Ti) via Direct3D 12")
    console.print("  - [bold]Vision Encoder[/bold]:   GPU (GTX 1050 Ti) or CPU (offloaded if VRAM > 3.5GB)")
    console.print("  - [bold]Audio Encoder[/bold]:    CPU (XNNPACK / AVX2 offload preserves ~400MB VRAM)")

if __name__ == "__main__":
    run_probe()
