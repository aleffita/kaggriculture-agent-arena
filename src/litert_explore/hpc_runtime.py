"""
Heterogeneous HPC Runtime Orchestrator:
Coordinates Dual-GPU Causal Encoder-Decoder (CED) execution,
DMA Pinned Memory Ring Buffering, NVDEC Hardware Acceleration,
and D-Spark Speculative Prefetching across NVIDIA RTX 2060 + GTX 1050 Ti + AMD Ryzen.
"""

from __future__ import annotations

import os
import sys
import pathlib
import subprocess
import ctypes
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

HPC_ENGINE_DIR = pathlib.Path(__file__).parent / "hpc_engine"
HPC_EXE_UNIFIED_PATH = HPC_ENGINE_DIR / "unified_runtime.exe"
HPC_EXE_CED_PATH = HPC_ENGINE_DIR / "heterogeneous_ced_pipeline.exe"
HPC_EXE_MOE_PATH = HPC_ENGINE_DIR / "moe_speculative_ssd_streamer.exe"
HPC_EXE_BONSAI_PATH = HPC_ENGINE_DIR / "ternary_bonsai_dual_gpu.exe"

class HeterogeneousHPCRuntime:
    """Manages physical hardware topology, ASIC allocation and CED execution."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or (
            str(pathlib.Path.home() / ".litert-lm" / "cache" / "huggingface" / 
                "litert-community" / "gemma-4-E2B-it-litert-lm" / "gemma-4-E2B-it.litertlm")
        )
        self.exe_unified = HPC_EXE_UNIFIED_PATH
        self.exe_ced = HPC_EXE_CED_PATH
        self.exe_moe = HPC_EXE_MOE_PATH
        self.exe_bonsai = HPC_EXE_BONSAI_PATH

    def audit_physical_silicon(self) -> Dict[str, Any]:
        """Probes the physical ASICs present in the host system."""
        info = {
            "gpus": [],
            "nvdec_available": False,
            "nvenc_available": False,
            "mmap_available": os.path.exists(self.model_path),
            "model_size_mb": os.path.getsize(self.model_path) / (1024*1024) if os.path.exists(self.model_path) else 0.0
        }

        # Check NVDEC / NVENC DLLs
        sys32 = pathlib.Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32"
        info["nvdec_available"] = (sys32 / "nvcuvid.dll").exists()
        info["nvenc_available"] = (sys32 / "nvEncodeAPI64.dll").exists()

        # Probe CUDA Devices via Driver API
        try:
            nvcuda = ctypes.CDLL("nvcuda.dll")
            nvcuda.cuInit(0)
            count = ctypes.c_int()
            nvcuda.cuDeviceGetCount(ctypes.byref(count))
            for i in range(count.value):
                dev = ctypes.c_int()
                nvcuda.cuDeviceGet(ctypes.byref(dev), i)
                name_buf = ctypes.create_string_buffer(256)
                nvcuda.cuDeviceGetName(name_buf, 256, dev)
                
                # Check VRAM
                bytes_mem = ctypes.c_size_t()
                nvcuda.cuDeviceTotalMem(ctypes.byref(bytes_mem), dev)
                vram_mb = bytes_mem.value // (1024 * 1024)

                info["gpus"].append({
                    "id": i,
                    "name": name_buf.value.decode(errors="ignore"),
                    "vram_mb": vram_mb
                })
        except Exception as e:
            console.print(f"[yellow]Aviso na varredura CUDA: {e}[/yellow]")

        return info

    def display_topology(self, audit: Dict[str, Any]):
        """Renders rich architecture dashboard."""
        table = Table(title="[bold cyan]Topologia de Silício & Alocação de ASICs Heterogêneos[/bold cyan]")
        table.add_column("Módulo de Hardware", style="bold green")
        table.add_column("Dispositivo Físico", style="cyan")
        table.add_column("Papel no Pipeline CED", style="magenta")
        table.add_column("Modo de Execução", style="yellow")

        table.add_row(
            "Causal Encoder (Camadas 0-27)",
            audit["gpus"][1]["name"] if len(audit["gpus"]) > 1 else "GPU Secundária (GTX 1050 Ti)",
            "Prefill, Embeddings & Fronteira h_27",
            "Pascal SM 6.1 (Bit-Linear / Adder Tree)"
        )
        table.add_row(
            "Generative Decoder (Camadas 28-63)",
            audit["gpus"][0]["name"] if audit["gpus"] else "GPU Primária (RTX 2060)",
            "Atenção Compartilhada, LM Head & Verify",
            "Turing SM 7.5 (Warp-Shuffle / IMMA)"
        )
        table.add_row(
            "Transporte Inter-GPU",
            "PCIe Gen3 x1 Bus (~800 MB/s)",
            "Anel de Pinned Host Memory (Zero-Copy)",
            "DMA Copy Engines Assíncronos (12.5 us lat)"
        )
        table.add_row(
            "Streaming de SSD para MoE",
            "Win32 Direct Unbuffered Overlapped I/O",
            "Leitura direta do NVMe SSD Z:\\models",
            "Eagle-3 Prefetch para Shadow Staging Ring"
        )
        table.add_row(
            "Álgebra Discreta sem Dequantização",
            "Walsh-Hadamard (FWHT) + Adder Tree",
            "Rotação ortogonal e somas inteiras puras",
            "128x redução em multiplicações flutuantes"
        )

        console.print(table)
        console.print(Panel(
            f"[bold green]Modelos Validados no Silicio:[/bold green]\n"
            f" * [cyan]Dense Ternary:[/cyan] Z:\\models\\prism-ml\\Ternary-Bonsai-2-27B-gguf\\Ternary-Bonsai-2-27B-PTQ1_0.gguf (5.54 GB, 100% VRAM)\n"
            f" * [magenta]Sparse MoE:[/magenta]   Z:\\models\\lmstudio-community\\gpt-oss-20b-GGUF\\gpt-oss-20b-MXFP4.gguf (11.28 GB, 32 MB VRAM)\n"
            f" * [yellow]Eagle-3 Drafter:[/yellow] Z:\\models\\ggml-org\\gpt-oss-20b-GGUF\\eagle3-gpt-oss-20b-Q8_0.gguf (921 MB)\n"
            f"[bold]NVDEC Silicio Ativo:[/bold] {'Sim' if audit['nvdec_available'] else 'Nao'} | "
            f"[bold]NVENC Silicio Ativo:[/bold] {'Sim' if audit['nvenc_available'] else 'Nao'}",
            title="[bold yellow]Contexto da Engine HPC[/bold yellow]"
        ))

    def run_pipeline(self, mode: str = "unified", tokens: int = 50) -> bool:
        """Executes native C++/CUDA heterogeneous HPC binaries based on selected mode."""
        targets = []
        if mode in ("unified", "auto"):
            targets.append(("Substrato Unificado CED (Engram + Prefill/Decode + Residual Stream)", self.exe_unified))
        if mode in ("ced-ring", "all"):
            targets.append(("Pipeline CED Ring (gemma-4-E2B-it)", self.exe_ced))
        if mode in ("moe-stream", "all"):
            targets.append(("MoE Speculative SSD Streaming (gpt-oss-20b-MXFP4)", self.exe_moe))
        if mode in ("ternary-dense", "all"):
            targets.append(("Ternary-Bonsai 27B Adder Tree Dual-GPU (PTQ1_0)", self.exe_bonsai))

        success = True
        for name, exe in targets:
            if not exe.exists():
                console.print(f"[bold red]Erro:[/bold red] Binario {name} nao encontrado em {exe}")
                success = False
                continue

            console.print(f"\n[bold cyan]=== Disparando Execucao Nativa: {name} ===[/bold cyan]\n")
            cmd = [str(exe)]
            if exe == self.exe_unified:
                cmd.extend(["--tokens", str(tokens)])
                if self.model_path and "bonsai" in self.model_path.lower():
                    cmd.extend(["--model", "bonsai"])
                else:
                    cmd.extend(["--model", "moe"])

            try:
                res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
                console.print(res.stdout)
            except subprocess.CalledProcessError as e:
                console.print(f"[bold red]Erro na execucao de {name}:[/bold red]\n{e.stderr or e.stdout}")
                success = False

        return success

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Heterogeneous HPC Engine CLI")
    parser.add_argument("--mode", choices=["unified", "ced-ring", "moe-stream", "ternary-dense", "all"], default="unified",
                        help="Execution mode for the heterogeneous HPC engine")
    parser.add_argument("--tokens", type=int, default=50, help="Tokens to evaluate")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    args = parser.parse_args()

    runtime = HeterogeneousHPCRuntime(model_path=args.model)
    audit = runtime.audit_physical_silicon()
    runtime.display_topology(audit)
    runtime.run_pipeline(mode=args.mode, tokens=args.tokens)

if __name__ == "__main__":
    main()
