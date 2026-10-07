"""
Evaluator Plugin: Prefill & Decode Benchmark for Unified Heterogeneous CED Runtime.
Evaluates both GPT-OSS-20B (MoE) and Ternary-Bonsai-27B on the unified engine.
"""
import subprocess
import json
from pathlib import Path
from typing import Dict, Any
from .base import BaseBenchmarkPlugin, BenchmarkResult

class PrefillDecodeEvalPlugin(BaseBenchmarkPlugin):
    name = "prefill_decode_eval"
    description = "Avaliação de Prefill e Decode de modelos MoE e Ternário no Unified CED Runtime"

    def __init__(self):
        self.workspace_root = Path(__file__).resolve().parent.parent.parent
        self.unified_exe = self.workspace_root / "src" / "litert_explore" / "hpc_engine" / "unified_runtime.exe"

    def run(self) -> BenchmarkResult:
        if not self.unified_exe.exists():
            return BenchmarkResult(
                plugin_name=self.name,
                target_hardware="Dual-GPU (GTX 1050 Ti + RTX 2060)",
                status="FAILED",
                error_message=f"Binário {self.unified_exe} não encontrado."
            )

        # 1. Avaliar modelo MoE (GPT-OSS-20B)
        cmd_moe = [str(self.unified_exe), "--model", "moe", "--prompt-len", "128", "--tokens", "30", "--json"]
        proc_moe = subprocess.run(cmd_moe, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc_moe.returncode != 0:
            return BenchmarkResult(
                plugin_name=self.name,
                target_hardware="Dual-GPU (GTX 1050 Ti + RTX 2060)",
                status="FAILED",
                error_message=proc_moe.stderr or proc_moe.stdout
            )
        data_moe = json.loads(proc_moe.stdout)

        # 2. Avaliar modelo Ternário (Ternary-Bonsai-27B)
        cmd_bonsai = [str(self.unified_exe), "--model", "bonsai", "--prompt-len", "128", "--tokens", "30", "--json"]
        proc_bonsai = subprocess.run(cmd_bonsai, capture_output=True, text=True, encoding="utf-8", errors="replace")
        data_bonsai = json.loads(proc_bonsai.stdout) if proc_bonsai.returncode == 0 else {}

        # Métrica consolidada (foco principal em MoE)
        return BenchmarkResult(
            plugin_name=self.name,
            target_hardware=f"{data_moe['hardware']['gpu1']} + {data_moe['hardware']['gpu0']}",
            prefill_tok_s=data_moe.get("prefill_tok_s", 0.0),
            prefill_ttft_ms=data_moe.get("prefill_ttft_ms", 0.0),
            decode_tok_s=data_moe.get("decode_tok_s", 0.0),
            decode_latency_ms=data_moe.get("decode_latency_ms", 0.0),
            stalls=data_moe.get("stalls", 0),
            speedup=data_moe.get("speedup", 1.0),
            extra_metrics={
                "gpt_oss_20b_moe": data_moe,
                "ternary_bonsai_27b": data_bonsai
            },
            status="SUCCESS"
        )
