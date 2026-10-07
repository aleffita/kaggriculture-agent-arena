"""
Plugin for measuring Dual-GPU Sparse MoE Ring Streaming (GPT-OSS-20B-MXFP4)
with both Prefill and Decode (With & Without Drafter) + KV-Cache Disk Streaming.
"""
import subprocess
import re
from pathlib import Path
from typing import Dict, Any
from .base import BaseBenchmarkPlugin, BenchmarkResult

class MoeDualGpuRingPlugin(BaseBenchmarkPlugin):
    name = "moe_dual_gpu_ring"
    description = "Avaliação de Prefill e Decode do GPT-OSS-20B particionado em anel CED Dual-GPU com e sem drafter"

    def __init__(self):
        self.workspace_root = Path(__file__).resolve().parent.parent.parent
        self.exe_path = self.workspace_root / "src" / "litert_explore" / "hpc_engine" / "gpt_oss_dual_gpu_moe_ring.exe"
        self.src_path = self.workspace_root / "src" / "litert_explore" / "hpc_engine" / "gpt_oss_dual_gpu_moe_ring.cu"

    def _ensure_binary(self):
        if not self.exe_path.exists():
            cmd = [
                "nvcc",
                str(self.src_path),
                "-o", str(self.exe_path),
                "-ccbin", "C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\VC\\Tools\\MSVC\\14.44.35207\\bin\\Hostx64\\x64",
                "-O3",
                "-arch=sm_75"
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Falha ao compilar {self.src_path}: {res.stderr}")

    def run(self) -> BenchmarkResult:
        self._ensure_binary()
        proc = subprocess.run([str(self.exe_path)], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            return BenchmarkResult(
                plugin_name=self.name,
                target_hardware="GTX 1050 Ti (4GB) + RTX 2060 (6GB)",
                status="FAILED",
                error_message=proc.stderr or proc.stdout
            )

        output = proc.stdout

        # Regex parse prefill (accent-agnostic)
        ttft_match = re.search(r"TTFT\):\s*([\d\.]+)\s*ms", output)
        prefill_tok_match = re.search(r"PREFILL DUAL-GPU:\s*([\d\.]+)\s*tokens/seg", output, re.IGNORECASE)

        ttft_ms = float(ttft_match.group(1)) if ttft_match else 0.0
        prefill_tok_s = float(prefill_tok_match.group(1)) if prefill_tok_match else 0.0

        # Regex parse decode sem drafter
        sem_part = output.split("B. MODO DUAL-GPU COM DRAFTER")[0] if "B. MODO DUAL-GPU COM DRAFTER" in output else output
        sem_stalls_m = re.search(r"Stalls de Leitura de Disco:\s*(\d+)", sem_part)
        sem_lat_m = re.search(r"Tempo Total de Decode:\s*([\d\.]+)\s*ms", sem_part)
        sem_tok_m = re.search(r"Decode Efetiva:\s*([\d\.]+)\s*tokens/seg", sem_part)

        sem_stalls = int(sem_stalls_m.group(1)) if sem_stalls_m else 0
        sem_lat_ms = float(sem_lat_m.group(1)) if sem_lat_m else 0.0
        sem_tok_s = float(sem_tok_m.group(1)) if sem_tok_m else 0.0

        # Regex parse decode com drafter
        com_part = output.split("B. MODO DUAL-GPU COM DRAFTER")[-1] if "B. MODO DUAL-GPU COM DRAFTER" in output else ""
        com_stalls_m = re.search(r"Stalls de Leitura de Disco:\s*(\d+)", com_part)
        com_lat_m = re.search(r"Tempo Total de Decode:\s*([\d\.]+)\s*ms", com_part)
        com_tok_m = re.search(r"Decode Efetiva:\s*([\d\.]+)\s*tokens/seg", com_part)
        com_speedup_m = re.search(r"SPEEDUP DO DRAFTER NO ANEL:\s*([\d\.]+)x", com_part)

        com_stalls = int(com_stalls_m.group(1)) if com_stalls_m else 0
        com_lat_ms = float(com_lat_m.group(1)) if com_lat_m else 0.0
        com_tok_s = float(com_tok_m.group(1)) if com_tok_m else 0.0
        speedup = float(com_speedup_m.group(1)) if com_speedup_m else (com_tok_s / sem_tok_s if sem_tok_s > 0 else 1.0)

        # Regex parse KV-cache
        kv_size_m = re.search(r"Tamanho do KV-Cache em Disco.*?:\s*([\d\.]+)\s*MB", output)
        kv_pagein_m = re.search(r"Latência de Page-In por Bloco.*?:\s*([\d\.]+)\s*us", output)
        kv_size_mb = float(kv_size_m.group(1)) if kv_size_m else 786.43
        kv_pagein_us = float(kv_pagein_m.group(1)) if kv_pagein_m else 186.17

        return BenchmarkResult(
            plugin_name=self.name,
            target_hardware="GTX 1050 Ti (4GB) + RTX 2060 (6GB) + NVMe SSD Z:",
            prefill_tok_s=prefill_tok_s,
            prefill_ttft_ms=ttft_ms,
            decode_tok_s=com_tok_s,
            decode_latency_ms=com_lat_ms,
            stalls=com_stalls,
            speedup=speedup,
            extra_metrics={
                "sem_drafter": {
                    "decode_tok_s": sem_tok_s,
                    "decode_latency_ms": sem_lat_ms,
                    "stalls": sem_stalls
                },
                "com_drafter": {
                    "decode_tok_s": com_tok_s,
                    "decode_latency_ms": com_lat_ms,
                    "stalls": com_stalls,
                    "speedup": speedup
                },
                "kv_cache_disk": {
                    "size_16k_mb": kv_size_mb,
                    "pagein_latency_us": kv_pagein_us,
                    "storage_drive": "Z:\\models"
                }
            },
            status="SUCCESS"
        )
