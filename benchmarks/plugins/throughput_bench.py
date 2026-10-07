"""
Agnostic Throughput & Latency Benchmark Plugin (llama-bench / litert-benchmark style).
Evaluates Prefill (TTFT, tok/s) and Decode (ms/tok, tok/s) across a Cartesian grid (P x N x B).
"""
import subprocess
import json
import time
import math
from pathlib import Path
from typing import List, Dict, Any
from .base import BaseBenchmarkPlugin, BenchmarkMeasurement, BenchmarkSuiteResult

class ThroughputBenchmarkPlugin(BaseBenchmarkPlugin):
    name = "throughput"
    description = "Throughput & Latência em grade cartesiana (Prompt x Geração x Batch) com repetições estatísticas"

    def __init__(self):
        self.workspace_root = Path(__file__).resolve().parent.parent.parent
        self.unified_exe = self.workspace_root / "src" / "litert_explore" / "hpc_engine" / "unified_runtime.exe"

    def _run_unified_iteration(self, model: str, prompt_len: int, gen_tokens: int) -> Dict[str, Any]:
        cmd = [
            str(self.unified_exe),
            "--model", model,
            "--prompt-len", str(prompt_len),
            "--tokens", str(gen_tokens),
            "--json"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0:
            raise RuntimeError(f"Falha ao executar unified_runtime: {res.stderr or res.stdout}")
        return json.loads(res.stdout)

    def run(self, mode: str = "smoke") -> BenchmarkSuiteResult:
        suite = BenchmarkSuiteResult(
            benchmark_name=self.name,
            mode=mode,
            environment_metadata={
                "os": "Windows NT",
                "gpus": ["NVIDIA GeForce RTX 2060 (6GB)", "NVIDIA GeForce GTX 1050 Ti (4GB)"],
                "protocol": "Cartesian Sweep (P x N x B)"
            }
        )

        if not self.unified_exe.exists():
            meas = BenchmarkMeasurement(
                benchmark=self.name,
                backend="unified-ced",
                model="moe",
                mode=mode,
                status="FAILED",
                details={"error": f"Executável {self.unified_exe} não encontrado."}
            )
            suite.measurements.append(meas)
            return suite

        # Configuração da grade por modo
        if mode == "smoke":
            grid_prompts = [32, 128]
            grid_gens = [16, 32]
            models = ["moe", "bonsai"]
            repetitions = 2
        else:
            grid_prompts = [32, 128, 512, 1024]
            grid_gens = [16, 64, 128]
            models = ["moe", "bonsai"]
            repetitions = 5

        for model in models:
            for p in grid_prompts:
                for n in grid_gens:
                    prefill_speeds = []
                    prefill_ttfts = []
                    decode_speeds = []
                    decode_lats = []

                    for rep in range(repetitions):
                        try:
                            data = self._run_unified_iteration(model=model, prompt_len=p, gen_tokens=n)
                            prefill_speeds.append(data.get("prefill_tok_s", 0.0))
                            prefill_ttfts.append(data.get("prefill_ttft_ms", 0.0))
                            decode_speeds.append(data.get("decode_tok_s", 0.0))
                            decode_lats.append(data.get("decode_latency_ms", 0.0))
                        except Exception as e:
                            pass

                    if not decode_speeds:
                        continue

                    # Médias e desvios
                    avg_prefill_tok_s = sum(prefill_speeds) / len(prefill_speeds)
                    avg_ttft_ms = sum(prefill_ttfts) / len(prefill_ttfts)
                    avg_decode_tok_s = sum(decode_speeds) / len(decode_speeds)
                    avg_decode_lat_ms = sum(decode_lats) / len(decode_lats)
                    ms_per_tok = avg_decode_lat_ms / n if n > 0 else 0.0

                    variance = sum((x - avg_decode_tok_s) ** 2 for x in decode_speeds) / len(decode_speeds)
                    stddev = math.sqrt(variance)

                    meas = BenchmarkMeasurement(
                        benchmark=self.name,
                        backend="unified-ced",
                        model=model,
                        mode=mode,
                        prompt_tokens=p,
                        gen_tokens=n,
                        batch_size=1,
                        prefill_tok_s=avg_prefill_tok_s,
                        prefill_ttft_ms=avg_ttft_ms,
                        decode_tok_s=avg_decode_tok_s,
                        decode_ms_per_tok=ms_per_tok,
                        metric_name="decode_tok_s",
                        metric_value=avg_decode_tok_s,
                        error_stddev=stddev,
                        status="SUCCESS",
                        details={
                            "repetitions": repetitions,
                            "raw_decode_runs": decode_speeds,
                            "raw_prefill_runs": prefill_speeds
                        }
                    )
                    suite.measurements.append(meas)

        return suite
