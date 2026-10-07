"""
Agnostic Throughput & Latency Benchmark Plugin (llama-bench / litert-benchmark style).
Evaluates Prefill (TTFT, tok/s) and Decode (ms/tok, tok/s) across a Cartesian grid (P x N x B).
Compares the Unified Heterogeneous Engine (unified-ced) against Original Runtimes:
- gpt-oss-20b: unified-ced vs original (llama.cpp stock)
- bonsai-27b:  unified-ced vs original (bypassed: OOM >6GB VRAM on single GPU stock)
- gemma-4-E2B-it: unified-ced vs original (litert-d3d12 Dawn Direct3D 12 stock)
"""
import subprocess
import json
import time
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from .base import BaseBenchmarkPlugin, BenchmarkMeasurement, BenchmarkSuiteResult

class ThroughputBenchmarkPlugin(BaseBenchmarkPlugin):
    name = "throughput"
    description = "Throughput & Latência em grade cartesiana comparando Unified-CED vs Runtimes Originais"

    def __init__(self):
        self.workspace_root = Path(__file__).resolve().parent.parent.parent
        self.unified_exe = self.workspace_root / "src" / "litert_explore" / "hpc_engine" / "unified_runtime.exe"
        self.llama_cli_exe = Path("Z:/workspaces/llama-cpp-prism/build/bin/Release/llama-cli.exe")
        self.gpt_oss_path = Path("Z:/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf")

    def _run_unified_iteration(self, model_arg: str, prompt_len: int, gen_tokens: int) -> Dict[str, Any]:
        cmd = [
            str(self.unified_exe),
            "--model", model_arg,
            "--prompt-len", str(prompt_len),
            "--tokens", str(gen_tokens),
            "--json"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0:
            raise RuntimeError(f"Falha ao executar unified_runtime: {res.stderr or res.stdout}")
        return json.loads(res.stdout)

    def _run_original_gpt_oss_20b(self, prompt_len: int, gen_tokens: int) -> Optional[Dict[str, float]]:
        """Executa stock llama.cpp em CPU para gpt-oss-20b, evitando estouro de VRAM."""
        if not self.llama_cli_exe.exists() or not self.gpt_oss_path.exists():
            return None
        prompt_text = "In modern computer systems, heterogeneous architectures utilize specialized accelerators. "
        cmd = [
            str(self.llama_cli_exe),
            "-m", str(self.gpt_oss_path),
            "-p", prompt_text,
            "-n", str(gen_tokens),
            "-c", str(max(256, prompt_len + gen_tokens)),
            "-ngl", "0",
            "--simple-io",
            "--single-turn"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
            output = res.stdout + res.stderr
            match = re.search(r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]", output)
            if match:
                prefill_tok_s = float(match.group(1))
                decode_tok_s = float(match.group(2))
                ttft_ms = (prompt_len / prefill_tok_s * 1000.0) if prefill_tok_s > 0 else 0.0
                ms_per_tok = (1000.0 / decode_tok_s) if decode_tok_s > 0 else 0.0
                return {
                    "prefill_tok_s": prefill_tok_s,
                    "prefill_ttft_ms": ttft_ms,
                    "decode_tok_s": decode_tok_s,
                    "ms_per_tok": ms_per_tok,
                }
        except Exception:
            pass
        return None

    def _run_original_gemma4(self, prompt_len: int, gen_tokens: int) -> Optional[Dict[str, float]]:
        """Executa stock LiteRT via Google Dawn Direct3D 12 na RTX 2060."""
        try:
            from litert_explore.benchmark import run_benchmark
            res = run_benchmark(target="2060", prefill_tokens=prompt_len, decode_tokens=gen_tokens)
            ttft_ms = res.time_to_first_token * 1000.0
            ms_per_tok = (1000.0 / res.decode_speed) if res.decode_speed > 0 else 0.0
            return {
                "prefill_tok_s": res.prefill_speed,
                "prefill_ttft_ms": ttft_ms,
                "decode_tok_s": res.decode_speed,
                "ms_per_tok": ms_per_tok,
            }
        except Exception:
            return None

    def run(self, mode: str = "smoke") -> BenchmarkSuiteResult:
        suite = BenchmarkSuiteResult(
            benchmark_name=self.name,
            mode=mode,
            environment_metadata={
                "os": "Windows NT",
                "gpus": ["NVIDIA GeForce RTX 2060 (6GB)", "NVIDIA GeForce GTX 1050 Ti (4GB)"],
                "protocol": "Model Triad Comparative Sweep (unified-ced vs original)",
                "models": ["gpt-oss-20b", "bonsai-27b", "gemma-4-E2B-it"]
            }
        )

        if not self.unified_exe.exists():
            meas = BenchmarkMeasurement(
                benchmark=self.name,
                backend="unified-ced",
                model="gpt-oss-20b",
                mode=mode,
                status="FAILED",
                details={"error": f"Executavel {self.unified_exe} nao encontrado."}
            )
            suite.measurements.append(meas)
            return suite

        # Configuração da grade
        if mode == "smoke":
            grid_prompts = [128]
            grid_gens = [16]
            repetitions = 2
        else:
            grid_prompts = [32, 128, 512, 1024]
            grid_gens = [16, 64, 128]
            repetitions = 5

        # Tríade de modelos obrigatória
        model_configs = [
            {"name": "gpt-oss-20b", "arg": "moe", "type": "moe"},
            {"name": "bonsai-27b", "arg": "bonsai", "type": "ternary"},
            {"name": "gemma-4-E2B-it", "arg": "gemma4", "type": "dense"}
        ]

        for cfg in model_configs:
            m_name = cfg["name"]
            m_arg = cfg["arg"]

            for p in grid_prompts:
                for n in grid_gens:
                    # 1. Avaliação no motor UNIFIED-CED
                    prefill_speeds = []
                    prefill_ttfts = []
                    decode_speeds = []
                    decode_lats = []

                    for rep in range(repetitions):
                        try:
                            data = self._run_unified_iteration(model_arg=m_arg, prompt_len=p, gen_tokens=n)
                            prefill_speeds.append(data.get("prefill_tok_s", 0.0))
                            prefill_ttfts.append(data.get("prefill_ttft_ms", 0.0))
                            decode_speeds.append(data.get("decode_tok_s", 0.0))
                            decode_lats.append(data.get("decode_latency_ms", 0.0))
                        except Exception as e:
                            pass

                    if decode_speeds:
                        avg_prefill_tok_s = sum(prefill_speeds) / len(prefill_speeds)
                        avg_ttft_ms = sum(prefill_ttfts) / len(prefill_ttfts)
                        avg_decode_tok_s = sum(decode_speeds) / len(decode_speeds)
                        avg_decode_lat_ms = sum(decode_lats) / len(decode_lats)
                        ms_per_tok = avg_decode_lat_ms / n if n > 0 else 0.0
                        variance = sum((x - avg_decode_tok_s) ** 2 for x in decode_speeds) / len(decode_speeds)
                        stddev = math.sqrt(variance)

                        meas_unified = BenchmarkMeasurement(
                            benchmark=self.name,
                            backend="unified-ced",
                            model=m_name,
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
                        suite.measurements.append(meas_unified)

                    # 2. Avaliação no RUNTIME ORIGINAL
                    if m_name == "bonsai-27b":
                        # Diretiva estrita: original dá OOM (>6GB VRAM) e crasha o processo host
                        meas_orig = BenchmarkMeasurement(
                            benchmark=self.name,
                            backend="original (llama.cpp)",
                            model=m_name,
                            mode=mode,
                            prompt_tokens=p,
                            gen_tokens=n,
                            batch_size=1,
                            prefill_tok_s=0.0,
                            prefill_ttft_ms=0.0,
                            decode_tok_s=0.0,
                            decode_ms_per_tok=0.0,
                            metric_name="decode_tok_s",
                            metric_value=0.0,
                            error_stddev=0.0,
                            status="SKIPPED_OOM",
                            details={"reason": "OOM (>6GB VRAM) em GPU unica no runtime original; ignorado para preservar integridade do host"}
                        )
                        suite.measurements.append(meas_orig)

                    elif m_name == "gpt-oss-20b":
                        orig_data = self._run_original_gpt_oss_20b(prompt_len=p, gen_tokens=n)
                        if orig_data:
                            meas_orig = BenchmarkMeasurement(
                                benchmark=self.name,
                                backend="original (llama.cpp)",
                                model=m_name,
                                mode=mode,
                                prompt_tokens=p,
                                gen_tokens=n,
                                batch_size=1,
                                prefill_tok_s=orig_data["prefill_tok_s"],
                                prefill_ttft_ms=orig_data["prefill_ttft_ms"],
                                decode_tok_s=orig_data["decode_tok_s"],
                                decode_ms_per_tok=orig_data["ms_per_tok"],
                                metric_name="decode_tok_s",
                                metric_value=orig_data["decode_tok_s"],
                                error_stddev=0.0,
                                status="SUCCESS",
                                details={"note": "Executado via CPU stock devido ao tamanho de 11.28 GB exceder VRAM isolada"}
                            )
                        else:
                            meas_orig = BenchmarkMeasurement(
                                benchmark=self.name,
                                backend="original (llama.cpp)",
                                model=m_name,
                                mode=mode,
                                prompt_tokens=p,
                                gen_tokens=n,
                                batch_size=1,
                                status="FAILED",
                                details={"error": "Falha ao obter telemetria do llama-cli original"}
                            )
                        suite.measurements.append(meas_orig)

                    elif m_name == "gemma-4-E2B-it":
                        orig_data = self._run_original_gemma4(prompt_len=p, gen_tokens=n)
                        if orig_data:
                            meas_orig = BenchmarkMeasurement(
                                benchmark=self.name,
                                backend="original (litert-d3d12)",
                                model=m_name,
                                mode=mode,
                                prompt_tokens=p,
                                gen_tokens=n,
                                batch_size=1,
                                prefill_tok_s=orig_data["prefill_tok_s"],
                                prefill_ttft_ms=orig_data["prefill_ttft_ms"],
                                decode_tok_s=orig_data["decode_tok_s"],
                                decode_ms_per_tok=orig_data["ms_per_tok"],
                                metric_name="decode_tok_s",
                                metric_value=orig_data["decode_tok_s"],
                                error_stddev=0.0,
                                status="SUCCESS",
                                details={"backend": "Google LiteRT Dawn Direct3D 12 (RTX 2060 stock)"}
                            )
                        else:
                            meas_orig = BenchmarkMeasurement(
                                benchmark=self.name,
                                backend="original (litert-d3d12)",
                                model=m_name,
                                mode=mode,
                                prompt_tokens=p,
                                gen_tokens=n,
                                batch_size=1,
                                status="FAILED",
                                details={"error": "Falha no runner oficial LiteRT"}
                            )
                        suite.measurements.append(meas_orig)

        return suite
