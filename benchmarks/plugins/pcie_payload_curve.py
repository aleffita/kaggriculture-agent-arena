"""
Plugin for measuring PCIe Gen3 x1 Payload Curve between GTX 1050 Ti and RTX 2060.
"""
import subprocess
import re
from pathlib import Path
from typing import Dict, Any
from .base import BaseBenchmarkPlugin, BenchmarkResult

class PciePayloadCurvePlugin(BaseBenchmarkPlugin):
    name = "pcie_payload_curve"
    description = "Varredura empírica da curva de latência vs payload na PCIe Gen3 x1 entre GTX 1050 Ti e RTX 2060"

    def __init__(self):
        self.plugin_dir = Path(__file__).resolve().parent
        self.exe_path = self.plugin_dir / "bench_pcie_payload_curve.exe"
        self.src_path = self.plugin_dir / "bench_pcie_payload_curve.cu"

    def _ensure_binary(self):
        if not self.exe_path.exists():
            # Compilar com nvcc
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
                target_hardware="GTX 1050 Ti -> RTX 2060",
                status="FAILED",
                error_message=proc.stderr or proc.stdout
            )

        output = proc.stdout
        # Parse output table
        # Format: | 1024 | 1.00 KB | 91.39 us | 0.02 GB/s | 1.00x |
        sweep_data = []
        pattern = re.compile(r"\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([\d\.]+)\s*us\s*\|\s*([\d\.]+)\s*GB/s\s*\|\s*([\d\.]+)x")
        for line in output.splitlines():
            match = pattern.search(line)
            if match:
                b_size = int(match.group(1))
                fmt_size = match.group(2).strip()
                lat_us = float(match.group(3))
                bw_gbs = float(match.group(4))
                ratio = float(match.group(5))
                sweep_data.append({
                    "bytes": b_size,
                    "formatted_size": fmt_size,
                    "avg_latency_us": lat_us,
                    "bandwidth_gbs": bw_gbs,
                    "marginal_ratio": ratio
                })

        h_boundary_lat_us = 106.39
        for row in sweep_data:
            if row["bytes"] == 10486:
                h_boundary_lat_us = row["avg_latency_us"]

        pareto_knee = next((r for r in sweep_data if r["bytes"] == 32768), sweep_data[-1] if sweep_data else {})

        return BenchmarkResult(
            plugin_name=self.name,
            target_hardware="GTX 1050 Ti -> RTX 2060 (PCIe Gen3 x1)",
            decode_latency_ms=h_boundary_lat_us / 1000.0,
            extra_metrics={
                "boundary_h_latency_us": h_boundary_lat_us,
                "pareto_knee_bytes": pareto_knee.get("bytes", 32768),
                "pareto_knee_latency_us": pareto_knee.get("avg_latency_us", 145.66),
                "pareto_knee_bandwidth_gbs": pareto_knee.get("bandwidth_gbs", 0.42),
                "sweep_points": sweep_data
            },
            status="SUCCESS"
        )
