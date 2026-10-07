"""
Base classes and schema for independent LLM Benchmarks & Evals.
Supports both performance grid sweeps (llama-bench style) and quality evals (HumanEval, Perplexity).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

@dataclass
class BenchmarkMeasurement:
    benchmark: str
    backend: str
    model: str
    mode: str  # "smoke" or "full"
    prompt_tokens: int = 0
    gen_tokens: int = 0
    batch_size: int = 1
    prefill_tok_s: float = 0.0
    prefill_ttft_ms: float = 0.0
    decode_tok_s: float = 0.0
    decode_ms_per_tok: float = 0.0
    metric_name: str = "decode_tok_s"
    metric_value: float = 0.0
    error_stddev: float = 0.0
    status: str = "SUCCESS"
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "benchmark": self.benchmark,
            "backend": self.backend,
            "model": self.model,
            "mode": self.mode,
            "prompt_tokens": self.prompt_tokens,
            "gen_tokens": self.gen_tokens,
            "batch_size": self.batch_size,
            "prefill_tok_s": round(self.prefill_tok_s, 2),
            "prefill_ttft_ms": round(self.prefill_ttft_ms, 2),
            "decode_tok_s": round(self.decode_tok_s, 2),
            "decode_ms_per_tok": round(self.decode_ms_per_tok, 4),
            "metric_name": self.metric_name,
            "metric_value": round(self.metric_value, 4),
            "error_stddev": round(self.error_stddev, 4),
            "status": self.status,
        }

@dataclass
class BenchmarkSuiteResult:
    benchmark_name: str
    mode: str
    measurements: List[BenchmarkMeasurement] = field(default_factory=list)
    environment_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_csv_rows(self) -> List[Dict[str, Any]]:
        return [m.to_csv_row() for m in self.measurements]

class BaseBenchmarkPlugin:
    name: str = "base_benchmark"
    description: str = "Base agnostic benchmark plugin"

    def run(self, mode: str = "smoke") -> BenchmarkSuiteResult:
        raise NotImplementedError("Subclasses must implement run(mode)")
