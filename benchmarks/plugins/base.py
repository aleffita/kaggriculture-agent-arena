"""
Base class and dataclasses for modular HPC benchmark plugins.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional

@dataclass
class BenchmarkResult:
    plugin_name: str
    target_hardware: str
    prefill_tok_s: float = 0.0
    prefill_ttft_ms: float = 0.0
    decode_tok_s: float = 0.0
    decode_latency_ms: float = 0.0
    stalls: int = 0
    speedup: float = 1.0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "SUCCESS"
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_csv_row(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "plugin": self.plugin_name,
            "hardware": self.target_hardware,
            "prefill_tok_s": round(self.prefill_tok_s, 2),
            "prefill_ttft_ms": round(self.prefill_ttft_ms, 2),
            "decode_tok_s": round(self.decode_tok_s, 2),
            "decode_latency_ms": round(self.decode_latency_ms, 2),
            "stalls": self.stalls,
            "speedup": round(self.speedup, 2),
            "status": self.status
        }

class BaseBenchmarkPlugin:
    name: str = "base_plugin"
    description: str = "Base benchmark plugin"

    def run(self) -> BenchmarkResult:
        raise NotImplementedError("Subclasses must implement run()")
