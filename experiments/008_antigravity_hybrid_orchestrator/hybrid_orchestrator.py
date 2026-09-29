"""Antigravity Hybrid Orchestrator — Cloud Architect & On-Device Sentinel Swarm.

Pairs Cloud Architectural Contracts (AST skeleton synthesis) with an on-device
Gemma 4 workforce on the NVIDIA GeForce GTX 1050 Ti (GPU 1) via LiteRT-LM and
the Google Antigravity SDK.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import pathlib
import sys
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Ensure repo src/ is importable
repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from litert_explore.gpu import select_gpu
from litert_explore.engine import resolve_model_path, LiteRtModelRunner

try:
    from .slingshot_stream import SlingshotTelemetryStream, TelemetryEvent
except ImportError:
    from slingshot_stream import SlingshotTelemetryStream, TelemetryEvent



def extract_public_skeleton(source_or_path: str) -> str:
    """Extracts module imports, global variable names, and function signatures via AST.
    
    Guarantees 0 bytes of function bodies, SQL queries, or private secrets leave
    the local environment.
    """
    if os.path.exists(source_or_path):
        with open(source_or_path, "r", encoding="utf-8") as f:
            code = f.read()
    else:
        code = source_or_path

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "imports=[] globals=[] defs=[]"

    imports: List[str] = []
    globals_list: List[str] = []
    defs: List[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    globals_list.append(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            globals_list.append(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [a.arg for a in node.args.args]
            prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
            defs.append(f"{prefix}{node.name}({', '.join(args)})")
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            defs.append(f"class {node.name}[{', '.join(methods)}]")

    return (
        f"imports=[{', '.join(imports)}] "
        f"globals=[{', '.join(globals_list)}] "
        f"defs=[{', '.join(defs)}]"
    )


@dataclass
class CloudArchitectContract:
    """Structured architectural contract emitted by Cloud Architect (e.g. Gemini 3.8 Flash)."""
    target_module: str
    role: str
    strategy: str
    invariants: List[str]
    allowed_tools: List[str]


def synthesize_cloud_contract(module_name: str, ast_skeleton: str) -> CloudArchitectContract:
    """Emulates Cloud Architect (Gemini 3.8 Flash) generating an architectural contract
    from zero-body AST signatures before switching the cloud session to IDLE.
    """
    return CloudArchitectContract(
        target_module=module_name,
        role="Bursty Bandwidth Sentinel & Rate-Limiter Controller",
        strategy=(
            "Inspect non-uniform telemetry events (values 0-100). "
            "When value >= 80 or burst acceleration occurs ('estilingue'), "
            "trigger throttle elevation and log an anomaly event. "
            "Maintain minimal host latency."
        ),
        invariants=[
            "Preserve telemetry processing function signatures.",
            "Do not exceed 4096 tokens of total context.",
            "Enforce pure text mode; zero multimedia encoder allocation.",
            "All local computation strictly pinned to GPU 1 (GTX 1050 Ti).",
        ],
        allowed_tools=["adjust_bandwidth_throttle", "record_telemetry_anomaly"],
    )


class LocalGemmaSentinel:
    """On-device workforce running Gemma-4-E2B on NVIDIA GeForce GTX 1050 Ti."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = resolve_model_path(model_path)
        self.gpu_target = 1  # Always GTX 1050 Ti
        self.throttle_level = 0
        self.anomaly_log: List[Dict[str, Any]] = []
        self._runner: Optional[LiteRtModelRunner] = None

    def initialize_on_1050ti(self):
        """Initializes LiteRT-LM runner on the GTX 1050 Ti in pure text mode."""
        self._runner = LiteRtModelRunner(
            model_path=self.model_path,
            backend="gpu",
            gpu_target="1050ti",
            max_num_tokens=4096,
        )

    # Local tools
    def adjust_bandwidth_throttle(self, level: int) -> str:
        """Adjusts the queue rate limiter (0 = nominal, 1 = moderate, 2 = aggressive)."""
        self.throttle_level = max(0, min(2, level))
        return f"Throttle adjusted to level {self.throttle_level}"

    def record_telemetry_anomaly(self, metric: int, reason: str) -> str:
        """Records a detected burst anomaly in the local audit register."""
        entry = {
            "timestamp": time.time(),
            "metric": metric,
            "reason": reason,
            "throttle_active": self.throttle_level,
        }
        self.anomaly_log.append(entry)
        return f"Logged anomaly for metric={metric}: {reason}"

    def evaluate_telemetry_turn(
        self,
        event: TelemetryEvent,
        contract: CloudArchitectContract,
    ) -> Dict[str, Any]:
        """Evaluates a telemetry event locally on the GTX 1050 Ti against the contract."""
        prompt = textwrap.dedent(f"""\
            <start_of_turn>user
            [ARCHITECTURAL INVARIANT CONTRACT]
            Strategy: {contract.strategy}
            Role: {contract.role}

            [TELEMETRY EVENT]
            Metric Value: {event.value}/100
            Event Label: {event.label}
            Is Burst Phase: {event.is_burst}

            Analyze this event. If metric >= 80 or burst spike, determine if throttle level (0, 1, 2)
            needs adjusting and return a compact JSON response with keys:
            "action": ("NOMINAL" | "THROTTLE_APPLIED"), "recommended_throttle": (0 | 1 | 2), "summary": (brief 1-sentence explanation).
            Return ONLY valid JSON.
            <end_of_turn>
            <start_of_turn>model
        """)

        if self._runner is not None:
            raw_response = self._runner.generate(prompt)
        else:
            # Fallback heuristic if runner not yet initialized
            raw_response = json.dumps({
                "action": "THROTTLE_APPLIED" if event.value >= 80 else "NOMINAL",
                "recommended_throttle": 2 if event.value >= 90 else (1 if event.value >= 80 else 0),
                "summary": f"Observed metric {event.value} in {event.label} state."
            })

        # Parse model output
        try:
            if not isinstance(raw_response, str):
                raw_response = str(raw_response)
            cleaned = raw_response.strip()
            if "```json" in cleaned:
                cleaned = cleaned.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```")[1].split("```")[0].strip()
            decision = json.loads(cleaned)
        except Exception:
            decision = {
                "action": "THROTTLE_APPLIED" if event.value >= 80 else "NOMINAL",
                "recommended_throttle": 2 if event.value >= 90 else (1 if event.value >= 80 else 0),
                "summary": f"Fallback: Parsed metric {event.value}",
                "raw": str(raw_response)[:100],
            }


        # Execute on-device tool intervention
        rec_throttle = decision.get("recommended_throttle", 0)
        if rec_throttle != self.throttle_level:
            self.adjust_bandwidth_throttle(rec_throttle)

        if event.value >= 80:
            self.record_telemetry_anomaly(event.value, decision.get("summary", "Spike detected"))

        return {
            "event": event,
            "decision": decision,
            "active_throttle": self.throttle_level,
        }
