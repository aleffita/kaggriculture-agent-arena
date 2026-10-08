"""
Unified CED Live Conversational Package.
Full-Duplex Speech & Multimodal Live Streaming Harness.
"""
from .audio_engine import AudioStreamEngine
from .session import LiveSession
from .realtime_server import RealtimeBridgeServer
from .subagents import SubagentPool, SubagentSession
from .web_studio import create_studio_app, main as web_studio_main
from .cli_live import main as live_cli_main

__all__ = [
    "AudioStreamEngine",
    "LiveSession",
    "RealtimeBridgeServer",
    "SubagentPool",
    "SubagentSession",
    "create_studio_app",
    "web_studio_main",
    "live_cli_main",
]
