"""Local OpenAI-Compatible Server for Google Antigravity SDK on GTX 1050 Ti."""

import os
import sys
import threading
import pathlib
from typing import Optional

# Ensure repo src/ is importable
repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from litert_explore.gpu import select_gpu
from litert_explore.engine import resolve_model_path
from google.antigravity.connections.local.litert_server import LiteRTOpenAIServer, LiteRTOpenAIHandler
import litert_lm
from litert_lm import interfaces
from rich.console import Console

console = Console()

def run_local_server(port: int = 9379, model_path: Optional[str] = None):
    console.print(f"[bold cyan]=== Antigravity Local Server (Target: GTX 1050 Ti, Port: {port}) ===[/bold cyan]\n")
    path = resolve_model_path(model_path)
    model_name = pathlib.Path(path).name

    with select_gpu("1050ti"):
        console.print(f"[yellow]Loading model on GTX 1050 Ti:[/yellow] {model_name}...")
        engine = litert_lm.Engine(
            model_path=path,
            backend=interfaces.GPU(),
            max_num_tokens=4096,
        )

        server_address = ("127.0.0.1", port)
        server = LiteRTOpenAIServer(
            server_address,
            LiteRTOpenAIHandler,
            engine=engine,
            model_name=model_name,
            max_output_tokens=2048,
            thinking_token_budget=None,
        )

        console.print(f"[bold green]Server is running and listening on http://127.0.0.1:{port}/[/bold green]")
        console.print("[dim]Endpoint: POST http://127.0.0.1:9379/v1/chat/completions (OpenAI Compatible)[/dim]")
        console.print("[dim]Press Ctrl+C to terminate server.[/dim]\n")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[yellow]Shutting down local server...[/yellow]")
            server.shutdown()
            server.server_close()
            console.print("[green]Server stopped cleanly.[/green]")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Antigravity Local LiteRT Server")
    parser.add_argument("--port", type=int, default=9379, help="Port to bind server to")
    parser.add_argument("--model", type=str, default=None, help="Path to .litertlm model")
    args = parser.parse_args()

    run_local_server(port=args.port, model_path=args.model)
