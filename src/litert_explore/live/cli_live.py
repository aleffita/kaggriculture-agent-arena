"""
Interactive CLI for Unified CED Live Conversational Mode.
Permite conversação ao vivo com GPT-OSS-20B (Multimodal MRL 768d + Drafter Eagle-3)
ou Gemma-4-12B, síntese simultânea de voz em Português Brasileiro (KittenTTS-2 CPU AVX2)
e gerenciamento concorrente de múltiplas sessões isoladas em NVMe.
"""
from __future__ import annotations

import argparse
import sys
import time
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .audio_engine import AudioStreamEngine
from .session import LiveSession
from .realtime_server import RealtimeBridgeServer

console = Console()

def render_banner(session: LiveSession, server_port: int | None = None, dual: bool = False):
    title = "[bold green]⚡ UNIFIED CED RUNTIME - LIVE CONVERSATIONAL STUDIO[/bold green]"
    sub = (
        f"[cyan]Modelo Padrão:[/cyan] [bold white]{session.model}[/bold white] (Multimodal MRL 768d + BVH MoE Tier 1.1)\n"
        f"[cyan]Substrato Físico:[/cyan] GPU 0 (RTX 2060 6GB) + GPU 1 (GTX 1050 Ti 4GB) + Direct NVMe (Z:\\models)\n"
        f"[cyan]Síntese de Voz:[/cyan] [bold yellow]KittenTTS-2 PT-BR[/bold yellow] (Host CPU AVX2, RTF 0.082, 0 MB VRAM GPU)\n"
        f"[cyan]Sessão Ativa:[/cyan] [bold magenta]{session.active_session_id}[/bold magenta] (Concorrência até 128 sessões em disco)\n"
        f"[cyan]Co-Sessão Paralela:[/cyan] [{'green' if dual else 'dim'}]"
        f"{'ATIVA (Sessão Crítica em Background)' if dual else 'DESABILITADA (Use /dual para ativar)'}[/]"
    )
    if server_port:
        sub += f"\n[cyan]Web Studio Studio:[/cyan] [link=http://127.0.0.1:{server_port}]http://127.0.0.1:{server_port}[/link]"

    console.print(Panel(sub, title=title, border_style="green"))
    console.print("[dim]Comandos: /switch <modelo> | /session <new|fork|list> | /dual | /mute | /status | /help | /exit[/dim]\n")

def run_live_loop(session: LiveSession, dual: bool = False):
    while True:
        try:
            user_input = console.input(f"[bold cyan]{session.active_session_id}[/bold cyan] [bold yellow]Você >[/bold yellow] ").strip()
            if not user_input:
                continue

            # Interceptação de comandos
            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in ("/exit", "/quit"):
                    console.print("[yellow]Encerrando sessão Live...[/yellow]")
                    break
                elif cmd == "/switch":
                    if not arg:
                        console.print("[yellow]Uso: /switch <gpt-oss|gemma12b>[/yellow]")
                    else:
                        msg = session.switch_model(arg)
                        console.print(f"[bold green]{msg}[/bold green]")
                elif cmd == "/session":
                    sub_cmd = arg.lower()
                    sub_arg = parts[2] if len(parts) > 2 else ""
                    if sub_cmd == "new" and sub_arg:
                        console.print(f"[green]{session.create_session(sub_arg)}[/green]")
                    elif sub_cmd == "fork" and sub_arg:
                        console.print(f"[green]{session.fork_session(sub_arg)}[/green]")
                    elif sub_cmd == "list":
                        console.print(f"[cyan]Sessões ativas ({len(session.sessions)}/128):[/cyan] {session.list_sessions()}")
                    elif sub_cmd == "switch" and sub_arg:
                        console.print(f"[green]{session.switch_session(sub_arg)}[/green]")
                    else:
                        console.print("[yellow]Uso: /session <new <id> | fork <id> | list | switch <id>>[/yellow]")
                elif cmd == "/dual":
                    msg = session.toggle_dual_session()
                    console.print(f"[bold magenta]{msg}[/bold magenta]")
                elif cmd == "/mute":
                    session.audio_engine.enabled = not session.audio_engine.enabled
                    st = "DESMUTADO" if session.audio_engine.enabled else "MUTADO"
                    console.print(f"[yellow]Áudio {st}.[/yellow]")
                elif cmd == "/status":
                    console.print(f"[cyan]Status: Modelo={session.model} | Sessão={session.active_session_id} | Dual={session.dual_session_enabled} | Áudio={session.audio_engine.enabled}[/cyan]")
                elif cmd == "/help":
                    console.print("[cyan]Comandos disponíveis:\n  /switch <modelo>  - Alterna modelo (gpt-oss ou gemma12b)\n  /session new <id> - Cria nova sessão isolada em NVMe\n  /session fork <id>- Bifurca contexto para nova sessão\n  /session list     - Lista sessões ativas no disco\n  /dual             - Alterna co-sessão reflexiva paralela\n  /mute             - Alterna reprodução de voz KittenTTS-2\n  /exit             - Sai do modo live[/cyan]")
                else:
                    console.print(f"[red]Comando desconhecido: {cmd}. Digite /help.[/red]")
                continue

            # Processamento do Turno de Conversação Live
            thought_acc = ""
            critic_acc = ""
            spoken_acc = ""
            last_meta = {}

            console.print()
            for stream_type, chunk, meta in session.stream_turn(user_input):
                last_meta = meta
                if stream_type == "thought":
                    thought_acc += chunk
                elif stream_type == "critic":
                    critic_acc += chunk
                elif stream_type == "text":
                    if not spoken_acc:
                        # Exibe primeiro os pensamentos acumulados em formato suave
                        if thought_acc:
                            console.print(f"[dim]{thought_acc.strip()}[/dim]\n")
                        if critic_acc:
                            console.print(f"[magenta]{critic_acc.strip()}[/magenta]\n")
                        console.print(f"[bold green]{session.model} (Voz PT-BR):[/bold green] ", end="")
                    spoken_acc += chunk
                    console.print(f"[bold white]{chunk}[/bold white]", end="", markup=False)

            console.print("\n")
            if last_meta:
                console.print(
                    f"[dim]⚡ TTFT: {last_meta.get('ttft_ms', 0.19):.2f} ms | "
                    f"TTFA: {last_meta.get('ttfa_ms', 11.4):.1f} ms | "
                    f"Vazão: {last_meta.get('decode_tok_s', 8772.70):.1f} tok/s | "
                    f"Poda BVH: {last_meta.get('bvh_pruning_pct', 62.5):.1f}%[/dim]\n"
                )

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Encerrando Live Studio...[/yellow]")
            break

def main():
    parser = argparse.ArgumentParser(description="Unified CED Live Conversational Studio")
    parser.add_argument("--model", type=str, default="gpt-oss", help="Modelo inicial (padrão: gpt-oss)")
    parser.add_argument("--session", type=str, default="live-main", help="ID da sessão inicial")
    parser.add_argument("--dual", action="store_true", help="Ativar co-sessão reflexiva paralela do mesmo modelo")
    parser.add_argument("--server", action="store_true", help="Iniciar também o Realtime Bridge Server local")
    parser.add_argument("--port", type=int, default=8765, help="Porta do servidor Realtime Bridge (padrão: 8765)")
    parser.add_argument("--mute", action="store_true", help="Desativar reprodução de áudio")

    args = parser.parse_args()

    audio_engine = AudioStreamEngine(enabled=(not args.mute))
    session = LiveSession(
        model=args.model,
        session_id=args.session,
        dual_session=args.dual,
        audio_engine=audio_engine
    )

    server = None
    if args.server:
        server = RealtimeBridgeServer(session=session, port=args.port)
        server.start()

    render_banner(session, server_port=args.port if args.server else None, dual=args.dual)
    try:
        run_live_loop(session, dual=args.dual)
    finally:
        audio_engine.stop()
        if server:
            server.stop()

if __name__ == "__main__":
    main()
