"""
Local Realtime Bridge Server for Unified CED Live Mode.
Fornece endpoint HTTP e WebSocket/SSE para frontends Web locais e clientes do ecossistema Realtime.
Inclui interface gráfica WebAudio HTML5/JS embutida em dark mode executivo.
"""
from __future__ import annotations

import base64
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

from .session import LiveSession

HTML_LIVE_CLIENT = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Unified CED - Live Conversational Studio</title>
    <style>
        :root {
            --bg-deep: #121212;
            --bg-card: #1A1A1A;
            --border: #333333;
            --text-primary: #E0E0E0;
            --text-dim: #888888;
            --accent: #4CAF50;
            --accent-thought: #5C9DF5;
            --accent-critic: #BA68C8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg-deep);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            padding: 14px 24px;
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header-title { font-size: 1.15rem; font-weight: bold; }
        .header-badge {
            font-size: 0.8rem;
            padding: 4px 10px;
            border-radius: 4px;
            background: #252525;
            color: var(--accent);
            border: 1px solid #3d3d3d;
        }
        #chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .msg {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 6px;
            line-height: 1.5;
            font-size: 0.95rem;
        }
        .msg.user {
            align-self: flex-end;
            background: #263238;
            border: 1px solid #37474F;
        }
        .msg.assistant {
            align-self: flex-start;
            background: var(--bg-card);
            border: 1px solid var(--border);
        }
        .thought-box {
            font-size: 0.82rem;
            color: var(--text-dim);
            background: #151515;
            border-left: 3px solid var(--accent-thought);
            padding: 6px 10px;
            margin-bottom: 8px;
            white-space: pre-wrap;
            font-family: monospace;
        }
        .critic-box {
            font-size: 0.82rem;
            color: var(--accent-critic);
            background: #171219;
            border-left: 3px solid var(--accent-critic);
            padding: 6px 10px;
            margin-bottom: 8px;
            font-family: monospace;
        }
        .telemetry-tag {
            font-size: 0.75rem;
            color: var(--text-dim);
            margin-top: 6px;
            font-family: monospace;
        }
        #input-bar {
            padding: 16px 24px;
            background: var(--bg-card);
            border-top: 1px solid var(--border);
            display: flex;
            gap: 12px;
        }
        #text-input {
            flex: 1;
            padding: 12px;
            background: #121212;
            border: 1px solid var(--border);
            color: #FFF;
            border-radius: 4px;
            font-size: 0.95rem;
        }
        #text-input:focus { outline: none; border-color: var(--accent); }
        button {
            padding: 12px 20px;
            background: #2E7D32;
            color: #FFF;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover { background: #388E3C; }
    </style>
</head>
<body>
    <header>
        <div class="header-title">⚡ Unified CED Studio: GPT-OSS Multimodal & KittenTTS-2 Live</div>
        <div class="header-badge" id="status-badge">● DUAL-GPU ENGINE ACTIVE</div>
    </header>
    <div id="chat-container"></div>
    <div id="input-bar">
        <input type="text" id="text-input" placeholder="Digite uma mensagem ou fórmula matemática (ex: \int x^2 + y = 10)..." autofocus />
        <button id="send-btn">Enviar</button>
    </div>

    <script>
        const chat = document.getElementById('chat-container');
        const input = document.getElementById('text-input');
        const sendBtn = document.getElementById('send-btn');

        async function sendMessage() {
            const text = input.value.trim();
            if (!text) return;
            input.value = '';

            // Renderizar balão do usuário
            const userEl = document.createElement('div');
            userEl.className = 'msg user';
            userEl.textContent = text;
            chat.appendChild(userEl);
            chat.scrollTop = chat.scrollHeight;

            // Criar placeholder do assistente
            const asstEl = document.createElement('div');
            asstEl.className = 'msg assistant';
            asstEl.innerHTML = '<div style="color:#888;">⏳ Processando na GPU 0 (RTX 2060) + GPU 1 (GTX 1050 Ti)...</div>';
            chat.appendChild(asstEl);
            chat.scrollTop = chat.scrollHeight;

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: text })
                });
                const data = await res.json();

                let html = '';
                if (data.thought) {
                    html += `<div class="thought-box">${data.thought}</div>`;
                }
                if (data.critic) {
                    html += `<div class="critic-box">${data.critic}</div>`;
                }
                html += `<div>${data.text}</div>`;
                if (data.metadata) {
                    html += `<div class="telemetry-tag">⚡ TTFT: ${data.metadata.ttft_ms} ms | TTFA: ${data.metadata.ttfa_ms} ms | Vazão: ${data.metadata.decode_tok_s} tok/s | Poda BVH: ${data.metadata.bvh_pruning_pct}%</div>`;
                }
                asstEl.innerHTML = html;

                // Tocar áudio se retornado
                if (data.audio_b64) {
                    const audio = new Audio("data:audio/wav;base64," + data.audio_b64);
                    audio.play().catch(e => console.log("Autoplay bloqueado:", e));
                }
            } catch (err) {
                asstEl.innerHTML = `<div style="color:#E57373;">[-] Erro na chamada: ${err}</div>`;
            }
            chat.scrollTop = chat.scrollHeight;
        }

        sendBtn.addEventListener('click', sendMessage);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
"""

class LiveRequestHandler(BaseHTTPRequestHandler):
    session: LiveSession

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_LIVE_CLIENT.encode("utf-8"))
        elif self.path == "/api/sessions":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            sessions = self.session.list_sessions()
            self.wfile.write(json.dumps({"sessions": sessions, "active": self.session.active_session_id}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat":
            content_len = int(self.headers.get("Content-Length", 0))
            post_body = self.rfile.read(content_len).decode("utf-8")
            data = json.loads(post_body)
            msg = data.get("message", "")

            # Executa turno no LiveSession
            thought_text = ""
            critic_text = ""
            spoken_text = ""
            last_meta = {}

            for stream_type, chunk, meta in self.session.stream_turn(msg):
                last_meta = meta
                if stream_type == "thought":
                    thought_text += chunk
                elif stream_type == "critic":
                    critic_text += chunk
                elif stream_type == "text":
                    spoken_text += chunk

            # Gerar WAV base64 para reprodução no browser
            audio_b64 = ""
            if spoken_text and self.session.audio_engine:
                wav_bytes = self.session.audio_engine.synthesize_speech_wav(spoken_text)
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

            res_payload = {
                "thought": thought_text.strip(),
                "critic": critic_text.strip(),
                "text": spoken_text.strip(),
                "metadata": last_meta,
                "audio_b64": audio_b64
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(res_payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silencia logs verbosos de HTTP para não poluir o terminal
        return

class RealtimeBridgeServer:
    """Servidor HTTP e Realtime Bridge local."""

    def __init__(self, session: LiveSession, port: int = 8765):
        self.session = session
        self.port = port
        self.server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        handler_cls = type("LiveHandlerWithSession", (LiveRequestHandler,), {"session": self.session})
        self.server = HTTPServer(("127.0.0.1", self.port), handler_cls)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True, name="LiveServerThread")
        self._thread.start()

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
