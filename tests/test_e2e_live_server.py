"""
End-to-End Live HTTP Socket Tests for Unified CED OpenAI-Compatible Runtime.
Launches the server on an ephemeral local TCP socket and exercises real HTTP wire requests:
1. HTTP GET /v1/models (wire headers, CORS, status 200).
2. HTTP OPTIONS /v1/chat/completions (CORS preflight 204).
3. HTTP POST /v1/chat/completions (real streaming SSE chunks across TCP socket).
4. HTTP POST /v1/responses (real streaming SSE events across TCP socket).
5. Authorization header tolerance: Accepts arbitrary token 'Bearer alefita'.
"""
from __future__ import annotations

import json
import socket
import threading
import time
import urllib.request
import urllib.error
import pytest

from litert_explore.live.web_studio import create_studio_app
from litert_explore.live.session import LiveSession


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def live_server_url():
    port = get_free_port()
    session = LiveSession(model="gemma-4-E2B-it")
    app = create_studio_app(session=session, headless=True)

    server_thread = threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False),
        daemon=True
    )
    server_thread.start()

    # Aguarda o socket aceitar conexões
    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=1.0) as resp:
                if resp.status == 200:
                    break
        except Exception:
            time.sleep(0.08)

    return base_url


def test_e2e_wire_models_list(live_server_url):
    """Testa requisição real de rede GET /v1/models via socket HTTP."""
    req = urllib.request.Request(f"{live_server_url}/v1/models")
    req.add_header("Authorization", "Bearer alefita")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status == 200
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        body = json.loads(resp.read().decode("utf-8"))
        assert body["object"] == "list"
        model_ids = [m["id"] for m in body["data"]]
        assert "gemma-4-E2B-it" in model_ids


def test_e2e_wire_options_preflight(live_server_url):
    """Testa requisição real de rede OPTIONS /v1/chat/completions (preflight CORS)."""
    req = urllib.request.Request(f"{live_server_url}/v1/chat/completions", method="OPTIONS")
    req.add_header("Origin", "http://localhost:3000")
    req.add_header("Access-Control-Request-Method", "POST")
    req.add_header("Access-Control-Request-Headers", "authorization,content-type")

    with urllib.request.urlopen(req, timeout=5.0) as resp:
        assert resp.status in (200, 204)
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"


def test_e2e_wire_chat_completions_streaming(live_server_url):
    """Testa streaming real de SSE via socket HTTP em /v1/chat/completions."""
    payload = json.dumps({
        "model": "gemma-4-E2B-it",
        "messages": [{"role": "user", "content": "Olá!"}],
        "stream": True,
        "reasoning_effort": "medium"
    }).encode("utf-8")

    req = urllib.request.Request(f"{live_server_url}/v1/chat/completions", data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer alefita")

    with urllib.request.urlopen(req, timeout=10.0) as resp:
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("Content-Type", "")

        chunks = []
        for line in resp:
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("data: "):
                chunks.append(decoded)

        assert len(chunks) > 0
        assert any("[DONE]" in c for c in chunks)


def test_e2e_wire_responses_streaming(live_server_url):
    """Testa streaming real de eventos SSE via socket HTTP em /v1/responses."""
    payload = json.dumps({
        "model": "gemma-4-E2B-it",
        "input": "Responda brevemente.",
        "stream": True,
        "reasoning": {"effort": "dynamic"}
    }).encode("utf-8")

    req = urllib.request.Request(f"{live_server_url}/v1/responses", data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer alefita")

    with urllib.request.urlopen(req, timeout=10.0) as resp:
        assert resp.status == 200
        assert "text/event-stream" in resp.headers.get("Content-Type", "")

        events = []
        for line in resp:
            decoded = line.decode("utf-8").strip()
            if decoded.startswith("event: "):
                events.append(decoded[7:])

        assert "response.created" in events
        assert "response.output_text.delta" in events
        assert "response.completed" in events


def test_e2e_wire_gpt_oss_conversation(live_server_url):
    """Verifica conversa real com GPT-OSS 20B pelo socket HTTP com identificação correta."""
    payload = json.dumps({
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": "Quem e voce?"}],
        "stream": False,
        "reasoning_effort": "medium"
    }).encode("utf-8")

    req = urllib.request.Request(f"{live_server_url}/v1/chat/completions", data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer alefita")

    with urllib.request.urlopen(req, timeout=10.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["model"] == "gpt-oss-20b"
        msg = data["choices"][0]["message"]
        assert msg["role"] == "assistant"
        content = msg["content"]
        assert "GPT-OSS" in content or "Unified" in content
        assert "Gemma" not in content


def test_e2e_wire_harness_interrupt_tool_no_loop(live_server_url):
    """Garante que ferramentas de controle de harness (interrupt_agent) não geram loop de tool_calls em conversa normal."""
    payload = json.dumps({
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": "Olá, tudo bem?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "interrupt_agent",
                    "description": "Interrompe execução em segundo plano",
                    "parameters": {
                        "type": "object",
                        "properties": {"target": {"type": "string"}},
                        "required": ["target"]
                    }
                }
            }
        ],
        "tool_choice": "auto",
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(f"{live_server_url}/v1/chat/completions", data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", "Bearer alefita")

    with urllib.request.urlopen(req, timeout=10.0) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        choice = data["choices"][0]
        # Não pode disparar tool_calls espúrio para interrupt_agent
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] is not None
        assert "tool_calls" not in choice["message"]


def test_e2e_web_ui_html():
    """Verifica renderização da interface Web minimalista estilo llama.cpp em GET /."""
    session = LiveSession(model="gpt-oss")
    app = create_studio_app(session=session, headless=False)
    client = app.test_client()

    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.data.decode("utf-8")
    assert "<title>Unified CED Runtime - Minimal Web UI" in html
    assert 'id="model-select"' in html
    assert 'id="effort-select"' in html
    assert 'id="endpoint-select"' in html
    assert "Three.js" not in html
    assert "webgl-canvas" not in html

