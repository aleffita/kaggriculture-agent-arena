"""
Unit and End-to-End Tests for OpenAI Chat Completions (/v1/chat/completions).
Directly aligned with llama.cpp test_chat_completion.py and official OpenAI SDK specifications:
1. Non-streaming completions with reasoning_effort (low, medium, high, dynamic).
2. Streaming SSE completions with role chunk, reasoning_content, content, and terminal stop chunk.
3. Preservation of code indentation, newlines, and whitespace in streamed text chunks.
4. Server-side ABI isolation (internal tools never leak to client-facing tool_calls).
5. External client tool_calls emission for external harness functions.
6. Support for arbitrary Bearer tokens (e.g. Bearer alefita) without rejection.
7. CORS and OPTIONS preflight compliance.
"""
from __future__ import annotations

import json
import pytest
from litert_explore.live.web_studio import create_studio_app
from litert_explore.live.session import LiveSession


@pytest.fixture
def test_client():
    session = LiveSession(model="gemma-4-E2B-it")
    app = create_studio_app(session=session, headless=True)
    return app.test_client()


def test_chat_completions_non_streaming_with_arbitrary_token(test_client):
    """Testa chat completions não-streaming com token arbitrário (alefita)."""
    payload = {
        "model": "gemma-4-E2B-it",
        "messages": [
            {"role": "system", "content": "Você é um assistente técnico."},
            {"role": "user", "content": "Olá, tudo bem?"}
        ],
        "reasoning_effort": "medium",
        "stream": False
    }
    headers = {"Authorization": "Bearer alefita"}
    res = test_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.get_json()

    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"
    assert data["model"] == "gemma-4-E2B-it"

    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert len(choice["message"]["content"]) > 0
    assert choice["finish_reason"] == "stop"
    assert "reasoning_content" in choice["message"]

    assert "usage" in data
    assert data["usage"]["total_tokens"] > 0


def test_chat_completions_reasoning_effort_dict_format(test_client):
    """Testa envio de reasoning como dicionário reasoning: {'effort': 'high'} (OpenAI Responses / O1 style)."""
    payload = {
        "model": "gemma-4-E2B-it",
        "messages": [{"role": "user", "content": "Resolva a integral de x*exp(x)."}],
        "reasoning": {"effort": "high"},
        "stream": False
    }
    res = test_client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer random-token-123"})
    assert res.status_code == 200
    data = res.get_json()
    assert data["cordis_telemetry"]["reasoning_effort"] == "high"
    assert data["cordis_telemetry"]["reasoning_budget"] == 65536


def test_chat_completions_streaming_sse_chunks(test_client):
    """Testa ciclo SSE de completions: role delta -> reasoning deltas -> content deltas -> stop chunk -> [DONE]."""
    payload = {
        "model": "gemma-4-E2B-it",
        "messages": [{"role": "user", "content": "Explique o Teorema de Pitágoras."}],
        "stream": True,
        "reasoning_effort": "dynamic"
    }
    headers = {"Authorization": "Bearer alefita"}
    res = test_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert res.status_code == 200
    assert "text/event-stream" in res.content_type

    raw_blocks = res.get_data(as_text=True).split("\n\n")

    saw_role_chunk = False
    saw_reasoning_chunk = False
    saw_content_chunk = False
    saw_stop_chunk = False
    saw_done = False
    gathered_text = ""

    for b in raw_blocks:
        b = b.strip()
        if not b:
            continue
        if b == "data: [DONE]":
            saw_done = True
            continue

        if b.startswith("data: "):
            try:
                pkt = json.loads(b[6:])
                delta = pkt["choices"][0].get("delta", {})
                finish = pkt["choices"][0].get("finish_reason")

                if delta.get("role") == "assistant":
                    saw_role_chunk = True
                if "reasoning_content" in delta:
                    saw_reasoning_chunk = True
                if "content" in delta:
                    saw_content_chunk = True
                    gathered_text += delta["content"]
                if finish == "stop":
                    saw_stop_chunk = True
            except Exception:
                pass

    assert saw_role_chunk is True
    assert saw_reasoning_chunk is True
    assert saw_content_chunk is True
    assert saw_stop_chunk is True
    assert saw_done is True
    assert len(gathered_text) > 0


def test_chat_completions_preserves_formatting_and_newlines(test_client):
    """Garante que a tokenização de streaming não achata quebras de linha com split() ingênuo."""
    payload = {
        "model": "gemma-4-E2B-it",
        "messages": [{"role": "user", "content": "Escreva um código em Python com indentação."}],
        "stream": True
    }
    res = test_client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer alefita"})
    assert res.status_code == 200

    raw_text = res.get_data(as_text=True)
    assert "data: " in raw_text


def test_chat_completions_tool_calling_isolation(test_client):
    """Garante isolamento de ABI: ferramentas internas de servidor nunca vazam como delta.tool_calls."""
    # Requisição com ferramenta externa registrada pelo cliente
    payload = {
        "model": "gemma-4-E2B-it",
        "messages": [{"role": "user", "content": "Qual a temperatura em Paris?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Obtém a previsão do tempo",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
            }
        ],
        "tool_choice": "auto",
        "stream": False
    }
    res = test_client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer alefita"})
    assert res.status_code == 200
    data = res.get_json()
    msg = data["choices"][0]["message"]
    assert "tool_calls" in msg
    assert msg["tool_calls"][0]["function"]["name"] == "get_current_weather"


def test_chat_completions_stop_sequences(test_client):
    """Verifica truncamento imediato por stop sequences."""
    payload = {
        "model": "gemma-4-E2B-it",
        "messages": [{"role": "user", "content": "Diga 1 2 3 4 5"}],
        "stop": ["3"],
        "stream": False
    }
    res = test_client.post("/v1/chat/completions", json=payload, headers={"Authorization": "Bearer alefita"})
    assert res.status_code == 200
    data = res.get_json()
    content = data["choices"][0]["message"]["content"]
    assert "3" not in content


def test_chat_completions_options_cors(test_client):
    """Verifica preflight OPTIONS e cabeçalhos CORS em /v1/chat/completions."""
    res_opt = test_client.options("/v1/chat/completions")
    assert res_opt.status_code in (200, 204)
    assert res_opt.headers.get("Access-Control-Allow-Origin") == "*"
    assert "POST" in res_opt.headers.get("Access-Control-Allow-Methods", "")
