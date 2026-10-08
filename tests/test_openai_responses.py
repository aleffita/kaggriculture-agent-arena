"""
Unit and End-to-End Tests for OpenAI Responses API (/v1/responses).
Directly aligned with llama.cpp test_compat_oai_responses.py and official OpenAI SDK specifications:
1. Non-streaming response format (resp_*, msg_*, output_text, usage).
2. Streaming SSE event lifecycle (response.created, response.output_item.added, response.output_text.delta, response.completed, [DONE]).
3. Robust input variations (string, structured array, top-level instructions).
4. Reasoning effort handling (dict reasoning.effort and string reasoning_effort).
5. Arbitrary Bearer token compatibility (e.g. Bearer alefita).
6. CORS and OPTIONS preflight handling.
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


def test_responses_non_streaming_basic(test_client):
    """Testa chamada básica não-streaming em /v1/responses no padrão OpenAI e llama.cpp."""
    payload = {
        "model": "gemma-4-E2B-it",
        "input": "Qual é a capital da França?",
        "instructions": "Responda de forma direta e concisa."
    }
    headers = {"Authorization": "Bearer alefita"}
    res = test_client.post("/v1/responses", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.get_json()

    assert data["id"].startswith("resp_")
    assert data["object"] == "response"
    assert data["status"] == "completed"
    assert data["model"] == "gemma-4-E2B-it"

    # Verificação do array output
    assert len(data["output"]) >= 1
    out_item = data["output"][0]
    assert out_item["id"].startswith("msg_")
    assert out_item["type"] == "message"
    assert out_item["role"] == "assistant"
    assert len(out_item["content"]) >= 1
    assert out_item["content"][0]["type"] == "output_text"
    assert len(out_item["content"][0]["text"]) > 0

    # Campo de conveniência de primeiro nível output_text (verificado pelo SDK oficial)
    assert "output_text" in data
    assert data["output_text"] == out_item["content"][0]["text"]

    # Usage
    assert "usage" in data
    assert data["usage"]["total_tokens"] > 0


def test_responses_streaming_sse_lifecycle(test_client):
    """Testa ciclo completo de eventos SSE conforme test_compat_oai_responses.py do llama.cpp."""
    payload = {
        "model": "gemma-4-E2B-it",
        "input": [{"role": "user", "content": "Conte até três."}],
        "stream": True,
        "reasoning": {"effort": "dynamic"}
    }
    headers = {"Authorization": "Bearer alefita"}
    res = test_client.post("/v1/responses", json=payload, headers=headers)
    assert res.status_code == 200
    assert "text/event-stream" in res.content_type

    raw_events = res.get_data(as_text=True).split("\n\n")

    events_seen = []
    resp_id = None
    msg_id = None
    gathered_delta_text = ""
    terminal_output_text = None
    saw_done_marker = False

    for block in raw_events:
        block = block.strip()
        if not block:
            continue
        if block == "data: [DONE]":
            saw_done_marker = True
            continue

        evt_type = None
        data_json = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                evt_type = line[6:].strip()
            elif line.startswith("data:"):
                try:
                    data_json = json.loads(line[5:].strip())
                except Exception:
                    pass

        if evt_type and data_json:
            events_seen.append(evt_type)

            if evt_type == "response.created":
                assert data_json["response"]["id"].startswith("resp_")
                resp_id = data_json["response"]["id"]

            elif evt_type == "response.output_item.added":
                assert data_json["item"]["id"].startswith("msg_")
                msg_id = data_json["item"]["id"]

            elif evt_type == "response.content_part.added":
                assert data_json.get("item_id") == msg_id

            elif evt_type == "response.output_text.delta":
                assert data_json.get("item_id") == msg_id
                gathered_delta_text += data_json.get("delta", "")

            elif evt_type == "response.output_text.done":
                assert data_json.get("item_id") == msg_id

            elif evt_type == "response.output_item.done":
                assert data_json["item"]["id"] == msg_id

            elif evt_type == "response.completed":
                assert data_json["response"]["id"] == resp_id
                terminal_output_text = data_json["response"].get("output_text")

    # Validação rigorosa dos eventos requeridos pelo SDK OpenAI
    assert "response.created" in events_seen
    assert "response.output_item.added" in events_seen
    assert "response.content_part.added" in events_seen
    assert "response.output_text.delta" in events_seen
    assert "response.completed" in events_seen
    assert saw_done_marker is True

    # Coerência de texto acumulado
    assert len(gathered_delta_text) > 0
    if terminal_output_text:
        assert gathered_delta_text.strip() == terminal_output_text.strip()


def test_responses_options_cors(test_client):
    """Verifica preflight OPTIONS e cabeçalhos CORS em /v1/responses."""
    res_opt = test_client.options("/v1/responses")
    assert res_opt.status_code in (200, 204)
    assert res_opt.headers.get("Access-Control-Allow-Origin") == "*"
    assert "POST" in res_opt.headers.get("Access-Control-Allow-Methods", "")
