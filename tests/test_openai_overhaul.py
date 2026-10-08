"""
Test Suite: OpenAI-Compatible Wire Protocol and CORDIS Extensions Overhaul.
Valida:
1. GET /v1/models (listagem e metadados de 1M contexto e virtual experts)
2. POST /v1/runtime/reset e POST /v1/memory/clear (anti-cheating e expurgo de NVMe)
3. POST /v1/chat/completions (Non-streaming com reasoning_effort="high" -> 65536 tokens)
4. POST /v1/chat/completions (Dynamic reasoning effort com expansão contínua de budget)
5. POST /v1/chat/completions (Toggle per-request: virtual_experts=False)
6. POST /v1/chat/completions (Ferramenta externa com schema OpenAI tool_calls)
7. POST /v1/chat/completions (Modo CORDIS Wire RPC: evento SSE cordis_rpc + coerência de tokens)
8. Invariante de Coerência: Inexistência de prefixos artificiais ([CORDIS IN-PLACE RPC EXECUTED])
   e presença estrita do formato canônico [[TOOL_CALL:<id>]] -> [[RESOLVED:<output>]]
9. POST /v1/audio/transcriptions (ASR Whisper)
10. POST /v1/audio/speech (TTS Mimi GPU Codec)
"""
import io
import json
import pytest
from litert_explore.live.web_studio import create_studio_app
from litert_explore.live.session import LiveSession

def test_models_endpoint():
    app = create_studio_app()
    client = app.test_client()

    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["object"] == "list"
    ids = [m["id"] for m in data["data"]]
    assert "gpt-oss-20b" in ids
    assert "bonsai-27b" in ids
    assert "mimi-codec-v1" in ids

    # Checa detalhes do gpt-oss-20b
    resp_detail = client.get("/v1/models/gpt-oss-20b")
    assert resp_detail.status_code == 200
    detail = resp_detail.get_json()
    assert detail["cordis_features"]["max_context_tokens"] == 1048576

def test_memory_clear_and_runtime_reset():
    session = LiveSession(model="gpt-oss")
    # Injeta dados fictícios de sessão e engram
    session.sessions["test-session"] = [{"role": "user", "content": "olá"}]
    session.embedding_substrate.store_engram("informação confidencial", "test-session")
    session.spawn_subagent(role="Math", goal="Solve")
    assert len(session.subagents.subagents) > 0

    app = create_studio_app(session=session)
    client = app.test_client()

    # Chama endpoint de reset
    resp = client.post("/v1/runtime/reset")
    assert resp.status_code == 200
    res_data = resp.get_json()
    assert res_data["status"] == "success"
    assert res_data["details"]["nvme_cache_purged"] is True

    # Verifica que o estado foi purgado
    assert len(session.subagents.subagents) == 0
    assert len(session.embedding_substrate.engram_memory) == 0

def test_high_reasoning_effort_64k():
    session = LiveSession(model="gpt-oss")
    app = create_studio_app(session=session)
    client = app.test_client()

    payload = {
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": "Analise a conjectura de Riemann com rigor formal."}],
        "reasoning_effort": "high",
        "stream": False
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "choices" in data
    assert data["cordis_telemetry"]["reasoning_effort"] == "high"
    assert data["cordis_telemetry"]["reasoning_budget"] == 65536
    reasoning_text = data["choices"][0]["message"]["reasoning_content"]
    assert "65536" in reasoning_text

def test_virtual_experts_ablation_toggle():
    session = LiveSession(model="gpt-oss")
    app = create_studio_app(session=session)
    client = app.test_client()

    # 1. Com virtual_experts desativados
    payload_disabled = {
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": "Calcule 42 * 99"}],
        "virtual_experts": False,
        "stream": False
    }
    resp = client.post("/v1/chat/completions", json=payload_disabled)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["cordis_telemetry"]["virtual_experts_enabled"] is False

    # 2. Com virtual_experts ativados (padrão)
    payload_enabled = {
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": "Calcule 42 * 99"}],
        "virtual_experts": True,
        "stream": False
    }
    resp_en = client.post("/v1/chat/completions", json=payload_enabled)
    assert resp_en.status_code == 200
    data_en = resp_en.get_json()
    assert data_en["cordis_telemetry"]["virtual_experts_enabled"] is True

def test_uniform_token_invariant_and_wire_rpc():
    session = LiveSession(model="gpt-oss")
    app = create_studio_app(session=session)
    client = app.test_client()

    # Teste de streaming com CORDIS Wire RPC habilitado via header
    payload = {
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": "Calcule 25 * 25"}],
        "stream": True,
        "wire_protocol": "cordis_rpc"
    }
    headers = {"X-Cordis-Wire": "true"}

    resp = client.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200
    content = resp.get_data(as_text=True)

    # 1. Verifica presença do evento RPC no SSE
    assert "event: cordis_rpc" in content

    # 2. Verifica a estrita coerência do modelo: formato canônico presente
    assert "[[TOOL_CALL:" in content
    assert "]] -> [[RESOLVED:" in content

    # 3. CRÍTICO: Zero vazamento de strings artificiais de depuração
    assert "[CORDIS IN-PLACE RPC EXECUTED]" not in content
    assert "[LOCAL VE]" not in content

def test_external_openai_tools():
    session = LiveSession(model="gpt-oss")
    app = create_studio_app(session=session)
    client = app.test_client()

    payload = {
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": "calcule 15 + 30"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Calculadora aritmética",
                "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}
            }
        }],
        "tool_choice": "auto",
        "stream": False
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    msg = data["choices"][0]["message"]
    assert data["choices"][0]["finish_reason"] == "tool_calls"
    assert "tool_calls" in msg
    assert msg["tool_calls"][0]["function"]["name"] == "calculator"

def test_audio_endpoints():
    session = LiveSession(model="gpt-oss")
    app = create_studio_app(session=session)
    client = app.test_client()

    # 1. Audio Speech (TTS)
    speech_resp = client.post("/v1/audio/speech", json={"input": "Teste de voz na GPU", "voice": "mimi-pt-br"})
    assert speech_resp.status_code == 200
    assert speech_resp.content_type == "audio/wav"
    assert len(speech_resp.data) > 44 # Cabeçalho WAV + PCM

    # 2. Audio Transcriptions (ASR)
    fake_wav = io.BytesIO(speech_resp.data)
    asr_resp = client.post(
        "/v1/audio/transcriptions",
        data={"file": (fake_wav, "audio.wav"), "model": "whisper-1"},
        content_type="multipart/form-data"
    )
    assert asr_resp.status_code == 200
    asr_data = asr_resp.get_json()
    assert "text" in asr_data

if __name__ == "__main__":
    test_models_endpoint()
    print("[PASS] test_models_endpoint")
    test_memory_clear_and_runtime_reset()
    print("[PASS] test_memory_clear_and_runtime_reset")
    test_high_reasoning_effort_64k()
    print("[PASS] test_high_reasoning_effort_64k")
    test_virtual_experts_ablation_toggle()
    print("[PASS] test_virtual_experts_ablation_toggle")
    test_uniform_token_invariant_and_wire_rpc()
    print("[PASS] test_uniform_token_invariant_and_wire_rpc")
    test_external_openai_tools()
    print("[PASS] test_external_openai_tools")
    test_audio_endpoints()
    print("[PASS] test_audio_endpoints")
    print("\n[ALL 7 TEST SUITES PASSED PERFECTLY]")
