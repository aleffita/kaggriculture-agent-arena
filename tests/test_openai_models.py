"""
Unit Tests for OpenAI /v1/models and /v1/models/<model_id> endpoints.
Verifies spec-compliance with official OpenAI API, Open WebUI, and LibreChat contracts:
1. Pure canonical model list (zero fictitious model hallucinations).
2. Metadata for reasoning effort selection (supports_reasoning, reasoning_effort_supported).
3. Dual-mode response schema (OpenAI 'data' list + OpenWebUI/Ollama 'models' array).
4. Robust case-insensitive lookup and non-breaking fallback for external harnesses.
5. CORS headers and HTTP OPTIONS preflight support.
"""
from __future__ import annotations

import pytest
from litert_explore.live.web_studio import create_studio_app
from litert_explore.live.session import LiveSession


@pytest.fixture
def test_client():
    session = LiveSession(model="gemma-4-E2B-it")
    app = create_studio_app(session=session, headless=True)
    return app.test_client()


def test_models_list_contains_only_real_models(test_client):
    """Garante que apenas modelos reais do repositório estão expostos em /v1/models."""
    res = test_client.get("/v1/models")
    assert res.status_code == 200
    data = res.get_json()

    assert data["object"] == "list"
    assert "data" in data
    model_ids = [m["id"] for m in data["data"]]

    # Modelos canônicos obrigatórios
    assert "gemma-4-E2B-it" in model_ids
    assert "gpt-oss-20b" in model_ids
    assert "bonsai-27b" in model_ids
    assert "ornith-35b" in model_ids

    # Proibição de modelos fictícios inventados
    assert "gpt-4o-mini" not in model_ids
    assert "gpt-4o" not in model_ids
    assert "o1-mini" not in model_ids
    assert "o1" not in model_ids
    assert "o3-mini" not in model_ids
    assert "gpt-3.5-turbo" not in model_ids


def test_models_have_reasoning_metadata_for_harnesses(test_client):
    """Verifica se os modelos expõem metadados de reasoning para o seletor de effort do harness."""
    res = test_client.get("/v1/models")
    data = res.get_json()

    gemma = next(m for m in data["data"] if m["id"] == "gemma-4-E2B-it")
    assert gemma["supports_reasoning"] is True
    assert gemma["reasoning_effort_supported"] is True
    assert "reasoning" in gemma["capabilities"]
    assert gemma["capabilities"]["reasoning"] is True
    assert set(gemma["supported_reasoning_efforts"]) == {"low", "medium", "high", "dynamic", "ultra", "ultra-dynamic"}
    assert gemma["max_tokens"] >= 65536
    assert gemma["context_window"] >= 2048


def test_models_list_openwebui_dual_schema(test_client):
    """Verifica o array 'models' de compatibilidade no estilo Ollama/OpenWebUI."""
    res = test_client.get("/v1/models")
    data = res.get_json()

    assert "models" in data
    assert isinstance(data["models"], list)
    model_names = [m["name"] for m in data["models"]]
    assert "gemma-4-E2B-it" in model_names
    assert "gpt-oss-20b" in model_names


def test_model_detail_lookup_case_insensitive(test_client):
    """Verifica busca de modelo específico por ID com tolerância a case."""
    res_exact = test_client.get("/v1/models/gemma-4-E2B-it")
    assert res_exact.status_code == 200
    m_exact = res_exact.get_json()
    assert m_exact["id"] == "gemma-4-E2B-it"

    res_upper = test_client.get("/v1/models/GEMMA-4-E2B-IT")
    assert res_upper.status_code == 200
    m_upper = res_upper.get_json()
    assert m_upper["id"] == "gemma-4-E2B-it"


def test_model_detail_lookup_fallback_prevents_harness_break(test_client):
    """Se o harness consultar um modelo genérico, retorna descritor válido com reasoning ao invés de 404."""
    res = test_client.get("/v1/models/custom-agent-v1")
    assert res.status_code == 200
    data = res.get_json()
    assert data["id"] == "custom-agent-v1"
    assert data["supports_reasoning"] is True
    assert data["reasoning_effort_supported"] is True


def test_models_cors_headers_and_options(test_client):
    """Verifica suporte a CORS e requisições HTTP OPTIONS preflight de navegadores."""
    res_opt = test_client.options("/v1/models")
    assert res_opt.status_code in (200, 204)
    assert res_opt.headers.get("Access-Control-Allow-Origin") == "*"
    assert "GET" in res_opt.headers.get("Access-Control-Allow-Methods", "")

    res_get = test_client.get("/v1/models")
    assert res_get.headers.get("Access-Control-Allow-Origin") == "*"
