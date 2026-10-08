"""
Stateless JWT Shadow Token Engine for Granular Virtual Expert Ablation and Multi-Model Routing.
Provides purely standard-library (HMAC-SHA256) JWT minting, validation, and claim resolution
without external dependencies or persistent database requirements.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

SERVER_SECRET_KEY = os.environ.get("LITERT_SHADOW_SECRET", "litert-ced-stateless-shadow-secret-2026").encode("utf-8")

DEFAULT_VIRTUAL_EXPERTS_STATE: Dict[str, bool] = {
    "enabled": True,
    "python_repl": True,
    "code_interpreter": True,
    "llm_code_gen": False,        # Proibição de synthetic code replacement em benchmarks
    "microtex_lean4": True,       # Formal math & theorem prover
    "analytical_plotter": True,   # Matryoshka projections & visual graphs
    "web_search": True,           # Dual web search (Tavily + webfetch)
    "browser": True,              # Interactive browser sandbox
    "request_budget": True,       # Dynamic reasoning budget expansion
    "disk_expert_store": True,    # Overlapped Direct NVMe mini-expert paging
    "llvm_jit": False,            # JIT execution (desativado por padrão)
    "lsp_language_server": False, # LSP AST analysis (desativado por padrão)
}


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    padding = len(s) % 4
    if padding:
        s += "=" * (4 - padding)
    return base64.urlsafe_b64decode(s.encode("ascii"))


def is_shadow_token(token_candidate: Optional[str]) -> bool:
    """Verifica de forma rápida se o texto fornecido tem o formato sintático de um JWT."""
    if not token_candidate or not isinstance(token_candidate, str):
        return False
    parts = token_candidate.strip().split(".")
    return len(parts) == 3


def decode_shadow_token(token_str: str) -> Optional[Dict[str, Any]]:
    """
    Decodifica e verifica a assinatura HMAC-SHA256 de um Shadow Token JWT.
    Retorna o dicionário de claims se válido, ou None se inválido/corrompido.
    """
    if not token_str or not isinstance(token_str, str):
        return None

    parts = token_str.strip().split(".")
    if len(parts) != 3:
        return None

    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")

    # Verificação da assinatura HMAC
    expected_sig = hmac.new(SERVER_SECRET_KEY, signing_input, hashlib.sha256).digest()
    try:
        actual_sig = _b64url_decode(signature_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
    except Exception:
        return None

    # Parsing do Payload
    try:
        payload_bytes = _b64url_decode(payload_b64)
        claims = json.loads(payload_bytes.decode("utf-8"))
        if isinstance(claims, dict):
            return claims
    except Exception:
        pass

    return None


def mint_shadow_token(
    claims_delta: Dict[str, Any],
    base_token: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Emite um novo Shadow Token JWT assinado.
    Se 'base_token' for informado, herda e mescla suas claims com o claims_delta.
    """
    # 1. Base Claims inicial
    final_claims: Dict[str, Any] = {
        "sub": "litert-shadow-session",
        "iat": int(time.time()),
        "model": "gemma-4-E2B-it",
        "reasoning_effort": "medium",
        "virtual_experts": copy.deepcopy(DEFAULT_VIRTUAL_EXPERTS_STATE),
        "raw_output": False,
        "metadata": {}
    }

    # 2. Se base_token for válido, sobrepõe
    if base_token:
        base_claims = decode_shadow_token(base_token)
        if base_claims:
            for k, v in base_claims.items():
                if k == "virtual_experts" and isinstance(v, dict):
                    final_claims["virtual_experts"].update(v)
                elif k != "iat":
                    final_claims[k] = v

    # 3. Aplica claims_delta
    for k, v in claims_delta.items():
        if k == "virtual_experts":
            if isinstance(v, bool):
                final_claims["virtual_experts"]["enabled"] = v
            elif isinstance(v, str):
                v_l = v.lower()
                if v_l in ("false", "0", "off", "no", "none"):
                    final_claims["virtual_experts"]["enabled"] = False
                elif v_l == "no_code":
                    final_claims["virtual_experts"]["enabled"] = True
                    final_claims["virtual_experts"]["python_repl"] = True         # Mantém REPL Python ativo para matemática
                    final_claims["virtual_experts"]["code_interpreter"] = True
                    final_claims["virtual_experts"]["llm_code_gen"] = False
                    final_claims["virtual_experts"]["microtex_lean4"] = False     # Desativa Lean 4 na variante no-code
                    final_claims["virtual_experts"]["llvm_jit"] = False           # Desativa LLVM JIT
                    final_claims["virtual_experts"]["lsp_language_server"] = False# Desativa LSP language server
                    final_claims["virtual_experts"]["analytical_plotter"] = True
                    final_claims["virtual_experts"]["web_search"] = True
                    final_claims["virtual_experts"]["browser"] = True
                    final_claims["virtual_experts"]["request_budget"] = True
                    final_claims["virtual_experts"]["disk_expert_store"] = True
                elif v_l == "math_only":
                    final_claims["virtual_experts"]["enabled"] = True
                    final_claims["virtual_experts"]["python_repl"] = True
                    final_claims["virtual_experts"]["code_interpreter"] = False
                    final_claims["virtual_experts"]["llm_code_gen"] = False
                    final_claims["virtual_experts"]["microtex_lean4"] = True
                    final_claims["virtual_experts"]["analytical_plotter"] = False
                    final_claims["virtual_experts"]["web_search"] = False
                    final_claims["virtual_experts"]["browser"] = False
                    final_claims["virtual_experts"]["request_budget"] = True
                    final_claims["virtual_experts"]["disk_expert_store"] = False
                else:
                    final_claims["virtual_experts"]["enabled"] = True
            elif isinstance(v, dict):
                final_claims["virtual_experts"].update(v)
        elif k != "iat":
            final_claims[k] = v

    final_claims["iat"] = int(time.time())

    # 4. Assinatura do JWT
    header = {"alg": "HS256", "typ": "JWT"}
    header_json = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload_json = json.dumps(final_claims, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    header_b64 = _b64url_encode(header_json)
    payload_b64 = _b64url_encode(payload_json)

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    signature = hmac.new(SERVER_SECRET_KEY, signing_input, hashlib.sha256).digest()
    signature_b64 = _b64url_encode(signature)

    jwt_token = f"{header_b64}.{payload_b64}.{signature_b64}"
    return jwt_token, final_claims


def resolve_effective_configuration(
    request_headers: Dict[str, str],
    request_args: Dict[str, str],
    request_body: Dict[str, Any],
    fallback_model: str = "gemma-4-E2B-it",
    fallback_ve_enabled: bool = True
) -> Dict[str, Any]:
    """
    Resolve dinamicamente a configuração efetiva para uma chamada:
    1. Extrai Shadow Token do header Authorization: Bearer <JWT> ou X-Config-Token ou query.
    2. Aplica overrides per-request (body['meta'], body['extra_body'], query params).
    Retorna dicionário canônico com 'model', 'virtual_experts', 'raw_output', etc.
    """
    token_candidate = None

    # A. Checar Authorization: Bearer <token>
    auth_header = request_headers.get("Authorization") or request_headers.get("authorization") or ""
    if auth_header.startswith("Bearer ") or auth_header.startswith("bearer "):
        possible_token = auth_header[7:].strip()
        if is_shadow_token(possible_token):
            token_candidate = possible_token

    # B. Checar Headers Customizados
    if not token_candidate:
        for h in ["X-Shadow-Token", "X-Config-Token", "X-Session-Token"]:
            v = request_headers.get(h)
            if v and is_shadow_token(v.strip()):
                token_candidate = v.strip()
                break

    # C. Checar Query Param
    if not token_candidate:
        q_tok = request_args.get("token") or request_args.get("shadow_token")
        if q_tok and is_shadow_token(q_tok.strip()):
            token_candidate = q_tok.strip()

    # D. Decodificar Shadow Token se presente
    claims = decode_shadow_token(token_candidate) if token_candidate else None

    # E. Estrutura base resultante
    effective: Dict[str, Any] = {
        "model": fallback_model,
        "reasoning_effort": "medium",
        "virtual_experts": copy.deepcopy(DEFAULT_VIRTUAL_EXPERTS_STATE),
        "raw_output": False,
        "from_shadow_token": bool(claims is not None),
    }
    effective["virtual_experts"]["enabled"] = fallback_ve_enabled

    if claims:
        if "model" in claims and claims["model"]:
            effective["model"] = claims["model"]
        if "reasoning_effort" in claims:
            effective["reasoning_effort"] = claims["reasoning_effort"]
        if "raw_output" in claims:
            effective["raw_output"] = bool(claims["raw_output"])
        if "virtual_experts" in claims and isinstance(claims["virtual_experts"], dict):
            effective["virtual_experts"].update(claims["virtual_experts"])

    # F. Overrides explícitos da requisição (body e query)
    if "model" in request_body and request_body["model"]:
        effective["model"] = request_body["model"]

    # Extrai reasoning_effort do body (compatível com OpenAI e Responses API)
    body_effort = None
    if "reasoning_effort" in request_body:
        body_effort = request_body["reasoning_effort"]
    elif "reasoning" in request_body and isinstance(request_body["reasoning"], dict):
        body_effort = request_body["reasoning"].get("effort")
    elif "reasoning" in request_body and isinstance(request_body["reasoning"], str):
        body_effort = request_body["reasoning"]
    elif "reasoning_effort" in request_args:
        body_effort = request_args["reasoning_effort"]

    if body_effort:
        eff_clean = str(body_effort).strip().lower().replace("_", "-")
        effective["reasoning_effort"] = eff_clean
        # Espectro de Reasoning Effort & Virtual Experts:
        # ultra / ultra-dynamic ativam Virtual Experts
        # low / medium / high / dynamic desativam por padrão (baseline neural pura)
        if eff_clean in ("ultra", "ultra-dynamic"):
            effective["virtual_experts"]["enabled"] = True
        elif eff_clean in ("low", "medium", "high", "dynamic") and not claims:
            effective["virtual_experts"]["enabled"] = False

    # Overrides via meta ou extra_body
    meta_dict = request_body.get("meta") or request_body.get("extra_body") or {}
    if isinstance(meta_dict, dict) and "virtual_experts" in meta_dict:
        ve_meta = meta_dict["virtual_experts"]
        if isinstance(ve_meta, dict):
            effective["virtual_experts"].update(ve_meta)
        elif isinstance(ve_meta, bool):
            effective["virtual_experts"]["enabled"] = ve_meta

    # Override direto via chave virtual_experts no body
    if "virtual_experts" in request_body:
        ve_body = request_body["virtual_experts"]
        if isinstance(ve_body, dict):
            effective["virtual_experts"].update(ve_body)
        elif isinstance(ve_body, bool):
            effective["virtual_experts"]["enabled"] = ve_body
        elif isinstance(ve_body, str):
            if ve_body.lower() in ("false", "0", "off", "no", "none"):
                effective["virtual_experts"]["enabled"] = False
            elif ve_body.lower() == "no_code":
                effective["virtual_experts"]["enabled"] = True
                effective["virtual_experts"]["python_repl"] = True
                effective["virtual_experts"]["code_interpreter"] = True
                effective["virtual_experts"]["llm_code_gen"] = False
                effective["virtual_experts"]["microtex_lean4"] = False
                effective["virtual_experts"]["llvm_jit"] = False
                effective["virtual_experts"]["lsp_language_server"] = False
            elif ve_body.lower() == "math_only":
                effective["virtual_experts"]["enabled"] = True
                for k in effective["virtual_experts"]:
                    if k != "enabled":
                        effective["virtual_experts"][k] = (k in ("python_repl", "microtex_lean4"))

    # Overrides via Query String
    q_ve = request_args.get("virtual_experts")
    if q_ve:
        if q_ve.lower() in ("false", "0", "off", "no", "none"):
            effective["virtual_experts"]["enabled"] = False
        elif q_ve.lower() == "no_code":
            effective["virtual_experts"]["enabled"] = True
            effective["virtual_experts"]["python_repl"] = True
            effective["virtual_experts"]["code_interpreter"] = True
            effective["virtual_experts"]["llm_code_gen"] = False
            effective["virtual_experts"]["microtex_lean4"] = False
            effective["virtual_experts"]["llvm_jit"] = False
            effective["virtual_experts"]["lsp_language_server"] = False
        elif q_ve.lower() == "math_only":
            effective["virtual_experts"]["enabled"] = True
            for k in effective["virtual_experts"]:
                if k != "enabled":
                    effective["virtual_experts"][k] = (k in ("python_repl", "microtex_lean4"))
        elif q_ve.lower() in ("true", "1", "on", "yes", "all"):
            effective["virtual_experts"]["enabled"] = True

    # Override de raw_output
    if "raw" in request_body or request_args.get("raw") in ("true", "1") or request_headers.get("X-Raw-Output") == "true":
        effective["raw_output"] = True

    return effective
