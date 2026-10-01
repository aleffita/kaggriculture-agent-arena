"""High-level runner and orchestration around LiteRT-LM Engine."""

from __future__ import annotations

import os
import pathlib
import threading
from typing import Generator, Optional

try:
    import litert_lm
    from litert_lm import interfaces
except ImportError:
    litert_lm = None
    interfaces = None

from .gpu import select_gpu

DEFAULT_MODEL_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
DEFAULT_MODEL_FILE = "gemma-4-E2B-it.litertlm"

def get_default_cache_dir() -> pathlib.Path:
    """Returns local cache path for downloaded litertlm models."""
    cache = pathlib.Path.home() / ".litert-lm" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache

def resolve_model_path(
    model_path: Optional[str] = None,
    repo_id: str = DEFAULT_MODEL_REPO,
    filename: str = DEFAULT_MODEL_FILE,
) -> str:
    """Resolves local model file path or downloads it from Hugging Face Hub."""
    if model_path and os.path.exists(model_path):
        return str(pathlib.Path(model_path).resolve())

    # Check LiteRT-LM default cache
    candidate = (
        pathlib.Path.home()
        / ".litert-lm"
        / "cache"
        / "huggingface"
        / repo_id
        / filename
    )
    if candidate.exists():
        return str(candidate)

    # Use huggingface_hub to download if available
    from huggingface_hub import hf_hub_download
    print(f"Baixando modelo {filename} de {repo_id}...")
    downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
    return downloaded


class LiteRtModelRunner:
    """Wraps LiteRT-LM model execution with granular backend and GPU routing."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "gpu",
        gpu_target: Optional[str | int] = None,
        max_num_tokens: int = 2048,
    ):
        self.model_path = resolve_model_path(model_path)
        self.backend_str = backend.lower()
        self.gpu_target = gpu_target
        self.max_num_tokens = max_num_tokens
        self.engine = None
        self._lock = threading.Lock()
        self._init_engine()

    def _init_engine(self):
        if litert_lm and hasattr(litert_lm, "set_min_log_severity"):
            try:
                litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
            except Exception:
                pass

        if self.backend_str == "cpu":
            backend_obj = interfaces.CPU()
            self.engine = litert_lm.Engine(
                model_path=self.model_path,
                backend=backend_obj,
                max_num_tokens=self.max_num_tokens,
                max_num_images=0,
            )
        else:
            # GPU backend
            target = self.gpu_target if self.gpu_target is not None else 0
            with select_gpu(target):
                backend_obj = interfaces.GPU()
                self.engine = litert_lm.Engine(
                    model_path=self.model_path,
                    backend=backend_obj,
                    max_num_tokens=self.max_num_tokens,
                    max_num_images=0,
                )

    def generate(self, prompt: str) -> str:
        """Executes one-shot text generation, returning a decoded string."""
        with self._lock:
            conversation = self.engine.create_conversation()
            response = conversation.send_message(prompt)
            return self._extract_text(response)

    def generate_structured(self, prompt: str, schema: dict) -> str:
        """Executes one-shot constrained structured JSON decoding using LL_GUIDANCE."""
        with self._lock:
            cdc = litert_lm.ConstrainedDecodingConfig(
                enable=True,
                provider=litert_lm.LiteRtLmConstraintProviderType.LL_GUIDANCE
            )
            rf = litert_lm.ResponseFormat.json(schema)
            conversation = self.engine.create_conversation(constrained_decoding_config=cdc)
            response = conversation.send_message(prompt, response_format=rf)
            return self._extract_text(response)

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Streams generation token by token."""
        conversation = self.engine.create_conversation()
        for chunk in conversation.send_message_async(prompt):
            text = self._extract_text(chunk)
            if text:
                yield text

    @staticmethod
    def _extract_text(resp: Any) -> str:
        """Helper to extract raw text content from LiteRT-LM response structures."""
        if isinstance(resp, str):
            return resp
        if isinstance(resp, dict):
            # Check 'content' list
            content = resp.get("content")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        parts.append(item["text"])
                    elif isinstance(item, str):
                        parts.append(item)
                if parts:
                    return "".join(parts)
            elif isinstance(content, str):
                return content

            # Check 'candidates' list
            if "candidates" in resp:
                candidates = resp["candidates"]
                if isinstance(candidates, list) and candidates:
                    cand = candidates[0]
                    if isinstance(cand, dict):
                        cand_content = cand.get("content")
                        if isinstance(cand_content, dict):
                            parts = cand_content.get("parts", [])
                            return "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                        elif isinstance(cand_content, str):
                            return cand_content

            if "text" in resp:
                return str(resp["text"])
            return json.dumps(resp)
        return str(resp)

