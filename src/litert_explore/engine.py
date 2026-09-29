"""High-level runner and orchestration around LiteRT-LM Engine."""

from __future__ import annotations

import os
import pathlib
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
        self._init_engine()

    def _init_engine(self):
        if self.backend_str == "cpu":
            backend_obj = interfaces.CPU()
            self.engine = litert_lm.Engine(
                model_path=self.model_path,
                backend=backend_obj,
                max_num_tokens=self.max_num_tokens,
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
                )

    def generate(self, prompt: str) -> str:
        """Executes one-shot text generation."""
        conversation = self.engine.create_conversation()
        response = conversation.send_message(prompt)
        return response

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """Streams generation token by token."""
        conversation = self.engine.create_conversation()
        # LiteRT-LM conversation.send_message_stream
        for chunk in conversation.send_message_stream(prompt):
            yield chunk
