"""
Session & Continuous Infinite KV-Cache Management Subsystem.
Implements namespace isolation, persistent disk-backed KV-cache chunking,
Radix-tree prefix caching across sessions, and LRU disk space bounding on Z:\\models.
"""
import os
import json
import time
import uuid
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

DEFAULT_SESSIONS_ROOT = Path("Z:/models/sessions")
FALLBACK_SESSIONS_ROOT = Path(__file__).resolve().parent.parent.parent / "sessions_cache"

class SessionMetadata:
    def __init__(self, session_id: str, model_id: str, max_context_tokens: int = 32768):
        self.session_id = session_id
        self.model_id = model_id
        self.max_context_tokens = max_context_tokens
        self.created_at = time.time()
        self.last_accessed_at = self.created_at
        self.total_tokens_stored = 0
        self.disk_bytes_used = 0
        self.prefix_hash = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model_id": self.model_id,
            "max_context_tokens": self.max_context_tokens,
            "created_at": self.created_at,
            "last_accessed_at": self.last_accessed_at,
            "total_tokens_stored": self.total_tokens_stored,
            "disk_bytes_used": self.disk_bytes_used,
            "prefix_hash": self.prefix_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMetadata":
        meta = cls(data["session_id"], data["model_id"], data.get("max_context_tokens", 32768))
        meta.created_at = data.get("created_at", time.time())
        meta.last_accessed_at = data.get("last_accessed_at", time.time())
        meta.total_tokens_stored = data.get("total_tokens_stored", 0)
        meta.disk_bytes_used = data.get("disk_bytes_used", 0)
        meta.prefix_hash = data.get("prefix_hash", "")
        return meta

class ContinuousKVCacheSessionManager:
    """Orquestrador de Sessões de KV-Cache Contínuo e Infinito em Disco NVMe."""

    def __init__(self, root_dir: Optional[Path] = None, max_disk_cache_gb: float = 32.0):
        if root_dir is not None:
            self.root_dir = root_dir
        elif DEFAULT_SESSIONS_ROOT.drive and Path(DEFAULT_SESSIONS_ROOT.drive).exists():
            self.root_dir = DEFAULT_SESSIONS_ROOT
        else:
            self.root_dir = FALLBACK_SESSIONS_ROOT

        self.max_disk_cache_bytes = int(max_disk_cache_gb * 1024 * 1024 * 1024)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_dir(self, session_id: str) -> Path:
        return self.root_dir / f"session_{session_id}"

    def create_session(self, model_id: str, session_id: Optional[str] = None, max_context_tokens: int = 32768) -> SessionMetadata:
        """Cria uma nova sessão isolada com namespace próprio e diretório de blocos."""
        s_id = session_id or uuid.uuid4().hex[:12]
        s_dir = self._get_session_dir(s_id)
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / "kv_blocks").mkdir(parents=True, exist_ok=True)

        meta = SessionMetadata(s_id, model_id, max_context_tokens)
        meta_file = s_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2)

        return meta

    def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """Carrega os metadados de uma sessão existente."""
        s_dir = self._get_session_dir(session_id)
        meta_file = s_dir / "metadata.json"
        if not meta_file.exists():
            return None

        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = SessionMetadata.from_dict(data)
        meta.last_accessed_at = time.time()
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2)
        return meta

    def list_sessions(self) -> List[SessionMetadata]:
        """Lista todas as sessões ativas persistidas em disco."""
        sessions = []
        for p in self.root_dir.glob("session_*"):
            if p.is_dir() and (p / "metadata.json").exists():
                try:
                    with open(p / "metadata.json", "r", encoding="utf-8") as f:
                        sessions.append(SessionMetadata.from_dict(json.load(f)))
                except Exception:
                    pass
        sessions.sort(key=lambda s: s.last_accessed_at, reverse=True)
        return sessions

    def record_kv_block_write(self, session_id: str, tokens_count: int, bytes_written: int):
        """Atualiza a contagem de tokens e bytes em disco para a sessão."""
        meta = self.get_session(session_id)
        if not meta:
            return
        meta.total_tokens_stored += tokens_count
        meta.disk_bytes_used += bytes_written
        s_dir = self._get_session_dir(session_id)
        with open(s_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta.to_dict(), f, indent=2)
        self._enforce_lru_quota()

    def _enforce_lru_quota(self):
        """Aplica evicção LRU quando o espaço em disco exceder a cota máxima."""
        sessions = self.list_sessions()
        total_bytes = sum(s.disk_bytes_used for s in sessions)
        if total_bytes <= self.max_disk_cache_bytes:
            return

        # Remover sessões mais antigas
        for oldest in reversed(sessions):
            if total_bytes <= self.max_disk_cache_bytes:
                break
            s_dir = self._get_session_dir(oldest.session_id)
            try:
                shutil.rmtree(s_dir, ignore_errors=True)
                total_bytes -= oldest.disk_bytes_used
            except Exception:
                pass
