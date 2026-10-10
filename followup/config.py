"""Central configuration, read from environment variables (or a local .env file).

Secrets (LLM API keys) are only ever read from the environment — on EC2 they are
injected from AWS Secrets Manager or the instance environment, never committed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # LLM: "openai", "gemini" or "mock" (offline, deterministic — no key needed).
    # "auto" picks openai/gemini when a key is present, otherwise mock.
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto").lower())
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    openai_embedding_model: str = field(
        default_factory=lambda: _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    google_api_key: str = field(default_factory=lambda: _env("GOOGLE_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-1.5-flash"))

    # Storage: "local" (folder that mirrors the S3 layout) or "s3".
    storage_backend: str = field(default_factory=lambda: _env("STORAGE_BACKEND", "local").lower())
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET"))
    s3_prefix: str = field(default_factory=lambda: _env("S3_PREFIX", "followup"))
    aws_region: str = field(default_factory=lambda: _env("AWS_REGION", "us-east-1"))
    local_storage_dir: Path = field(
        default_factory=lambda: Path(_env("LOCAL_STORAGE_DIR", str(PROJECT_ROOT / "storage")))
    )

    # Vector store: "chroma" (persistent) or "memory" (in-process).
    vector_backend: str = field(default_factory=lambda: _env("VECTOR_BACKEND", "chroma").lower())
    chroma_dir: Path = field(
        default_factory=lambda: Path(_env("CHROMA_DIR", str(PROJECT_ROOT / "vectorstore")))
    )

    # Business rules
    duplicate_reminder_window_days: int = field(
        default_factory=lambda: int(_env("DUPLICATE_REMINDER_WINDOW_DAYS", "2"))
    )
    upcoming_window_days: int = field(default_factory=lambda: int(_env("UPCOMING_WINDOW_DAYS", "7")))

    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.openai_api_key:
            return "openai"
        if self.google_api_key:
            return "gemini"
        return "mock"


settings = Settings()
