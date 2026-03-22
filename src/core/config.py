"""Configuration loader with environment variable interpolation.

Reads config/settings.yaml and expands ${VAR:-default} placeholders
from the process environment, then validates via Pydantic models.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


_ENV_PATTERN = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _interpolate(value: Any) -> Any:
    """Recursively replace ${VAR:-default} in strings."""
    if isinstance(value, str):
        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default)
        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


# ── Pydantic settings models ─────────────────────────────────────────

class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "ollama"
    model: str = "llama3.1:8b"
    temperature: float = 0.0
    max_tokens: int = 2048
    request_timeout: int = 120


class EmbeddingConfig(BaseModel):
    """Sentence-transformer embedding settings."""

    model: str = "all-MiniLM-L6-v2"
    dimension: int = 384


class DatabaseConfig(BaseModel):
    """SQLite database connection settings."""

    path: str = "data/fintech.db"
    read_only: bool = True
    max_rows_returned: int = 500
    query_timeout_seconds: int = 30


class FAISSConfig(BaseModel):
    """FAISS index paths and retrieval settings."""

    index_path: str = "data/schema.faiss"
    metadata_path: str = "data/schema_meta.pkl"
    top_k: int = 5


class RolePermissions(BaseModel):
    """Per-role access control definition."""

    allowed_tables: list[str] = Field(default_factory=lambda: ["*"])
    denied_tables: list[str] = Field(default_factory=list)
    max_rows: int = 1000
    can_write: bool = False


class AccessControlConfig(BaseModel):
    """Role-based access control settings."""

    roles: dict[str, RolePermissions] = Field(default_factory=dict)


class ValidationConfig(BaseModel):
    """SQL validation guard-rails."""

    blocked_keywords: list[str] = Field(default_factory=list)
    max_joins: int = 4
    max_subqueries: int = 3
    warn_full_scan_threshold: int = 10000


class OrchestratorConfig(BaseModel):
    """Pipeline orchestration settings."""

    max_retries: int = 2
    parallel_timeout: int = 30


class LoggingConfig(BaseModel):
    """Structured logging settings."""

    level: str = "INFO"
    format: str = "json"


class Settings(BaseModel):
    """Top-level application settings."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    access_control: AccessControlConfig = Field(default_factory=AccessControlConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ── Singleton loader ──────────────────────────────────────────────────

_settings: Settings | None = None


def load_settings(path: Path | None = None) -> Settings:
    """Load and cache application settings from YAML.

    Args:
        path: Explicit path to settings.yaml. Defaults to config/settings.yaml.

    Returns:
        Validated Settings instance.
    """
    global _settings
    if _settings is not None:
        return _settings

    if path is None:
        path = _PROJECT_ROOT / "config" / "settings.yaml"

    raw: dict[str, Any] = {}
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    raw = _interpolate(raw)
    _settings = Settings(**raw)
    return _settings


def reset_settings() -> None:
    """Clear cached settings (useful in tests)."""
    global _settings
    _settings = None
