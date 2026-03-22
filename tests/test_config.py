"""Tests for config loading and environment variable interpolation."""

from __future__ import annotations

import os
from pathlib import Path

from src.core.config import Settings, load_settings, reset_settings


class TestSettings:
    """Test suite for configuration loading."""

    def test_defaults_without_file(self, tmp_path: Path) -> None:
        """Settings should use defaults when YAML doesn't exist."""
        reset_settings()
        settings = load_settings(path=tmp_path / "nonexistent.yaml")
        assert settings.llm.provider == "ollama"
        assert settings.database.read_only is True
        assert "admin" not in settings.access_control.roles

    def test_loads_yaml(self) -> None:
        """Settings should load from the repo config/settings.yaml."""
        reset_settings()
        settings = load_settings()
        assert settings.llm.temperature == 0.0
        assert settings.validation.max_joins == 4
        assert "analyst" in settings.access_control.roles

    def test_env_override(self, monkeypatch: object) -> None:
        """Environment variables should override YAML defaults."""
        reset_settings()
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_MODEL"] = "gpt-4o"
        try:
            settings = load_settings()
            assert settings.llm.provider == "openai"
            assert settings.llm.model == "gpt-4o"
        finally:
            del os.environ["LLM_PROVIDER"]
            del os.environ["LLM_MODEL"]
            reset_settings()

    def test_access_control_roles(self) -> None:
        """Role definitions should parse correctly."""
        reset_settings()
        settings = load_settings()
        analyst = settings.access_control.roles.get("analyst")
        assert analyst is not None
        assert "users" in analyst.denied_tables
        assert analyst.can_write is False
        assert analyst.max_rows == 1000
