"""Tests for the LLM abstraction layer and JSON parsing."""

from __future__ import annotations

import pytest

from src.core.llm import parse_json_response, load_prompt


class TestParseJsonResponse:
    """Test JSON extraction from various LLM output formats."""

    def test_clean_json(self) -> None:
        """Should parse clean JSON directly."""
        result = parse_json_response('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_markdown_fenced(self) -> None:
        """Should strip markdown code fences."""
        raw = '```json\n{"key": "value"}\n```'
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_json_with_preamble(self) -> None:
        """Should extract JSON from text with surrounding prose."""
        raw = 'Here is the result:\n\n{"intent": "sql_query", "complexity": "simple"}\n\nHope that helps!'
        result = parse_json_response(raw)
        assert result["intent"] == "sql_query"

    def test_nested_json(self) -> None:
        """Should handle nested structures."""
        raw = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        result = parse_json_response(raw)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_invalid_json_raises(self) -> None:
        """Should raise ValueError for unparseable content."""
        with pytest.raises(ValueError, match="Could not parse JSON"):
            parse_json_response("This is just plain text with no JSON at all.")

    def test_empty_string_raises(self) -> None:
        """Should raise ValueError for empty input."""
        with pytest.raises(ValueError):
            parse_json_response("")

    def test_markdown_fence_without_lang(self) -> None:
        """Should handle fences without language annotation."""
        raw = '```\n{"result": true}\n```'
        result = parse_json_response(raw)
        assert result["result"] is True


class TestLoadPrompt:
    """Test prompt template loading."""

    def test_loads_router_prompt(self) -> None:
        """Should load the router prompt with system and user keys."""
        prompt = load_prompt("router")
        assert "system" in prompt
        assert "user" in prompt
        assert "{question}" in prompt["user"]

    def test_loads_all_prompts(self) -> None:
        """All prompt files should load without error."""
        for name in ("router", "schema_rag", "sql_generator", "validation", "explainer"):
            prompt = load_prompt(name)
            assert "system" in prompt
            assert "user" in prompt

    def test_missing_prompt_raises(self) -> None:
        """Should raise FileNotFoundError for nonexistent prompt."""
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt_xyz")
