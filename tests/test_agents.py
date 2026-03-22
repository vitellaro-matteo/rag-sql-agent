"""Integration tests for agents using mocked LLM responses.

These tests verify the full agent logic — prompt rendering, LLM call,
response parsing, and context mutation — without requiring an API key.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.explainer import ExplainerAgent
from src.agents.router import QueryRouterAgent
from src.agents.sql_generator import SQLGeneratorAgent
from src.core.logging import AgentTrace
from tests.conftest import MOCK_RESPONSES


def _make_mock(key: str) -> AsyncMock:
    """Create a mock that returns the JSON for a specific agent."""

    async def _mock(system: str, user: str, **kwargs: Any) -> str:
        return json.dumps(MOCK_RESPONSES[key])

    return AsyncMock(side_effect=_mock)


@pytest.mark.asyncio
class TestRouterAgent:
    """Tests for QueryRouterAgent."""

    async def test_classifies_sql_query(self, mock_llm: AsyncMock) -> None:
        """Router should classify a data question as sql_query."""
        agent = QueryRouterAgent()
        trace = AgentTrace()
        context = {
            "question": "Show me the top 10 transactions",
            "available_tables": ["transactions", "accounts", "merchants"],
        }
        result = await agent.run(context, trace)

        assert result["intent"] == "sql_query"
        assert result["complexity"] in ("simple", "moderate", "complex")
        assert len(trace.events) >= 2  # classifying + classified

    async def test_trace_recorded(self, mock_llm: AsyncMock) -> None:
        """Router should record both start and end trace events."""
        agent = QueryRouterAgent()
        trace = AgentTrace()
        context = {
            "question": "Hello",
            "available_tables": [],
        }
        await agent.run(context, trace)
        actions = [e["action"] for e in trace.events]
        assert "classifying" in actions
        assert "classified" in actions


@pytest.mark.asyncio
class TestSQLGeneratorAgent:
    """Tests for SQLGeneratorAgent."""

    async def test_generates_sql(self, mock_llm: AsyncMock) -> None:
        """Generator should produce SQL and chain-of-thought."""
        agent = SQLGeneratorAgent()
        trace = AgentTrace()
        context = {
            "question": "Show top 10 transactions",
            "schema_context": "TABLE transactions (amount REAL, created_at DATETIME)",
            "allowed_tables": ["transactions"],
            "max_rows": 100,
        }
        result = await agent.run(context, trace)

        assert "generated_sql" in result
        assert "SELECT" in result["generated_sql"].upper()
        assert "chain_of_thought" in result
        assert result["sql_confidence"] > 0

    async def test_adds_limit_if_missing(self, mock_llm: AsyncMock) -> None:
        """Generator should append LIMIT if the LLM omits it."""
        # Override mock to return SQL without LIMIT
        no_limit_response = {**MOCK_RESPONSES["sql_generator"]}
        no_limit_response["sql"] = "SELECT amount FROM transactions ORDER BY amount DESC"

        async def _mock(system: str, user: str, **kwargs: Any) -> str:
            return json.dumps(no_limit_response)

        with patch("src.agents.base.complete", AsyncMock(side_effect=_mock)):
            agent = SQLGeneratorAgent()
            trace = AgentTrace()
            context = {
                "question": "Show all transactions",
                "schema_context": "TABLE transactions",
                "allowed_tables": ["transactions"],
                "max_rows": 200,
            }
            result = await agent.run(context, trace)
            assert "LIMIT" in result["generated_sql"].upper()


@pytest.mark.asyncio
class TestExplainerAgent:
    """Tests for ExplainerAgent."""

    async def test_explains_results(self, mock_llm: AsyncMock) -> None:
        """Explainer should produce a summary and highlights."""
        agent = ExplainerAgent()
        trace = AgentTrace()
        context = {
            "question": "Show top transactions",
            "generated_sql": "SELECT * FROM transactions LIMIT 5",
            "query_results": [
                {"transaction_id": 1, "amount": -2450.0},
                {"transaction_id": 2, "amount": -1200.0},
            ],
        }
        result = await agent.run(context, trace)

        assert "explanation" in result
        assert len(result["explanation"]) > 0

    async def test_fallback_on_empty_results(self, mock_llm: AsyncMock) -> None:
        """Explainer should handle empty results gracefully."""
        agent = ExplainerAgent()
        trace = AgentTrace()
        context = {
            "question": "Show transactions for 2099",
            "generated_sql": "SELECT * FROM transactions WHERE created_at > '2099-01-01'",
            "query_results": [],
        }
        result = await agent.run(context, trace)
        assert "explanation" in result

    def test_format_results_truncation(self) -> None:
        """Formatter should truncate long result sets."""
        agent = ExplainerAgent()
        rows = [{"id": i, "val": f"row_{i}"} for i in range(50)]
        formatted = agent._format_results(rows, max_display=5)
        assert "more rows" in formatted

    def test_fallback_explanation(self) -> None:
        """Fallback should mention row count and columns."""
        result = ExplainerAgent._fallback_explanation(
            "test query",
            [{"col_a": 1, "col_b": "x"}],
        )
        assert "1 row" in result
        assert "col_a" in result
