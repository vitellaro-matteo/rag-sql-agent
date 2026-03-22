"""Integration tests for the full pipeline orchestrator.

These tests run the complete agent chain with mocked LLM responses
and a real (temporary) SQLite database to verify end-to-end behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.orchestrator import PipelineOrchestrator
from src.core.logging import AgentTrace
from tests.conftest import MOCK_RESPONSES


def _make_sequential_mock() -> AsyncMock:
    """Mock that returns the right JSON based on system prompt content."""

    async def _mock(system: str, user: str, **kwargs: Any) -> str:
        lower = system.lower()
        if "routing" in lower or "classify" in lower:
            return json.dumps(MOCK_RESPONSES["router"])
        if "schema" in lower and "analysis" in lower:
            return json.dumps(MOCK_RESPONSES["schema_rag"])
        if "sql engineer" in lower:
            return json.dumps(MOCK_RESPONSES["sql_generator"])
        if "auditor" in lower or "security" in lower:
            return json.dumps(MOCK_RESPONSES["validation"])
        if "communication" in lower or "translate" in lower:
            return json.dumps(MOCK_RESPONSES["explainer"])
        return json.dumps(MOCK_RESPONSES["router"])

    return AsyncMock(side_effect=_mock)


@pytest.mark.asyncio
class TestPipelineOrchestrator:
    """End-to-end pipeline tests with mocked LLM."""

    async def test_full_pipeline(self, tmp_db: Path) -> None:
        """Complete pipeline should produce an explanation and trace."""
        mock = _make_sequential_mock()

        # Mock the schema store so we don't need FAISS
        mock_store = MagicMock()
        mock_store.load = MagicMock()
        mock_store.query = MagicMock(return_value=[
            {"text": "TABLE transactions (amount REAL)", "table": "transactions", "score": 0.95}
        ])
        mock_store.format_context = MagicMock(
            return_value="[1] (table: transactions)\nTABLE transactions (amount REAL)"
        )

        with patch("src.core.llm.complete", mock), \
             patch("src.agents.base.complete", mock):
            orch = PipelineOrchestrator()
            orch._db._path = tmp_db
            orch._store = mock_store
            orch._schema_rag._store = mock_store

            await orch.initialize()

            trace = AgentTrace()
            result = await orch.process(
                "Show me the top 10 transactions by amount",
                role="analyst",
                trace=trace,
            )

            # Should have an explanation
            assert "explanation" in result
            assert len(result["explanation"]) > 0

            # Trace should have multiple events
            assert len(trace.events) >= 5

            # Should have generated SQL
            assert "generated_sql" in result

            # Validation should have passed
            assert result.get("validation_verdict") == "safe"

            await orch.shutdown()

    async def test_blocked_query(self, tmp_db: Path) -> None:
        """Pipeline should block dangerous SQL."""
        # Make the SQL generator return a DROP statement
        bad_sql_response = {**MOCK_RESPONSES["sql_generator"]}
        bad_sql_response["sql"] = "DROP TABLE transactions;"

        async def _mock(system: str, user: str, **kwargs: Any) -> str:
            lower = system.lower()
            if "routing" in lower or "classify" in lower:
                return json.dumps(MOCK_RESPONSES["router"])
            if "sql engineer" in lower:
                return json.dumps(bad_sql_response)
            if "auditor" in lower or "security" in lower:
                return json.dumps({
                    "verdict": "blocked",
                    "issues": [{"severity": "critical", "category": "safety",
                                "description": "DROP detected", "suggestion": "Don't."}],
                    "estimated_cost": "unknown",
                    "reasoning": "DROP is destructive.",
                })
            if "communication" in lower:
                return json.dumps(MOCK_RESPONSES["explainer"])
            if "schema" in lower:
                return json.dumps(MOCK_RESPONSES["schema_rag"])
            return json.dumps(MOCK_RESPONSES["router"])

        mock = AsyncMock(side_effect=_mock)
        mock_store = MagicMock()
        mock_store.load = MagicMock()
        mock_store.query = MagicMock(return_value=[])
        mock_store.format_context = MagicMock(return_value="")

        with patch("src.core.llm.complete", mock), \
             patch("src.agents.base.complete", mock):
            orch = PipelineOrchestrator()
            orch._db._path = tmp_db
            orch._store = mock_store
            orch._schema_rag._store = mock_store
            await orch.initialize()

            trace = AgentTrace()
            result = await orch.process("DROP TABLE transactions", role="analyst", trace=trace)

            # Should be blocked — no query_results
            assert "blocked" in result.get("explanation", "").lower() or \
                   result.get("validation_verdict") == "blocked"

            await orch.shutdown()

    async def test_greeting_intent(self, tmp_db: Path) -> None:
        """Greetings should short-circuit without SQL generation."""
        greeting_response = {
            "intent": "greeting",
            "complexity": "simple",
            "tables_mentioned": [],
            "reasoning": "User said hello.",
        }

        async def _mock(system: str, user: str, **kwargs: Any) -> str:
            return json.dumps(greeting_response)

        mock = AsyncMock(side_effect=_mock)
        mock_store = MagicMock()
        mock_store.load = MagicMock()

        with patch("src.core.llm.complete", mock), \
             patch("src.agents.base.complete", mock):
            orch = PipelineOrchestrator()
            orch._db._path = tmp_db
            orch._store = mock_store
            orch._schema_rag._store = mock_store
            await orch.initialize()

            trace = AgentTrace()
            result = await orch.process("Hello!", role="analyst", trace=trace)

            assert "Hello" in result.get("explanation", "") or "SQL assistant" in result.get("explanation", "")
            assert "generated_sql" not in result

            await orch.shutdown()

    async def test_access_control_filters_tables(self, tmp_db: Path) -> None:
        """Analyst role should not see 'users' in allowed_tables."""
        mock = _make_sequential_mock()
        mock_store = MagicMock()
        mock_store.load = MagicMock()
        mock_store.query = MagicMock(return_value=[])
        mock_store.format_context = MagicMock(return_value="")

        with patch("src.core.llm.complete", mock), \
             patch("src.agents.base.complete", mock):
            orch = PipelineOrchestrator()
            orch._db._path = tmp_db
            orch._store = mock_store
            orch._schema_rag._store = mock_store
            await orch.initialize()

            trace = AgentTrace()
            result = await orch.process(
                "Show me user emails",
                role="analyst",
                trace=trace,
            )

            # Access control should have filtered out 'users'
            assert "users" not in result.get("allowed_tables", [])

            await orch.shutdown()
