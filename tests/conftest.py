"""Shared fixtures: mocked LLM, temp database, schema store.

All tests run without an API key by patching the LLM layer with
deterministic mock responses.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from src.core.config import Settings, reset_settings

# ── Mock LLM responses keyed by agent ────────────────────────────────

MOCK_RESPONSES: dict[str, dict[str, Any]] = {
    "router": {
        "intent": "sql_query",
        "complexity": "simple",
        "tables_mentioned": ["transactions"],
        "reasoning": "User asks for transaction data.",
    },
    "schema_rag": {
        "relevant_tables": [
            {
                "table": "transactions",
                "columns": ["transaction_id (INTEGER)", "amount (REAL)", "created_at (DATETIME)"],
                "notes": "Core transaction table.",
            }
        ],
        "joins": [],
        "warnings": [],
    },
    "sql_generator": {
        "chain_of_thought": {
            "understand": "User wants the 10 largest transactions.",
            "plan": "Single table query on transactions, ORDER BY amount DESC.",
            "construct": "SELECT columns FROM transactions ORDER BY amount DESC LIMIT 10",
            "verify": "No joins needed, LIMIT present.",
        },
        "sql": "SELECT transaction_id, amount, created_at FROM transactions ORDER BY amount DESC LIMIT 10;",
        "confidence": 0.92,
    },
    "validation": {
        "verdict": "safe",
        "issues": [],
        "estimated_cost": "low",
        "reasoning": "Simple single-table SELECT with LIMIT.",
    },
    "explainer": {
        "summary": "Here are the 10 largest transactions by amount.",
        "highlights": ["Largest: $2,450.00", "All completed"],
        "follow_up_suggestions": ["Show fraud-flagged transactions"],
    },
}


def _mock_complete_factory(agent_key: str | None = None) -> AsyncMock:
    """Create a mock for ``src.core.llm.complete`` that returns JSON strings."""

    async def _mock(system: str, user: str, **kwargs: Any) -> str:
        # Detect which agent is calling based on system prompt keywords
        # Order matters: check more specific patterns first
        key = agent_key
        if key is None:
            lower = system.lower()
            if "routing" in lower or "classify" in lower:
                key = "router"
            elif "sql engineer" in lower or "sql" in lower and "generate" in lower:
                key = "sql_generator"
            elif "auditor" in lower or "security" in lower:
                key = "validation"
            elif "communication" in lower or "translate" in lower:
                key = "explainer"
            elif "schema" in lower:
                key = "schema_rag"
            else:
                key = "router"
        return json.dumps(MOCK_RESPONSES.get(key, MOCK_RESPONSES["router"]))

    return AsyncMock(side_effect=_mock)


@pytest.fixture(autouse=True)
def _reset_config() -> Generator[None, None, None]:
    """Reset cached settings between tests."""
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def mock_llm() -> Generator[AsyncMock, None, None]:
    """Patch the LLM complete function globally."""
    mock = _mock_complete_factory()
    with patch("src.core.llm.complete", mock):
        with patch("src.agents.base.complete", mock):
            yield mock


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Create a minimal test database."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY,
            email TEXT NOT NULL,
            full_name TEXT NOT NULL,
            country TEXT NOT NULL,
            kyc_status TEXT DEFAULT 'verified'
        );
        CREATE TABLE accounts (
            account_id INTEGER PRIMARY KEY,
            user_id INTEGER REFERENCES users(user_id),
            account_type TEXT DEFAULT 'checking',
            balance REAL DEFAULT 0.0
        );
        CREATE TABLE merchants (
            merchant_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL
        );
        CREATE TABLE transactions (
            transaction_id INTEGER PRIMARY KEY,
            account_id INTEGER REFERENCES accounts(account_id),
            merchant_id INTEGER REFERENCES merchants(merchant_id),
            amount REAL NOT NULL,
            transaction_type TEXT DEFAULT 'purchase',
            status TEXT DEFAULT 'completed',
            fraud_flag INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT (datetime('now'))
        );

        INSERT INTO users VALUES (1, 'alice@test.com', 'Alice Test', 'US', 'verified');
        INSERT INTO users VALUES (2, 'bob@test.com', 'Bob Test', 'DE', 'pending');
        INSERT INTO accounts VALUES (1, 1, 'checking', 5000.00);
        INSERT INTO accounts VALUES (2, 2, 'savings', 12000.00);
        INSERT INTO merchants VALUES (1, 'TestMart', 'Groceries');
        INSERT INTO merchants VALUES (2, 'CloudCo', 'Software & SaaS');

        INSERT INTO transactions VALUES (1, 1, 1, -45.99, 'purchase', 'completed', 0, '2024-06-01');
        INSERT INTO transactions VALUES (2, 1, 2, -120.00, 'purchase', 'completed', 0, '2024-06-15');
        INSERT INTO transactions VALUES (3, 2, NULL, 3000.00, 'deposit', 'completed', 0, '2024-07-01');
        INSERT INTO transactions VALUES (4, 1, 1, -22.50, 'purchase', 'completed', 1, '2024-07-10');
        INSERT INTO transactions VALUES (5, 2, NULL, -500.00, 'transfer', 'pending', 0, '2024-07-15');
    """)
    conn.commit()
    conn.close()
    return db_path
