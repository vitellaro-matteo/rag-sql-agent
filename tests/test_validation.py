"""Tests for the SQL validation agent (deterministic checks)."""

from __future__ import annotations

import pytest

from src.agents.validation import ValidationAgent
from src.core.logging import AgentTrace


class TestValidationDeterministic:
    """Test the rule-based validation pass (no LLM needed)."""

    def setup_method(self) -> None:
        """Fresh agent per test."""
        self.agent = ValidationAgent()

    def test_blocks_drop(self) -> None:
        """DROP should be blocked."""
        issues = self.agent._deterministic_check(
            "DROP TABLE transactions",
            {"denied_tables": []},
        )
        assert any(i["severity"] == "critical" for i in issues)
        assert any("DROP" in i["description"] for i in issues)

    def test_blocks_delete(self) -> None:
        """DELETE should be blocked."""
        issues = self.agent._deterministic_check(
            "DELETE FROM transactions WHERE 1=1",
            {"denied_tables": []},
        )
        assert any(i["severity"] == "critical" for i in issues)

    def test_warns_on_excessive_joins(self) -> None:
        """Too many JOINs should produce a warning."""
        sql = (
            "SELECT * FROM a "
            "JOIN b ON a.id=b.id "
            "JOIN c ON b.id=c.id "
            "JOIN d ON c.id=d.id "
            "JOIN e ON d.id=e.id "
            "JOIN f ON e.id=f.id"
        )
        issues = self.agent._deterministic_check(sql, {"denied_tables": []})
        assert any(i["category"] == "performance" for i in issues)

    def test_detects_denied_table(self) -> None:
        """SQL with denied tables should produce a critical issue."""
        sql = "SELECT email FROM users"
        issues = self.agent._deterministic_check(sql, {"denied_tables": ["users"]})
        assert any(i["category"] == "access" for i in issues)

    def test_detects_injection_pattern(self) -> None:
        """Tautology pattern should be flagged."""
        sql = "SELECT * FROM transactions WHERE 1=1"
        issues = self.agent._deterministic_check(sql, {"denied_tables": []})
        assert any(i["category"] == "injection" for i in issues)

    def test_clean_query_passes(self) -> None:
        """A normal SELECT should have no critical issues."""
        sql = "SELECT amount, created_at FROM transactions WHERE amount > 100 LIMIT 50"
        issues = self.agent._deterministic_check(sql, {"denied_tables": []})
        critical = [i for i in issues if i["severity"] == "critical"]
        assert len(critical) == 0

    def test_verdict_logic(self) -> None:
        """Verdict computation should follow severity hierarchy."""
        assert self.agent._compute_verdict([]) == "safe"
        assert self.agent._compute_verdict([{"severity": "info"}]) == "safe"
        assert self.agent._compute_verdict([{"severity": "warning"}]) == "warning"
        assert self.agent._compute_verdict([{"severity": "critical"}]) == "blocked"
        assert self.agent._compute_verdict([
            {"severity": "warning"},
            {"severity": "critical"},
        ]) == "blocked"
