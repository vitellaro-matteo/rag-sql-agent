"""Tests for role-based access control."""

from __future__ import annotations

from src.agents.access_control import AccessControlLayer


ALL_TABLES = ["users", "accounts", "merchants", "transactions"]


class TestAccessControl:
    """Test suite for the AccessControlLayer."""

    def test_admin_sees_everything(self) -> None:
        """Admin role should access all tables."""
        acl = AccessControlLayer()
        allowed = acl.get_allowed_tables("admin", ALL_TABLES)
        assert set(allowed) == set(ALL_TABLES)

    def test_analyst_denied_users(self) -> None:
        """Analyst should not see the users table."""
        acl = AccessControlLayer()
        allowed = acl.get_allowed_tables("analyst", ALL_TABLES)
        assert "users" not in allowed
        assert "transactions" in allowed

    def test_viewer_limited(self) -> None:
        """Viewer should only see transactions and merchants."""
        acl = AccessControlLayer()
        allowed = acl.get_allowed_tables("viewer", ALL_TABLES)
        assert set(allowed) == {"transactions", "merchants"}

    def test_unknown_role_fallback(self) -> None:
        """Unknown roles should fall back to viewer permissions."""
        acl = AccessControlLayer()
        allowed = acl.get_allowed_tables("intern", ALL_TABLES)
        # Falls back to viewer
        assert "users" not in allowed

    def test_check_sql_catches_denied_table(self) -> None:
        """SQL referencing a denied table should produce violations."""
        acl = AccessControlLayer()
        sql = "SELECT u.email FROM users u JOIN accounts a ON u.user_id = a.user_id"
        violations = acl.check_sql(sql, "analyst", ALL_TABLES)
        assert any("users" in v for v in violations)

    def test_check_sql_clean(self) -> None:
        """SQL referencing only allowed tables should produce no violations."""
        acl = AccessControlLayer()
        sql = "SELECT * FROM transactions WHERE amount > 100"
        violations = acl.check_sql(sql, "analyst", ALL_TABLES)
        assert violations == []

    def test_check_sql_blocks_writes_for_analyst(self) -> None:
        """Non-write roles should be flagged for write operations."""
        acl = AccessControlLayer()
        sql = "INSERT INTO transactions (amount) VALUES (100)"
        violations = acl.check_sql(sql, "analyst", ALL_TABLES)
        assert any("write" in v.lower() for v in violations)
