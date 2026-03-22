"""Tests for the async database wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.database import Database


@pytest.mark.asyncio
class TestDatabase:
    """Test suite for the Database class."""

    async def test_connect_and_query(self, tmp_db: Path) -> None:
        """Should connect and execute a simple query."""
        db = Database(path=str(tmp_db))
        await db.connect()
        rows = await db.execute_query("SELECT COUNT(*) as cnt FROM transactions")
        assert rows[0]["cnt"] == 5
        await db.close()

    async def test_get_table_names(self, tmp_db: Path) -> None:
        """Should list all user tables."""
        db = Database(path=str(tmp_db))
        await db.connect()
        tables = await db.get_table_names()
        assert set(tables) == {"users", "accounts", "merchants", "transactions"}
        await db.close()

    async def test_max_rows_cap(self, tmp_db: Path) -> None:
        """Should cap results to max_rows."""
        db = Database(path=str(tmp_db))
        await db.connect()
        rows = await db.execute_query("SELECT * FROM transactions", max_rows=2)
        assert len(rows) == 2
        await db.close()

    async def test_blocks_write_operations(self, tmp_db: Path) -> None:
        """Should raise PermissionError on write queries."""
        db = Database(path=str(tmp_db))
        await db.connect()
        with pytest.raises(PermissionError, match="DELETE"):
            await db.execute_query("DELETE FROM transactions WHERE 1=1")
        await db.close()

    async def test_blocks_drop(self, tmp_db: Path) -> None:
        """Should raise PermissionError on DROP."""
        db = Database(path=str(tmp_db))
        await db.connect()
        with pytest.raises(PermissionError, match="DROP"):
            await db.execute_query("DROP TABLE transactions")
        await db.close()

    async def test_schema_context(self, tmp_db: Path) -> None:
        """get_full_schema_context should return readable text."""
        db = Database(path=str(tmp_db))
        await db.connect()
        ctx = await db.get_full_schema_context()
        assert "transactions" in ctx
        assert "amount" in ctx
        await db.close()

    async def test_missing_db_raises(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for missing DB."""
        db = Database(path=str(tmp_path / "nope.db"))
        with pytest.raises(FileNotFoundError):
            await db.connect()
