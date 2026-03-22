"""Async SQLite database interface with safety guardrails.

Wraps aiosqlite to provide:
- Read-only mode enforcement via URI parameters.
- Row-limit capping to prevent accidental full-table dumps.
- Query timeout protection.
- Schema introspection for agent context.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite

from src.core.config import load_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class Database:
    """Async SQLite connection manager.

    Usage::

        db = Database()
        await db.connect()
        rows = await db.execute_query("SELECT * FROM transactions LIMIT 10")
        await db.close()
    """

    def __init__(self, path: str | None = None) -> None:
        cfg = load_settings().database
        self._path = Path(path or cfg.path)
        self._read_only = cfg.read_only
        self._max_rows = cfg.max_rows_returned
        self._timeout = cfg.query_timeout_seconds
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the database connection."""
        if self._conn is not None:
            return

        if not self._path.exists():
            raise FileNotFoundError(
                f"Database not found at {self._path}. Run `python -m scripts.seed_db` first."
            )

        uri = f"file:{self._path}"
        if self._read_only:
            uri += "?mode=ro"

        self._conn = await aiosqlite.connect(uri, uri=True)
        self._conn.row_factory = aiosqlite.Row
        logger.info("db_connected", path=str(self._path), read_only=self._read_only)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def execute_query(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
        max_rows: int | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a SELECT query and return results as dicts.

        Args:
            sql: The SQL query to execute.
            params: Bind parameters for the query.
            max_rows: Override the default row limit.

        Returns:
            List of row dictionaries.

        Raises:
            PermissionError: If a write operation is attempted in read-only mode.
            asyncio.TimeoutError: If the query exceeds the timeout.
            RuntimeError: For other database errors.
        """
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        cap = max_rows or self._max_rows
        sql_upper = sql.strip().upper()

        # Block write operations even if not in read-only mode
        write_keywords = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"}
        first_word = sql_upper.split()[0] if sql_upper else ""
        if first_word in write_keywords:
            raise PermissionError(f"Write operation blocked: {first_word}")

        logger.info("db_execute", sql=sql[:200], max_rows=cap)

        try:
            async with asyncio.timeout(self._timeout):
                cursor = await self._conn.execute(sql, params)
                rows = await cursor.fetchmany(cap)
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                return [dict(zip(columns, row)) for row in rows]
        except asyncio.TimeoutError:
            logger.error("db_timeout", sql=sql[:200])
            raise
        except sqlite3.OperationalError as exc:
            logger.error("db_error", error=str(exc), sql=sql[:200])
            raise RuntimeError(f"Query execution failed: {exc}") from exc

    async def get_table_names(self) -> list[str]:
        """Return all user table names in the database."""
        rows = await self.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [r["name"] for r in rows]

    async def get_schema_ddl(self) -> str:
        """Return the full CREATE TABLE DDL for all tables."""
        rows = await self.execute_query(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return "\n\n".join(r["sql"] for r in rows if r["sql"])

    async def get_table_info(self, table: str) -> list[dict[str, Any]]:
        """Return column metadata for a single table.

        Args:
            table: Table name.

        Returns:
            List of column info dicts (cid, name, type, notnull, dflt_value, pk).
        """
        return await self.execute_query(f"PRAGMA table_info('{table}')")

    async def get_row_count(self, table: str) -> int:
        """Return the approximate row count for a table.

        Args:
            table: Table name.

        Returns:
            Integer row count.
        """
        rows = await self.execute_query(f"SELECT COUNT(*) as cnt FROM '{table}'")
        return int(rows[0]["cnt"]) if rows else 0

    async def get_full_schema_context(self) -> str:
        """Build a human-readable schema summary for LLM context.

        Returns:
            Multi-line string describing every table, its columns, and row counts.
        """
        tables = await self.get_table_names()
        parts: list[str] = []
        for table in sorted(tables):
            cols = await self.get_table_info(table)
            count = await self.get_row_count(table)
            col_lines = []
            for c in cols:
                pk = " [PK]" if c["pk"] else ""
                nn = " NOT NULL" if c["notnull"] else ""
                col_lines.append(f"    {c['name']} {c['type']}{pk}{nn}")
            parts.append(
                f"TABLE {table} ({count:,} rows):\n" + "\n".join(col_lines)
            )
        return "\n\n".join(parts)
