"""Build the FAISS schema index from the seeded SQLite database.

Reads the database DDL, column metadata, sample values, and relationship
info, chunks them into retrieval-friendly fragments, embeds them, and
writes a FAISS index + metadata pickle.

Run directly::

    python -m scripts.build_index
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path("data/fintech.db")

# ── Business-context annotations ──────────────────────────────────────
# These enrich the raw DDL with semantic meaning the LLM can leverage.

TABLE_DESCRIPTIONS: dict[str, str] = {
    "users": (
        "Contains PII for every registered user: name, email, phone, "
        "country, KYC verification status, and a risk score (0-1) computed "
        "by the compliance pipeline. This table is restricted to admin roles."
    ),
    "accounts": (
        "Financial accounts owned by users. Types include checking, savings, "
        "business, and credit. Tracks balance, currency, credit limits, and "
        "account status (active/frozen/closed). Linked to users via user_id FK."
    ),
    "merchants": (
        "Businesses where transactions occur. Includes category labels "
        "(Groceries, Airlines, etc.), MCC codes, geographic info, and whether "
        "the merchant operates online. avg_ticket stores typical spend."
    ),
    "transactions": (
        "Core financial events. Each row is a single monetary movement tied "
        "to an account and optionally a merchant. Supports purchases, refunds, "
        "transfers, withdrawals, deposits, and fees. Tracks channel "
        "(POS/online/ATM/wire/app), fraud flags, and timestamps."
    ),
}

RELATIONSHIP_DOCS: list[str] = [
    "transactions.account_id → accounts.account_id (every transaction belongs to one account)",
    "transactions.merchant_id → merchants.merchant_id (purchases and refunds reference a merchant; NULLable for transfers/deposits/fees)",
    "accounts.user_id → users.user_id (every account is owned by one user)",
    "To get user-level transaction data, JOIN transactions → accounts → users.",
    "To analyze spending by merchant category, JOIN transactions → merchants.",
    "The fraud_flag column on transactions is a boolean (0/1) set by the fraud detection system.",
    "Transaction amounts are signed: negative = money out (purchases, fees, withdrawals), positive = money in (deposits, refunds).",
    "Date columns use ISO 8601 format. Use date() or strftime() for SQLite date math.",
]


def _get_column_chunks(conn: sqlite3.Connection, table: str) -> list[dict[str, str]]:
    """Build one chunk per column with type and sample values."""
    cols = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    chunks = []
    for col in cols:
        name, dtype, notnull, default, pk = col[1], col[2], col[3], col[4], col[5]
        # Sample values
        try:
            samples = conn.execute(
                f"SELECT DISTINCT \"{name}\" FROM \"{table}\" WHERE \"{name}\" IS NOT NULL LIMIT 5"
            ).fetchall()
            sample_str = ", ".join(str(s[0]) for s in samples)
        except Exception:
            sample_str = "(unable to sample)"

        text = (
            f"Table: {table}\n"
            f"Column: {name} ({dtype})"
            f"{' [PRIMARY KEY]' if pk else ''}"
            f"{' NOT NULL' if notnull else ''}"
            f"{f' DEFAULT {default}' if default else ''}\n"
            f"Sample values: {sample_str}"
        )
        chunks.append({"text": text, "table": table, "column": name, "kind": "column"})
    return chunks


def _get_ddl_chunk(conn: sqlite3.Connection, table: str) -> dict[str, str]:
    """Full CREATE TABLE statement as a chunk."""
    ddl = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return {
        "text": f"CREATE TABLE DDL for {table}:\n{ddl[0] if ddl else '(not found)'}",
        "table": table,
        "kind": "ddl",
    }


def build_chunks(conn: sqlite3.Connection) -> list[dict[str, str]]:
    """Assemble all schema chunks for the FAISS index.

    Returns:
        List of dicts, each with ``text``, ``table``, and ``kind`` keys.
    """
    chunks: list[dict[str, str]] = []

    tables = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    ]

    for table in tables:
        # Table-level description
        desc = TABLE_DESCRIPTIONS.get(table, f"Table {table}")
        chunks.append({
            "text": f"Table: {table}\nDescription: {desc}",
            "table": table,
            "kind": "description",
        })

        # DDL
        chunks.append(_get_ddl_chunk(conn, table))

        # Per-column
        chunks.extend(_get_column_chunks(conn, table))

    # Relationships
    for rel in RELATIONSHIP_DOCS:
        # Determine which table the relationship primarily describes
        table = rel.split(".")[0] if "." in rel else "general"
        chunks.append({"text": rel, "table": table, "kind": "relationship"})

    return chunks


def main() -> None:
    """Build and persist the FAISS index."""
    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}. Run `python -m scripts.seed_db` first.")
        return

    conn = sqlite3.connect(str(DB_PATH))
    chunks = build_chunks(conn)
    conn.close()

    print(f"Built {len(chunks)} schema chunks.")

    # Import here to avoid slow sentence-transformers load at module level
    from src.core.schema_store import SchemaStore

    store = SchemaStore()
    store.build(chunks)
    print("FAISS index written successfully.")


if __name__ == "__main__":
    main()
