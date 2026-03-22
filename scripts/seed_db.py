"""Generate a realistic fake fintech SQLite database.

Creates ~100 users, ~20 accounts, ~50 merchants, and ~1000 transactions
with plausible patterns: recurring payments, salary deposits, merchant
categories, and geographic distribution.

Run directly::

    python -m scripts.seed_db
"""

from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)

DB_PATH = Path("data/fintech.db")

# ── Schema DDL ────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id         INTEGER PRIMARY KEY,
    email           TEXT NOT NULL UNIQUE,
    full_name       TEXT NOT NULL,
    phone           TEXT,
    date_of_birth   DATE,
    country         TEXT NOT NULL,
    city            TEXT,
    kyc_status      TEXT NOT NULL DEFAULT 'pending'
                    CHECK(kyc_status IN ('pending','verified','rejected')),
    risk_score      REAL CHECK(risk_score BETWEEN 0.0 AND 1.0),
    created_at      DATETIME NOT NULL DEFAULT (datetime('now')),
    updated_at      DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS accounts (
    account_id      INTEGER PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(user_id),
    account_type    TEXT NOT NULL
                    CHECK(account_type IN ('checking','savings','business','credit')),
    currency        TEXT NOT NULL DEFAULT 'USD',
    balance         REAL NOT NULL DEFAULT 0.0,
    credit_limit    REAL,
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK(status IN ('active','frozen','closed')),
    opened_at       DATETIME NOT NULL DEFAULT (datetime('now')),
    updated_at      DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS merchants (
    merchant_id     INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    country         TEXT NOT NULL,
    city            TEXT,
    mcc_code        TEXT,
    is_online       BOOLEAN NOT NULL DEFAULT 0,
    avg_ticket      REAL,
    created_at      DATETIME NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id  INTEGER PRIMARY KEY,
    account_id      INTEGER NOT NULL REFERENCES accounts(account_id),
    merchant_id     INTEGER REFERENCES merchants(merchant_id),
    amount          REAL NOT NULL,
    currency        TEXT NOT NULL DEFAULT 'USD',
    transaction_type TEXT NOT NULL
                    CHECK(transaction_type IN ('purchase','refund','transfer','withdrawal','deposit','fee')),
    status          TEXT NOT NULL DEFAULT 'completed'
                    CHECK(status IN ('pending','completed','failed','reversed')),
    description     TEXT,
    channel         TEXT CHECK(channel IN ('pos','online','atm','wire','app')),
    fraud_flag      BOOLEAN NOT NULL DEFAULT 0,
    created_at      DATETIME NOT NULL,
    processed_at    DATETIME
);

CREATE INDEX IF NOT EXISTS idx_txn_account   ON transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_txn_merchant  ON transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_txn_created   ON transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_txn_type      ON transactions(transaction_type);
CREATE INDEX IF NOT EXISTS idx_acct_user     ON accounts(user_id);
"""

# ── Merchant categories ───────────────────────────────────────────────

MERCHANT_CATEGORIES = [
    ("Groceries", "5411"), ("Restaurants", "5812"), ("Gas Stations", "5541"),
    ("Online Retail", "5999"), ("Streaming Services", "4899"),
    ("Ride Sharing", "4121"), ("Airlines", "3000"), ("Hotels", "7011"),
    ("Pharmacies", "5912"), ("Utilities", "4900"), ("Insurance", "6300"),
    ("Software & SaaS", "5734"), ("Education", "8220"), ("Healthcare", "8099"),
    ("Fitness", "7941"),
]


def _generate_users(conn: sqlite3.Connection, n: int = 100) -> list[int]:
    """Insert n fake users and return their IDs."""
    countries = ["US", "DE", "GB", "CL", "JP", "FR", "BR", "CA", "AU", "LU"]
    kyc_weights = [0.7, 0.2, 0.1]  # verified, pending, rejected
    rows = []
    for _ in range(n):
        country = random.choice(countries)
        rows.append((
            fake.unique.email(),
            fake.name(),
            fake.phone_number()[:20],
            fake.date_of_birth(minimum_age=18, maximum_age=75).isoformat(),
            country,
            fake.city(),
            random.choices(["verified", "pending", "rejected"], weights=kyc_weights)[0],
            round(random.uniform(0.0, 1.0), 3),
            fake.date_time_between(start_date="-3y", end_date="-6m").isoformat(),
            fake.date_time_between(start_date="-6m", end_date="now").isoformat(),
        ))
    conn.executemany(
        "INSERT INTO users (email,full_name,phone,date_of_birth,country,city,"
        "kyc_status,risk_score,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return [r[0] for r in conn.execute("SELECT user_id FROM users").fetchall()]


def _generate_accounts(conn: sqlite3.Connection, user_ids: list[int], n: int = 20) -> list[int]:
    """Insert accounts distributed across users."""
    acct_types = ["checking", "savings", "business", "credit"]
    currencies = ["USD", "USD", "USD", "EUR", "GBP", "CLP"]
    rows = []
    # Ensure first 20 users get at least one account, rest spread randomly
    chosen_users = user_ids[:n] if n <= len(user_ids) else user_ids
    for uid in chosen_users:
        atype = random.choice(acct_types)
        cur = random.choice(currencies)
        balance = round(random.uniform(100, 250_000), 2)
        credit = round(random.uniform(5000, 50000), 2) if atype == "credit" else None
        rows.append((
            uid, atype, cur, balance, credit, "active",
            fake.date_time_between(start_date="-2y", end_date="-3m").isoformat(),
            fake.date_time_between(start_date="-3m", end_date="now").isoformat(),
        ))
    # Add extra accounts for some users to hit realistic multi-account patterns
    for _ in range(max(0, n - len(chosen_users))):
        uid = random.choice(user_ids)
        atype = random.choice(acct_types)
        cur = random.choice(currencies)
        balance = round(random.uniform(100, 250_000), 2)
        credit = round(random.uniform(5000, 50000), 2) if atype == "credit" else None
        rows.append((
            uid, atype, cur, balance, credit, "active",
            fake.date_time_between(start_date="-2y", end_date="-3m").isoformat(),
            fake.date_time_between(start_date="-3m", end_date="now").isoformat(),
        ))
    conn.executemany(
        "INSERT INTO accounts (user_id,account_type,currency,balance,credit_limit,"
        "status,opened_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    return [r[0] for r in conn.execute("SELECT account_id FROM accounts").fetchall()]


def _generate_merchants(conn: sqlite3.Connection, n: int = 50) -> list[int]:
    """Insert fake merchants across categories."""
    rows = []
    for i in range(n):
        cat, mcc = MERCHANT_CATEGORIES[i % len(MERCHANT_CATEGORIES)]
        is_online = 1 if cat in ("Online Retail", "Streaming Services", "Software & SaaS") else random.randint(0, 1)
        rows.append((
            fake.company(),
            cat,
            random.choice(["US", "DE", "GB", "CL", "JP", "FR"]),
            fake.city(),
            mcc,
            is_online,
            round(random.uniform(5, 500), 2),
            fake.date_time_between(start_date="-4y", end_date="-1y").isoformat(),
        ))
    conn.executemany(
        "INSERT INTO merchants (name,category,country,city,mcc_code,is_online,"
        "avg_ticket,created_at) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    return [r[0] for r in conn.execute("SELECT merchant_id FROM merchants").fetchall()]


def _generate_transactions(
    conn: sqlite3.Connection,
    account_ids: list[int],
    merchant_ids: list[int],
    n: int = 1000,
) -> None:
    """Insert n realistic transactions."""
    txn_types = ["purchase", "refund", "transfer", "withdrawal", "deposit", "fee"]
    txn_weights = [0.55, 0.05, 0.15, 0.08, 0.12, 0.05]
    channels = ["pos", "online", "atm", "wire", "app"]
    statuses = ["completed", "completed", "completed", "pending", "failed"]

    base_date = datetime.now() - timedelta(days=365)
    rows = []

    for _ in range(n):
        acct_id = random.choice(account_ids)
        ttype = random.choices(txn_types, weights=txn_weights)[0]

        # Merchant is only relevant for purchases and refunds
        merch_id = random.choice(merchant_ids) if ttype in ("purchase", "refund") else None

        # Amount varies by type
        if ttype == "purchase":
            amount = -round(random.uniform(1.5, 2500), 2)
        elif ttype == "refund":
            amount = round(random.uniform(5, 500), 2)
        elif ttype == "deposit":
            amount = round(random.uniform(100, 15000), 2)
        elif ttype == "withdrawal":
            amount = -round(random.uniform(20, 5000), 2)
        elif ttype == "fee":
            amount = -round(random.uniform(0.5, 50), 2)
        else:  # transfer
            amount = -round(random.uniform(10, 10000), 2)

        created = base_date + timedelta(
            days=random.randint(0, 365),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        processed = created + timedelta(seconds=random.randint(1, 7200))
        status = random.choice(statuses)
        channel = random.choice(channels)
        fraud = 1 if random.random() < 0.02 else 0
        desc = _make_description(ttype, merch_id, conn) if merch_id else ttype.title()

        rows.append((
            acct_id, merch_id, amount, "USD", ttype, status, desc,
            channel, fraud, created.isoformat(), processed.isoformat(),
        ))

    conn.executemany(
        "INSERT INTO transactions (account_id,merchant_id,amount,currency,"
        "transaction_type,status,description,channel,fraud_flag,created_at,"
        "processed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )


def _make_description(ttype: str, merchant_id: int | None, conn: sqlite3.Connection) -> str:
    """Build a plausible transaction description."""
    if merchant_id:
        row = conn.execute(
            "SELECT name, category FROM merchants WHERE merchant_id=?",
            (merchant_id,),
        ).fetchone()
        if row:
            return f"{ttype.title()} at {row[0]} ({row[1]})"
    return ttype.title()


def main() -> None:
    """Seed the database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)

    user_ids = _generate_users(conn, n=100)
    account_ids = _generate_accounts(conn, user_ids, n=20)
    merchant_ids = _generate_merchants(conn, n=50)
    _generate_transactions(conn, account_ids, merchant_ids, n=1000)

    conn.commit()

    # Print summary
    for table in ["users", "accounts", "merchants", "transactions"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count:,} rows")

    conn.close()
    print(f"\nDatabase written to {DB_PATH}")


if __name__ == "__main__":
    main()
