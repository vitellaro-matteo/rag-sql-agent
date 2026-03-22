"""Access Control Layer — enforces role-based table and operation permissions.

This is a deterministic agent (no LLM call). It reads the user's role from
context, resolves their permissions from config, and filters the list of
accessible tables before SQL generation begins.
"""

from __future__ import annotations

import re
from typing import Any

from src.core.config import RolePermissions, load_settings
from src.core.logging import AgentTrace, get_logger

logger = get_logger(__name__)


class AccessControlLayer:
    """Deterministic role-based access control gate.

    Sits between the schema retrieval and SQL generation steps.
    Ensures that the SQL generator only sees tables the user is
    allowed to query, and that generated SQL doesn't reference
    forbidden tables.
    """

    name = "access_control"

    def __init__(self) -> None:
        self._roles = load_settings().access_control.roles

    def _resolve_role(self, role: str) -> RolePermissions:
        """Look up permissions for a role, falling back to viewer.

        Args:
            role: Role name (e.g. "analyst").

        Returns:
            The RolePermissions for that role.
        """
        if role in self._roles:
            return self._roles[role]
        logger.warning("unknown_role_fallback", role=role, fallback="viewer")
        return self._roles.get("viewer", RolePermissions())

    def get_allowed_tables(self, role: str, all_tables: list[str]) -> list[str]:
        """Return the subset of tables this role may query.

        Args:
            role: User role name.
            all_tables: All tables in the database.

        Returns:
            Filtered list of allowed table names.
        """
        perms = self._resolve_role(role)

        if "*" in perms.allowed_tables:
            allowed = set(all_tables)
        else:
            allowed = set(perms.allowed_tables) & set(all_tables)

        denied = set(perms.denied_tables)
        return sorted(allowed - denied)

    def check_sql(self, sql: str, role: str, all_tables: list[str]) -> list[str]:
        """Scan generated SQL for references to denied tables.

        Args:
            sql: The SQL query to check.
            role: User role name.
            all_tables: All tables in the database.

        Returns:
            List of violation messages (empty if clean).
        """
        perms = self._resolve_role(role)
        denied = set(perms.denied_tables)
        violations: list[str] = []

        sql_upper = sql.upper()
        for table in denied:
            # Match table name as whole word (case-insensitive)
            pattern = rf"\b{re.escape(table.upper())}\b"
            if re.search(pattern, sql_upper):
                violations.append(
                    f"Role '{role}' is denied access to table '{table}'"
                )

        # Check write operations for non-write roles
        if not perms.can_write:
            write_ops = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"}
            first_word = sql.strip().split()[0].upper() if sql.strip() else ""
            if first_word in write_ops:
                violations.append(
                    f"Role '{role}' does not have write permission"
                )

        return violations

    async def run(self, context: dict[str, Any], trace: AgentTrace) -> dict[str, Any]:
        """Apply access control filtering to the pipeline context.

        Reads:
            context["user_role"]: The requesting user's role.
            context["available_tables"]: All database tables.

        Writes:
            context["allowed_tables"]: Tables the user may query.
            context["denied_tables"]: Tables the user may NOT query.
            context["max_rows"]: Row limit for this role.
            context["can_write"]: Whether writes are permitted.

        Returns:
            Updated context.
        """
        role = context.get("user_role", "viewer")
        all_tables = context.get("available_tables", [])

        perms = self._resolve_role(role)
        allowed = self.get_allowed_tables(role, all_tables)
        denied = sorted(set(perms.denied_tables) & set(all_tables))

        context["allowed_tables"] = allowed
        context["denied_tables"] = denied
        context["max_rows"] = perms.max_rows
        context["can_write"] = perms.can_write

        trace.record(
            self.name,
            "permissions_resolved",
            detail={
                "role": role,
                "allowed": allowed,
                "denied": denied,
                "max_rows": perms.max_rows,
            },
        )

        return context
