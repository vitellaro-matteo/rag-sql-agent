"""Validation Agent — audits generated SQL before execution.

Combines deterministic rule checks (blocked keywords, join count,
table access) with an LLM-based semantic audit for more nuanced
safety and correctness analysis.
"""

from __future__ import annotations

import re
from typing import Any

from src.agents.base import BaseAgent
from src.core.config import load_settings
from src.core.logging import AgentTrace


class ValidationAgent(BaseAgent):
    """Two-pass SQL validator: deterministic rules + LLM audit."""

    name = "validation"
    prompt_name = "validation"

    def __init__(self) -> None:
        super().__init__()
        self._cfg = load_settings().validation

    def _deterministic_check(self, sql: str, context: dict[str, Any]) -> list[dict[str, str]]:
        """Run fast, rule-based checks that don't need an LLM.

        Args:
            sql: The SQL query to check.
            context: Pipeline context with access control info.

        Returns:
            List of issue dicts with severity/category/description/suggestion.
        """
        issues: list[dict[str, str]] = []
        sql_upper = sql.upper()

        # 1. Blocked keywords
        for kw in self._cfg.blocked_keywords:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, sql_upper):
                issues.append({
                    "severity": "critical",
                    "category": "safety",
                    "description": f"Blocked keyword detected: {kw}",
                    "suggestion": "Only SELECT queries are permitted.",
                })

        # 2. Join count
        join_count = len(re.findall(r"\bJOIN\b", sql_upper))
        if join_count > self._cfg.max_joins:
            issues.append({
                "severity": "warning",
                "category": "performance",
                "description": f"Query has {join_count} JOINs (max: {self._cfg.max_joins})",
                "suggestion": "Simplify query or break into sub-queries.",
            })

        # 3. Subquery count
        subq_count = sql_upper.count("SELECT") - 1
        if subq_count > self._cfg.max_subqueries:
            issues.append({
                "severity": "warning",
                "category": "performance",
                "description": f"Query has {subq_count} subqueries (max: {self._cfg.max_subqueries})",
                "suggestion": "Consider using CTEs or simplifying logic.",
            })

        # 4. Missing WHERE on large tables (heuristic)
        if "WHERE" not in sql_upper and "GROUP BY" not in sql_upper:
            issues.append({
                "severity": "info",
                "category": "performance",
                "description": "No WHERE clause — may scan full table.",
                "suggestion": "Add filters to improve performance.",
            })

        # 5. Denied table references
        denied = set(context.get("denied_tables", []))
        for table in denied:
            if re.search(rf"\b{re.escape(table.upper())}\b", sql_upper):
                issues.append({
                    "severity": "critical",
                    "category": "access",
                    "description": f"Query references denied table: {table}",
                    "suggestion": f"Remove references to '{table}' — your role cannot access it.",
                })

        # 6. SQL injection patterns
        injection_patterns = [
            r";\s*--",
            r"'\s*OR\s+'1'\s*=\s*'1",
            r"1\s*=\s*1",
            r"UNION\s+SELECT",
        ]
        for pat in injection_patterns:
            if re.search(pat, sql_upper):
                issues.append({
                    "severity": "critical",
                    "category": "injection",
                    "description": f"Suspicious pattern detected: {pat}",
                    "suggestion": "Query appears to contain injection attempt.",
                })

        return issues

    def _compute_verdict(self, issues: list[dict[str, str]]) -> str:
        """Derive overall verdict from issue list.

        Args:
            issues: All detected issues.

        Returns:
            "blocked" if any critical, "warning" if any warnings, else "safe".
        """
        severities = {i["severity"] for i in issues}
        if "critical" in severities:
            return "blocked"
        if "warning" in severities:
            return "warning"
        return "safe"

    async def run(self, context: dict[str, Any], trace: AgentTrace) -> dict[str, Any]:
        """Validate the generated SQL.

        Reads:
            context["generated_sql"]: SQL to validate.
            context["schema_context"]: Schema for correctness checks.
            context["user_role"]: Role for access checks.
            context["allowed_tables"]: Permitted tables.
            context["denied_tables"]: Blocked tables.

        Writes:
            context["validation_verdict"]: "safe", "warning", or "blocked".
            context["validation_issues"]: List of issue dicts.
            context["validation_reasoning"]: Summary from deterministic + LLM audit.

        Returns:
            Updated context.
        """
        sql = context.get("generated_sql", "")
        trace.record(self.name, "validating", detail=sql[:200])

        # Pass 1: deterministic rules
        issues = self._deterministic_check(sql, context)

        # Short-circuit if already blocked
        if self._compute_verdict(issues) == "blocked":
            context["validation_verdict"] = "blocked"
            context["validation_issues"] = issues
            context["validation_reasoning"] = "Blocked by deterministic safety rules."
            trace.record(
                self.name, "blocked_deterministic",
                detail=issues, status="warning",
            )
            return context

        # Pass 2: LLM semantic audit
        try:
            llm_result = await self.call_llm({
                "sql": sql,
                "schema_context": context.get("schema_context", ""),
                "user_role": context.get("user_role", "viewer"),
                "allowed_tables": ", ".join(context.get("allowed_tables", [])),
                "denied_tables": ", ".join(context.get("denied_tables", [])),
            })
            llm_issues = llm_result.get("issues", [])
            issues.extend(llm_issues)
        except (ValueError, RuntimeError) as exc:
            self.logger.warning("llm_audit_failed", error=str(exc))
            issues.append({
                "severity": "info",
                "category": "correctness",
                "description": "LLM audit unavailable; relying on rule-based checks only.",
                "suggestion": "",
            })

        verdict = self._compute_verdict(issues)
        context["validation_verdict"] = verdict
        context["validation_issues"] = issues
        context["validation_reasoning"] = f"{len(issues)} issues found. Verdict: {verdict}."

        trace.record(
            self.name,
            "validated",
            detail={
                "verdict": verdict,
                "issue_count": len(issues),
                "critical": sum(1 for i in issues if i["severity"] == "critical"),
            },
        )

        return context
