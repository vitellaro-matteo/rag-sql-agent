"""SQL Generator Agent — produces SQL with visible chain-of-thought reasoning.

Uses the synthesized schema context and access-control constraints to
generate a SQLite-compatible query.  The full reasoning chain (understand →
plan → construct → verify) is preserved in the trace for observability.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.core.logging import AgentTrace


class SQLGeneratorAgent(BaseAgent):
    """Generate SQL from natural language with explicit reasoning steps."""

    name = "sql_generator"
    prompt_name = "sql_generator"

    async def run(self, context: dict[str, Any], trace: AgentTrace) -> dict[str, Any]:
        """Generate a SQL query for the user's question.

        Reads:
            context["question"]: The user's question.
            context["schema_context"]: Formatted schema from the RAG agent.
            context["allowed_tables"]: Tables the user may access.
            context["max_rows"]: Row limit for this role.

        Writes:
            context["generated_sql"]: The SQL string.
            context["chain_of_thought"]: Dict with understand/plan/construct/verify.
            context["sql_confidence"]: Float 0.0–1.0.

        Returns:
            Updated context.
        """
        trace.record(self.name, "generating", detail=context["question"][:100])

        result = await self.call_llm({
            "question": context["question"],
            "schema_context": context.get("schema_context", "No schema available."),
            "allowed_tables": ", ".join(context.get("allowed_tables", [])),
            "max_rows": context.get("max_rows", 500),
        })

        cot = result.get("chain_of_thought", {})
        sql = result.get("sql", "")
        confidence = result.get("confidence", 0.5)

        # Ensure the SQL has a LIMIT if not already present
        if sql and "LIMIT" not in sql.upper():
            max_rows = context.get("max_rows", 500)
            sql = sql.rstrip().rstrip(";") + f"\nLIMIT {max_rows};"

        context["generated_sql"] = sql
        context["chain_of_thought"] = cot
        context["sql_confidence"] = confidence

        trace.record(
            self.name,
            "generated",
            detail={
                "sql": sql[:300],
                "confidence": confidence,
                "chain_of_thought": cot,
            },
        )

        return context
