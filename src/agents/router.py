"""Query Router Agent — classifies user intent and query complexity.

First agent in the pipeline. Determines whether the question requires
SQL generation, schema lookup, or is out of scope, and estimates the
query complexity to inform downstream agents.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.core.logging import AgentTrace


class QueryRouterAgent(BaseAgent):
    """Classify intent and complexity, route to the correct sub-pipeline."""

    name = "router"
    prompt_name = "router"

    async def run(self, context: dict[str, Any], trace: AgentTrace) -> dict[str, Any]:
        """Classify the user's question.

        Reads:
            context["question"]: The raw user question.
            context["available_tables"]: List of table names.

        Writes:
            context["intent"]: One of sql_query, schema_info, explanation, greeting, out_of_scope.
            context["complexity"]: One of simple, moderate, complex.
            context["tables_mentioned"]: Tables the router detected.
            context["router_reasoning"]: Explanation of the classification.

        Returns:
            Updated context.
        """
        trace.record(self.name, "classifying", detail=context["question"][:120])

        result = await self.call_llm({
            "question": context["question"],
            "available_tables": ", ".join(context.get("available_tables", [])),
        })

        context["intent"] = result.get("intent", "out_of_scope")
        context["complexity"] = result.get("complexity", "simple")
        context["tables_mentioned"] = result.get("tables_mentioned", [])
        context["router_reasoning"] = result.get("reasoning", "")

        trace.record(
            self.name,
            "classified",
            detail={
                "intent": context["intent"],
                "complexity": context["complexity"],
                "tables": context["tables_mentioned"],
                "reasoning": context["router_reasoning"],
            },
        )

        return context
