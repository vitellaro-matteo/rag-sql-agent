"""Explainer Agent — translates SQL results into natural language.

Takes the raw query results and original question and produces a
business-friendly summary with key highlights and follow-up suggestions.
"""

from __future__ import annotations

import json
from typing import Any

from src.agents.base import BaseAgent
from src.core.logging import AgentTrace


class ExplainerAgent(BaseAgent):
    """Translate query results into clear, actionable natural language."""

    name = "explainer"
    prompt_name = "explainer"

    def _format_results(self, rows: list[dict[str, Any]], max_display: int = 20) -> str:
        """Format result rows into a compact text table for the LLM.

        Args:
            rows: Query result rows.
            max_display: Maximum rows to include in the LLM context.

        Returns:
            Formatted string representation.
        """
        if not rows:
            return "(no results)"

        display = rows[:max_display]
        truncated = len(rows) > max_display

        # Build simple text table
        headers = list(display[0].keys())
        lines = [" | ".join(headers)]
        lines.append("-" * len(lines[0]))
        for row in display:
            lines.append(" | ".join(str(row.get(h, "")) for h in headers))
        if truncated:
            lines.append(f"... ({len(rows) - max_display} more rows)")

        return "\n".join(lines)

    async def run(self, context: dict[str, Any], trace: AgentTrace) -> dict[str, Any]:
        """Explain query results in natural language.

        Reads:
            context["question"]: Original user question.
            context["generated_sql"]: The executed SQL.
            context["query_results"]: List of result dicts.

        Writes:
            context["explanation"]: Plain-English summary.
            context["highlights"]: Key data points.
            context["follow_up_suggestions"]: Suggested next questions.

        Returns:
            Updated context.
        """
        results = context.get("query_results", [])
        trace.record(self.name, "explaining", detail=f"{len(results)} result rows")

        formatted = self._format_results(results)

        try:
            llm_result = await self.call_llm({
                "question": context["question"],
                "sql": context.get("generated_sql", ""),
                "row_count": len(results),
                "results": formatted,
            })

            context["explanation"] = llm_result.get("summary", "Results returned successfully.")
            context["highlights"] = llm_result.get("highlights", [])
            context["follow_up_suggestions"] = llm_result.get("follow_up_suggestions", [])
        except (ValueError, RuntimeError) as exc:
            self.logger.warning("explanation_failed", error=str(exc))
            # Fallback to a simple mechanical summary
            context["explanation"] = self._fallback_explanation(context["question"], results)
            context["highlights"] = []
            context["follow_up_suggestions"] = []

        trace.record(
            self.name,
            "explained",
            detail={
                "summary_len": len(context.get("explanation", "")),
                "highlights": context.get("highlights", []),
            },
        )

        return context

    @staticmethod
    def _fallback_explanation(question: str, results: list[dict[str, Any]]) -> str:
        """Generate a basic explanation without the LLM.

        Args:
            question: The original question.
            results: Query results.

        Returns:
            A simple summary string.
        """
        if not results:
            return "The query returned no results matching your criteria."

        n = len(results)
        cols = list(results[0].keys())
        preview = json.dumps(results[0], default=str)
        if len(preview) > 200:
            preview = preview[:200] + "..."

        return (
            f"The query returned {n} row{'s' if n != 1 else ''}. "
            f"Columns: {', '.join(cols)}. "
            f"First row: {preview}"
        )
