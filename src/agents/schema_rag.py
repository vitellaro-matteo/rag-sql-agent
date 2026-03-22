"""Schema RAG Agent — retrieves relevant schema context from the FAISS index.

Uses vector similarity search to find the most relevant table definitions,
column descriptions, and relationship documentation for the user's question,
then asks the LLM to synthesize a focused schema context block.
"""

from __future__ import annotations

from typing import Any

from src.agents.base import BaseAgent
from src.core.logging import AgentTrace
from src.core.schema_store import SchemaStore


class SchemaRAGAgent(BaseAgent):
    """Retrieve and synthesize schema context for SQL generation."""

    name = "schema_rag"
    prompt_name = "schema_rag"

    def __init__(self, store: SchemaStore) -> None:
        super().__init__()
        self._store = store

    async def run(self, context: dict[str, Any], trace: AgentTrace) -> dict[str, Any]:
        """Retrieve schema chunks and synthesize context.

        Reads:
            context["question"]: The user's question.

        Writes:
            context["schema_hits"]: Raw retrieval results.
            context["schema_context"]: Formatted schema context string.
            context["schema_analysis"]: LLM-synthesized structured analysis.

        Returns:
            Updated context.
        """
        question = context["question"]
        trace.record(self.name, "retrieving", detail=f"query: {question[:80]}")

        hits = self._store.query(question)
        context["schema_hits"] = hits
        formatted = self._store.format_context(hits)
        context["schema_context"] = formatted

        trace.record(
            self.name,
            "retrieved",
            detail={
                "num_hits": len(hits),
                "top_tables": list({h.get("table", "?") for h in hits[:5]}),
                "top_score": round(hits[0]["score"], 3) if hits else 0.0,
            },
        )

        # Ask the LLM to synthesize the retrieved fragments
        try:
            analysis = await self.call_llm({
                "question": question,
                "schema_context": formatted,
            })
            context["schema_analysis"] = analysis
            trace.record(
                self.name,
                "synthesized",
                detail={
                    "relevant_tables": [
                        t.get("table") for t in analysis.get("relevant_tables", [])
                    ],
                    "warnings": analysis.get("warnings", []),
                },
            )
        except (ValueError, RuntimeError) as exc:
            self.logger.warning("synthesis_failed", error=str(exc))
            context["schema_analysis"] = {
                "relevant_tables": [],
                "joins": [],
                "warnings": ["Schema synthesis failed; using raw retrieval context."],
            }
            trace.record(self.name, "synthesis_fallback", status="warning")

        return context
