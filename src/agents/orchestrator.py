"""Pipeline Orchestrator — coordinates the multi-agent execution flow.

Manages the end-to-end lifecycle of a user query:
  1. Route (classify intent)
  2. Retrieve schema (FAISS RAG)
  3. Enforce access control
  4. Generate SQL
  5. Validate SQL
  6. Execute query
  7. Explain results

Supports async execution and parallel agent runs where dependencies allow.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.agents.access_control import AccessControlLayer
from src.agents.explainer import ExplainerAgent
from src.agents.router import QueryRouterAgent
from src.agents.schema_rag import SchemaRAGAgent
from src.agents.sql_generator import SQLGeneratorAgent
from src.agents.validation import ValidationAgent
from src.core.config import load_settings
from src.core.database import Database
from src.core.logging import AgentTrace, get_logger
from src.core.schema_store import SchemaStore

logger = get_logger(__name__)


class PipelineOrchestrator:
    """Coordinates multi-agent query processing with retry logic.

    Usage::

        orch = PipelineOrchestrator()
        await orch.initialize()
        result = await orch.process("Show me top 10 transactions", role="analyst")
    """

    def __init__(self) -> None:
        self._cfg = load_settings().orchestrator
        self._db = Database()
        self._store = SchemaStore()

        # Agents
        self._router = QueryRouterAgent()
        self._schema_rag = SchemaRAGAgent(self._store)
        self._access = AccessControlLayer()
        self._sql_gen = SQLGeneratorAgent()
        self._validator = ValidationAgent()
        self._explainer = ExplainerAgent()

        self._initialized = False

    async def initialize(self) -> None:
        """Connect to the database and load the FAISS index."""
        if self._initialized:
            return
        await self._db.connect()
        self._store.load()
        self._initialized = True
        logger.info("orchestrator_initialized")

    async def shutdown(self) -> None:
        """Clean up database connections."""
        await self._db.close()
        self._initialized = False

    async def process(
        self,
        question: str,
        *,
        role: str = "analyst",
        trace: AgentTrace | None = None,
    ) -> dict[str, Any]:
        """Run the full pipeline for a user question.

        Args:
            question: Natural-language query.
            role: User role for access control (default: analyst).
            trace: Optional trace recorder; a new one is created if not provided.

        Returns:
            Final context dict with all agent outputs.
        """
        if not self._initialized:
            await self.initialize()

        if trace is None:
            trace = AgentTrace()

        context: dict[str, Any] = {
            "question": question,
            "user_role": role,
            "available_tables": await self._db.get_table_names(),
            "trace": trace,
        }

        trace.record("orchestrator", "pipeline_start", detail={
            "question": question[:120],
            "role": role,
        })

        try:
            # Step 1: Route
            context = await self._router.run(context, trace)

            intent = context.get("intent", "out_of_scope")
            if intent in ("greeting", "out_of_scope"):
                context["explanation"] = self._handle_non_query(intent, question)
                trace.record("orchestrator", "non_query_response", detail=intent)
                return context

            if intent == "schema_info":
                schema_ddl = await self._db.get_full_schema_context()
                context["explanation"] = schema_ddl
                trace.record("orchestrator", "schema_info_response")
                return context

            # Step 2 + 3: Schema retrieval and access control (parallel)
            schema_task = asyncio.create_task(self._schema_rag.run(context, trace))
            access_task = asyncio.create_task(self._access.run(context, trace))

            await asyncio.gather(schema_task, access_task)

            # Step 4: Generate SQL (with retries)
            for attempt in range(1, self._cfg.max_retries + 1):
                context = await self._sql_gen.run(context, trace)

                # Step 5: Validate
                context = await self._validator.run(context, trace)

                verdict = context.get("validation_verdict", "safe")
                if verdict == "safe":
                    break
                if verdict == "blocked":
                    context["explanation"] = self._format_blocked(context)
                    trace.record("orchestrator", "query_blocked", status="warning")
                    return context
                if attempt < self._cfg.max_retries:
                    trace.record(
                        "orchestrator", "retry",
                        detail=f"Attempt {attempt} had warnings, retrying.",
                        status="warning",
                    )

            # Step 6: Execute
            sql = context.get("generated_sql", "")
            if sql:
                try:
                    results = await self._db.execute_query(
                        sql, max_rows=context.get("max_rows", 500)
                    )
                    context["query_results"] = results
                    trace.record("orchestrator", "query_executed", detail={
                        "rows_returned": len(results),
                    })
                except (PermissionError, RuntimeError, asyncio.TimeoutError) as exc:
                    context["query_results"] = []
                    context["execution_error"] = str(exc)
                    trace.record(
                        "orchestrator", "execution_error",
                        detail=str(exc), status="error",
                    )
            else:
                context["query_results"] = []

            # Step 7: Explain
            context = await self._explainer.run(context, trace)

        except Exception as exc:
            logger.error("pipeline_error", error=str(exc))
            context["explanation"] = f"An error occurred: {exc}"
            trace.record("orchestrator", "pipeline_error", detail=str(exc), status="error")

        trace.record("orchestrator", "pipeline_complete")
        return context

    @staticmethod
    def _handle_non_query(intent: str, question: str) -> str:
        """Return a friendly response for non-SQL intents.

        Args:
            intent: Classified intent.
            question: Original question.

        Returns:
            User-facing message.
        """
        if intent == "greeting":
            return (
                "Hello! I'm a SQL assistant for fintech data. Ask me about "
                "transactions, accounts, merchants, or users and I'll query "
                "the database for you."
            )
        return (
            "That question doesn't seem related to the fintech database. "
            "I can help with queries about transactions, accounts, merchants, "
            "and users. Could you rephrase?"
        )

    @staticmethod
    def _format_blocked(context: dict[str, Any]) -> str:
        """Format a blocked-query explanation for the user.

        Args:
            context: Pipeline context with validation issues.

        Returns:
            User-facing blocked message.
        """
        issues = context.get("validation_issues", [])
        critical = [i for i in issues if i.get("severity") == "critical"]
        reasons = [i.get("description", "Unknown issue") for i in critical]
        return (
            "Your query was blocked for safety reasons:\n"
            + "\n".join(f"  • {r}" for r in reasons)
            + "\n\nPlease rephrase your question using only SELECT queries."
        )
