"""Streamlit interface for the RAG-SQL multi-agent pipeline.

Shows live agent reasoning steps, the generated SQL, query results,
and the natural-language explanation.

Run::

    streamlit run src/ui/app.py
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import streamlit as st

from src.agents.orchestrator import PipelineOrchestrator
from src.core.logging import AgentTrace, setup_logging

# ── Page config ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG-SQL Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@400;500;700&display=swap');

    .stApp {
        font-family: 'DM Sans', sans-serif;
    }
    code, pre, .stCode {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .agent-step {
        border-left: 3px solid;
        padding: 0.5rem 0.75rem;
        margin: 0.25rem 0;
        border-radius: 0 6px 6px 0;
        font-size: 0.88rem;
    }
    .agent-step.ok { border-color: #22c55e; background: rgba(34,197,94,0.06); }
    .agent-step.warning { border-color: #eab308; background: rgba(234,179,8,0.06); }
    .agent-step.error { border-color: #ef4444; background: rgba(239,68,68,0.06); }

    .agent-name {
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .agent-action {
        color: #64748b;
        margin-left: 0.5rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── State ─────────────────────────────────────────────────────────────

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def _get_orchestrator() -> PipelineOrchestrator:
    """Lazy-initialize the orchestrator."""
    if st.session_state.orchestrator is None:
        setup_logging()
        orch = PipelineOrchestrator()
        asyncio.get_event_loop().run_until_complete(orch.initialize())
        st.session_state.orchestrator = orch
    return st.session_state.orchestrator


def _render_trace(trace: AgentTrace) -> None:
    """Render trace events as styled HTML blocks."""
    for event in trace.events:
        status = event.get("status", "ok")
        agent = event.get("agent", "?")
        action = event.get("action", "")
        detail = event.get("detail", "")

        if isinstance(detail, dict):
            detail_str = json.dumps(detail, indent=2, default=str)
        else:
            detail_str = str(detail)[:300]

        st.markdown(
            f'<div class="agent-step {status}">'
            f'<span class="agent-name">{agent}</span>'
            f'<span class="agent-action">→ {action}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if detail_str and detail_str != "{}":
            with st.expander(f"Detail: {agent}.{action}", expanded=False):
                st.code(detail_str, language="json")


def _render_results(context: dict[str, Any]) -> None:
    """Render SQL, results table, and explanation."""
    sql = context.get("generated_sql")
    if sql:
        st.subheader("Generated SQL")
        st.code(sql, language="sql")

    cot = context.get("chain_of_thought")
    if cot:
        with st.expander("Chain of Thought", expanded=False):
            for step, content in cot.items():
                st.markdown(f"**{step.upper()}**: {content}")

    results = context.get("query_results", [])
    if results:
        st.subheader(f"Results ({len(results)} rows)")
        st.dataframe(results, use_container_width=True)

    explanation = context.get("explanation", "")
    if explanation:
        st.subheader("Answer")
        st.success(explanation)

    highlights = context.get("highlights", [])
    if highlights:
        cols = st.columns(min(len(highlights), 4))
        for i, h in enumerate(highlights):
            cols[i % len(cols)].metric(label=f"Highlight {i + 1}", value=h)

    suggestions = context.get("follow_up_suggestions", [])
    if suggestions:
        st.caption("Suggested follow-ups:")
        for s in suggestions:
            st.caption(f"→ {s}")


# ── Sidebar ───────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔍 RAG-SQL Agent")
    st.caption("Multi-agent natural language → SQL")
    st.divider()

    role = st.selectbox(
        "User Role",
        ["analyst", "admin", "viewer"],
        index=0,
        help="Controls which tables you can query.",
    )

    st.divider()
    st.markdown("**Role permissions:**")
    role_info = {
        "admin": "Full access to all tables including users PII.",
        "analyst": "Transactions, accounts, merchants. No PII access.",
        "viewer": "Read-only: transactions and merchants only.",
    }
    st.info(role_info.get(role, "Unknown role."))

    st.divider()
    show_trace = st.toggle("Show agent trace", value=True)

    st.divider()
    st.caption("Built with LangChain, FAISS, LiteLLM")

# ── Main area ─────────────────────────────────────────────────────────

st.title("Ask your fintech database anything")

# Example queries
with st.expander("Example queries", expanded=False):
    examples = [
        "What are the top 10 merchants by total transaction volume?",
        "How many transactions were flagged for fraud this year?",
        "Show me the average transaction amount by channel",
        "Which accounts have the highest balance?",
        "List all pending transactions from the last 30 days",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{hash(ex)}"):
            st.session_state.pending_query = ex

# Chat input
query = st.chat_input("Ask a question about your fintech data...")

# Handle example button clicks
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.spinner("Processing query across agents..."):
        start = time.time()
        trace = AgentTrace()
        orch = _get_orchestrator()
        result = asyncio.get_event_loop().run_until_complete(
            orch.process(query, role=role, trace=trace)
        )
        elapsed = time.time() - start

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result.get("explanation", ""),
        "context": result,
        "trace": trace,
        "elapsed": elapsed,
    })

# Render chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            ctx = msg.get("context", {})
            tr = msg.get("trace")
            elapsed = msg.get("elapsed", 0)

            if show_trace and tr:
                with st.expander(f"Agent Trace ({len(tr.events)} steps, {elapsed:.1f}s)", expanded=True):
                    _render_trace(tr)

            _render_results(ctx)
