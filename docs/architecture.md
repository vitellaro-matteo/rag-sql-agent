# Architecture

## System Overview

This system converts natural-language questions into SQL queries against a fintech database. It does this through a pipeline of six cooperating agents, each with a single responsibility, orchestrated by a central coordinator.

```
User Question
     │
     ▼
┌─────────────────┐
│  QueryRouter     │  Classify intent + complexity
└────────┬────────┘
         │ sql_query
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│SchemaRAG│ │AccessControl │  (parallel)
└────┬───┘ └──────┬───────┘
     └──────┬─────┘
            ▼
     ┌──────────────┐
     │ SQLGenerator  │  Chain-of-thought SQL
     └──────┬───────┘
            ▼
     ┌──────────────┐
     │ Validation    │  Rule-based + LLM audit
     └──────┬───────┘
            │ safe?
            ▼
     ┌──────────────┐
     │  Execute SQL  │  Async SQLite
     └──────┬───────┘
            ▼
     ┌──────────────┐
     │  Explainer    │  Results → English
     └──────────────┘
```

## Design Decisions

### 1. Why a multi-agent pipeline instead of a single prompt?

A single "NL-to-SQL" prompt works for demos. It breaks in production for three reasons:

- **No access control.** A monolithic prompt has no enforcement point to block unauthorized table access. The `AccessControlLayer` sits as a deterministic gate between schema retrieval and SQL generation — it never touches an LLM and cannot be prompt-injected around.

- **No validation before execution.** In a single-shot system, malicious or malformed SQL goes straight to the database. The `ValidationAgent` catches DDL, injection patterns, and performance risks before any query touches SQLite.

- **Opaque failures.** When a single prompt produces wrong SQL, you can't tell whether the problem was intent misclassification, missing schema context, or a generation error. Separate agents produce separate trace events, making debugging tractable.

### 2. Why FAISS for schema retrieval instead of stuffing the full DDL?

With 4 tables, full DDL fits in any context window. This won't scale. Real enterprise databases have 50-500+ tables. FAISS retrieval over chunked schema documentation means:

- Only relevant columns and relationships enter the SQL generator's context.
- Adding tables requires only re-indexing, not prompt rewriting.
- Retrieval scores provide a signal for when context is insufficient (low scores → ask the user to clarify).

The schema store chunks at three granularities: table-level descriptions (business context), column-level metadata (types, constraints, sample values), and relationship documentation (foreign keys, join paths). This gives the generator both the what (column types) and the why (business semantics).

### 3. Why LiteLLM instead of direct OpenAI SDK?

The system needs to work with:
- **Ollama** for local development (no API key, no cost, no data leaving the machine).
- **OpenAI** for production deployments where latency matters.
- **Anthropic/Azure** for enterprise customers with existing contracts.

LiteLLM provides a single `acompletion()` interface that routes to any backend based on a model string. Swapping providers is a one-line config change. The `_build_model_string()` function maps our config's `provider`/`model` pair to LiteLLM's expected format.

### 4. Why prompt templates in YAML, not hardcoded strings?

- **Version control.** Prompt changes show up as clean diffs in Git, not buried inside Python logic.
- **Non-engineer iteration.** A domain expert can tune the SQL generator's system prompt without reading Python.
- **Testing.** Tests can swap prompt files to verify behavior under different instructions.
- **Template variables** use Python's `str.format()`, so `{question}` and `{schema_context}` are interpolated at runtime.

### 5. Why a deterministic AccessControlLayer instead of another LLM agent?

Security gates must never hallucinate. An LLM-based access control agent might say "this looks fine" for a cleverly phrased query that still references a denied table. The `AccessControlLayer`:

- Reads role definitions from `config/settings.yaml` (no LLM involved).
- Uses regex to scan generated SQL for denied table names.
- Returns violations as structured data, not natural language.
- Falls back to the most restrictive role (`viewer`) for unknown roles.

This makes access control auditable, testable, and immune to prompt injection.

### 6. Why two-pass validation (deterministic + LLM)?

The deterministic pass catches the obvious things fast: blocked keywords (DROP, DELETE), injection patterns (1=1, UNION SELECT), join count limits, and denied table references. These checks are ~1ms and never wrong.

The LLM pass catches subtle issues the rules miss: semantic correctness (did the query actually answer the question?), column/table mismatches the regex didn't catch, and nuanced performance concerns. If the LLM audit fails (timeout, bad response), the system gracefully degrades to rule-based-only validation.

### 7. Why async with parallel agent execution?

Schema retrieval (FAISS vector search + LLM synthesis) and access control (pure config lookup) have no data dependency on each other. Running them in parallel with `asyncio.gather()` saves one full LLM round-trip on every request. The orchestrator uses `asyncio.create_task()` for this:

```python
schema_task = asyncio.create_task(self._schema_rag.run(context, trace))
access_task = asyncio.create_task(self._access.run(context, trace))
await asyncio.gather(schema_task, access_task)
```

The rest of the pipeline is sequential because each step depends on the previous one's output.

### 8. Why structured logging with AgentTrace?

Every agent step is recorded as a structured event with timestamp, agent name, action, detail payload, and status. This serves two audiences:

- **Developers** get JSON logs via structlog for debugging and monitoring.
- **End users** see the trace rendered in the Streamlit UI as a live reasoning chain, which builds trust and makes error diagnosis self-service.

The trace is a simple append-only list, not a complex DAG, because the pipeline is fundamentally sequential (with one parallel fan-out). If the architecture evolved to support speculative execution or backtracking, the trace model would need to become a tree.

### 9. Why SQLite instead of Postgres?

Portability. The system should work on a reviewer's laptop with `docker compose up`. SQLite requires zero infrastructure. The `Database` class abstracts the connection; swapping to asyncpg for Postgres requires changing only the connect method and query dialect. The `read_only` mode uses SQLite's URI parameter `?mode=ro` to enforce immutability at the driver level.

### 10. Why test with mocked LLM responses?

Tests must run in CI without API keys. The `conftest.py` defines `MOCK_RESPONSES` — deterministic JSON payloads for each agent. The `mock_llm` fixture patches `src.core.llm.complete` globally. This means:

- `pytest` runs in <10 seconds with zero network calls.
- Tests verify agent logic (prompt rendering, JSON parsing, context mutation), not LLM quality.
- LLM quality is a prompt-engineering concern tested via manual evaluation, not unit tests.

## Data Flow

```
question: str
    │
    ├─→ RouterAgent         → intent, complexity, tables_mentioned
    │
    ├─→ SchemaRAGAgent      → schema_hits, schema_context, schema_analysis
    │   └── FAISS query     → top-k schema chunks
    │
    ├─→ AccessControlLayer  → allowed_tables, denied_tables, max_rows
    │
    ├─→ SQLGeneratorAgent   → generated_sql, chain_of_thought, sql_confidence
    │
    ├─→ ValidationAgent     → validation_verdict, validation_issues
    │   ├── deterministic   → blocked keywords, injection, joins, access
    │   └── LLM audit       → correctness, semantic safety
    │
    ├─→ Database.execute    → query_results
    │
    └─→ ExplainerAgent      → explanation, highlights, follow_up_suggestions
```

Every value listed above is a key in the shared `context: dict[str, Any]` that flows through the pipeline. Agents read what they need and write what they produce. The orchestrator manages the flow.

## Security Model

| Layer | Threat | Mitigation |
|---|---|---|
| Access Control | Unauthorized table access | Deterministic role-based filtering before SQL generation |
| Validation (rules) | SQL injection, DDL | Regex patterns for blocked keywords and injection signatures |
| Validation (LLM) | Semantic attacks | LLM audit for subtle correctness and safety issues |
| Database | Write operations | Read-only SQLite URI + explicit write-keyword blocking |
| Database | Resource exhaustion | Row caps, query timeouts, join limits |

## Scaling Considerations

This architecture was designed for single-user local use. For production multi-tenant deployment:

- **Database**: Replace SQLite with Postgres + connection pooling (asyncpg). The `Database` interface is already async.
- **FAISS**: For >10k schema chunks, switch to FAISS IVF or use a managed vector DB (Pinecone, Weaviate). The `SchemaStore` interface hides this.
- **LLM calls**: Add request queuing and rate limiting. LiteLLM supports this natively.
- **Caching**: Cache schema retrieval results (schema changes rarely) and SQL generation for repeated queries.
- **Auth**: The role string currently comes from the UI dropdown. In production, extract it from JWT/OAuth tokens.
- **Observability**: Ship structlog events to an OTEL collector for distributed tracing.
