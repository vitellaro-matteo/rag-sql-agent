"""Multi-agent pipeline: router, schema RAG, access control, SQL gen, validation, explainer."""

from src.agents.router import QueryRouterAgent
from src.agents.access_control import AccessControlLayer
from src.agents.sql_generator import SQLGeneratorAgent
from src.agents.validation import ValidationAgent
from src.agents.explainer import ExplainerAgent

# SchemaRAGAgent imported lazily because it depends on sentence_transformers/FAISS
def __getattr__(name: str) -> type:
    if name == "SchemaRAGAgent":
        from src.agents.schema_rag import SchemaRAGAgent
        return SchemaRAGAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "QueryRouterAgent",
    "SchemaRAGAgent",
    "AccessControlLayer",
    "SQLGeneratorAgent",
    "ValidationAgent",
    "ExplainerAgent",
]
