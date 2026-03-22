"""FAISS-backed schema knowledge base for retrieval-augmented generation.

Stores chunked schema descriptions (table DDL, column semantics, sample
values, relationships) as dense vectors and retrieves the top-k most
relevant fragments given a natural-language query.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.core.config import load_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


class SchemaStore:
    """FAISS vector store over schema documentation chunks.

    Each chunk is a short text describing a table, column, relationship,
    or business rule.  At query time the store returns the top-k most
    relevant chunks to feed into the SchemaRAGAgent's context window.
    """

    def __init__(
        self,
        index_path: str | None = None,
        metadata_path: str | None = None,
        model_name: str | None = None,
    ) -> None:
        cfg = load_settings()
        self._index_path = Path(index_path or cfg.faiss.index_path)
        self._meta_path = Path(metadata_path or cfg.faiss.metadata_path)
        self._model_name = model_name or cfg.embedding.model
        self._top_k = cfg.faiss.top_k

        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict[str, Any]] = []
        self._encoder: Any = None

    @property
    def encoder(self) -> Any:
        """Lazy-load the sentence transformer model."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading_encoder", model=self._model_name)
            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder

    def load(self) -> None:
        """Load a pre-built FAISS index and metadata from disk.

        Raises:
            FileNotFoundError: If the index or metadata files don't exist.
        """
        if not self._index_path.exists() or not self._meta_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found. Run `python -m scripts.build_index` first.\n"
                f"  Expected: {self._index_path}, {self._meta_path}"
            )
        self._index = faiss.read_index(str(self._index_path))
        with open(self._meta_path, "rb") as f:
            self._metadata = pickle.load(f)
        logger.info(
            "schema_store_loaded",
            vectors=self._index.ntotal,
            chunks=len(self._metadata),
        )

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """Build a new FAISS index from schema chunks and persist to disk.

        Args:
            chunks: List of dicts with at least a ``text`` key.
                    Additional keys are stored as metadata.
        """
        texts = [c["text"] for c in chunks]
        embeddings = self.encoder.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        self._metadata = chunks

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self._metadata, f)

        logger.info("schema_store_built", vectors=self._index.ntotal)

    def query(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve the most relevant schema chunks for a question.

        Args:
            question: Natural-language query.
            top_k: Number of results (defaults to config value).

        Returns:
            List of metadata dicts augmented with a ``score`` key,
            sorted by descending relevance.

        Raises:
            RuntimeError: If the index hasn't been loaded or built.
        """
        if self._index is None:
            raise RuntimeError("Schema store not initialized. Call load() or build().")

        k = min(top_k or self._top_k, self._index.ntotal)
        q_vec = self.encoder.encode([question], normalize_embeddings=True)
        q_vec = np.array(q_vec, dtype=np.float32)

        scores, indices = self._index.search(q_vec, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = {**self._metadata[idx], "score": float(score)}
            results.append(entry)

        logger.debug("schema_query", question=question[:80], hits=len(results))
        return results

    def format_context(self, hits: list[dict[str, Any]]) -> str:
        """Format retrieval hits into a prompt-friendly string.

        Args:
            hits: Results from ``query()``.

        Returns:
            Newline-separated text blocks with relevance scores.
        """
        parts: list[str] = []
        for i, hit in enumerate(hits, 1):
            score = hit.get("score", 0.0)
            text = hit.get("text", "")
            table = hit.get("table", "unknown")
            parts.append(f"[{i}] (table: {table}, relevance: {score:.3f})\n{text}")
        return "\n\n".join(parts)
