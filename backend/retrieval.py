"""Utilities for searching the FAISS indexes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .embeddings import OpenAIEmbeddingClient
from .logging_config import get_logger
from .settings import Settings, get_settings

LOGGER = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore


@dataclass
class ChunkRecord:
    """Representation of a stored chunk."""

    document_id: str
    document_name: str
    text: str
    page_number: int
    chunk_id: str
    embedding_id: int


class DualIndexRetriever:
    """Perform search over short and long FAISS indexes."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding_client: Optional[OpenAIEmbeddingClient] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedding_client = embedding_client or OpenAIEmbeddingClient(
            api_key=self.settings.openai_api_key
        )
        if faiss is None:
            raise RuntimeError("faiss-cpu is required to perform retrieval")
        self.short_index = self._load_index(self.settings.short_index_dir / "index.faiss")
        self.long_index = self._load_index(self.settings.long_index_dir / "index.faiss")
        self.short_chunks = self._load_chunks(self.settings.short_chunks_path)
        self.long_chunks = self._load_chunks(self.settings.long_chunks_path)

    def _load_index(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {path}. Run the ingestion pipeline first."
            )
        return faiss.read_index(str(path))

    @staticmethod
    def _load_chunks(path: Path) -> Dict[int, ChunkRecord]:
        if not path.exists():
            raise FileNotFoundError(
                f"Chunk metadata not found at {path}. Run the ingestion pipeline first."
            )
        chunks: Dict[int, ChunkRecord] = {}
        with path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                record = json.loads(line)
                chunks[int(record["embedding_id"])] = ChunkRecord(**record)
        return chunks

    def _embed_query(self, query: str, model: str) -> np.ndarray:
        vector = self.embedding_client.embed_texts([query], model=model)
        if not vector:
            raise ValueError("Query embedding returned no data")
        normalized = self.embedding_client.normalize(vector)
        return normalized.astype("float32")

    def search_documents(self, query: str, top_k: int = 10) -> List[dict]:
        """Return the most relevant documents for the provided query."""

        embedding = self._embed_query(query, self.settings.model_embedding_short)
        scores, ids = self.short_index.search(embedding, top_k)
        doc_scores: Dict[str, float] = {}
        doc_names: Dict[str, str] = {}
        for score, chunk_id in zip(scores[0], ids[0]):
            if chunk_id == -1:
                continue
            chunk = self.short_chunks.get(int(chunk_id))
            if not chunk:
                continue
            doc_scores[chunk.document_id] = max(doc_scores.get(chunk.document_id, float("-inf")), float(score))
            doc_names[chunk.document_id] = chunk.document_name
        ranked = sorted(
            (
                {"document_id": doc_id, "document_name": doc_names[doc_id], "score": score}
                for doc_id, score in doc_scores.items()
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        return ranked

    def retrieve_context(
        self,
        query: str,
        selected_documents: Iterable[str],
        top_k: int = 5,
    ) -> List[dict]:
        """Return the most relevant long chunks for the selected documents."""

        embedding = self._embed_query(query, self.settings.model_embedding_long)
        # Expand candidate pool before filtering by selected docs to avoid empty results
        try:
            total = int(getattr(self.long_index, "ntotal", 0)) or (top_k * 100)
        except Exception:
            total = top_k * 100
        search_k = max(top_k * 50, 200)
        search_k = min(search_k, total)
        scores, ids = self.long_index.search(embedding, search_k)
        selected_documents = {doc.lower() for doc in selected_documents}
        contexts: List[dict] = []
        for score, chunk_id in zip(scores[0], ids[0]):
            if chunk_id == -1:
                continue
            chunk = self.long_chunks.get(int(chunk_id))
            if not chunk:
                continue
            if chunk.document_id.lower() not in selected_documents:
                continue
            contexts.append(
                {
                    "document": chunk.document_name,
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "score": float(score),
                    "text": chunk.text,
                }
            )
            if len(contexts) >= top_k:
                break
        return contexts


__all__ = ["DualIndexRetriever", "ChunkRecord"]
