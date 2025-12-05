"""Embedding helpers built on top of OpenAI's API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import numpy as np  # noqa: F401

from .logging_config import get_logger

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore


LOGGER = get_logger(__name__)


@dataclass
class OpenAIEmbeddingClient:
    """Wrapper responsible for producing embeddings via OpenAI."""

    api_key: str

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required to generate embeddings")
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        self._client = OpenAI(api_key=self.api_key)

    def embed_texts(
        self,
        texts: Iterable[str],
        model: str,
        batch_size: int = 512,
        max_tokens_per_request: int = 300_000,
        safety_margin: float = 0.9,
    ) -> List[List[float]]:
        """Return embeddings for the given texts.

        - Coerces all items to strings
        - Replaces empty/whitespace-only strings with a single space
        - Sends requests in batches to avoid oversized payloads
        """

        items: List[str] = ["" if t is None else str(t) for t in texts]
        if not items:
            return []
        # Avoid invalid empty inputs
        items = [(" " if s.strip() == "" else s) for s in items]

        # Compute per-item token estimates
        def estimate_tokens_list(text_list: List[str]) -> List[int]:
            if tiktoken is not None:
                try:
                    try:
                        enc = tiktoken.encoding_for_model(model)
                    except Exception:
                        enc = tiktoken.get_encoding("cl100k_base")
                    return [len(enc.encode(s)) for s in text_list]
                except Exception:
                    pass
            # Heuristic fallback: max of words and chars/4
            est: List[int] = []
            for s in text_list:
                by_words = max(1, len(s.split()))
                by_chars = max(1, int(len(s) / 4))
                est.append(max(by_words, by_chars))
            return est

        token_estimates = estimate_tokens_list(items)
        token_limit = int(max_tokens_per_request * safety_margin)

        all_embeddings: List[List[float]] = []
        start = 0
        while start < len(items):
            current_tokens = 0
            end = start
            # pack as many as fit both the item count and estimated token limit
            while end < len(items) and (end - start) < batch_size:
                next_tokens = token_estimates[end]
                if current_tokens + next_tokens > token_limit and end > start:
                    break
                # Always include at least one item
                if current_tokens + next_tokens > token_limit and end == start:
                    # Extremely long single item; send alone
                    end += 1
                    break
                current_tokens += next_tokens
                end += 1

            batch = items[start:end]
            LOGGER.debug(
                "Requesting %d embeddings from OpenAI (items %d-%d, est_tokens=%d)",
                len(batch), start, end - 1, current_tokens,
            )
            response = self._client.embeddings.create(model=model, input=batch)
            all_embeddings.extend([item.embedding for item in response.data])
            start = end

        return all_embeddings

    @staticmethod
    def normalize(vectors: Iterable[Iterable[float]]) -> "np.ndarray":
        """Return L2 normalised numpy array."""

        if np is None:
            raise RuntimeError("numpy is required to normalise embeddings")

        array = np.array(list(vectors), dtype="float32")
        if not array.size:
            return array
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms


__all__ = ["OpenAIEmbeddingClient"]
