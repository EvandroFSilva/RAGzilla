from __future__ import annotations

"""Utilities for generating answers from retrieved context."""

from typing import Iterable, List, TYPE_CHECKING

from .logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .settings import Settings

LOGGER = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


def format_prompt(question: str, contexts: Iterable[dict]) -> str:
    """Build a prompt containing the retrieved contexts."""

    context_text = "\n\n".join(
        f"Fonte: {item['document']} (pagina {item['page_number']})\n{item['text']}" for item in contexts
    )
    instructions = (
        "Voce e um assistente especializado em contratos. Responda somente com base nas fontes fornecidas.\n"
        "Se nao houver informacao suficiente, diga que nao e possivel responder com os documentos selecionados."
    )
    return f"{instructions}\n\nContexto:\n{context_text}\n\nPergunta: {question}\nResposta:"


def generate_answer(
    question: str,
    contexts: List[dict],
    settings: "Settings" | None = None,
) -> dict:
    """Generate a grounded answer with references."""

    if settings is None:
        from .settings import get_settings

        settings = get_settings()
    if not contexts:
        return {
            "answer": "Nao foi possivel localizar contexto suficiente para responder a pergunta.",
            "sources": [],
        }

    if OpenAI is None:
        raise RuntimeError("openai package is required to call GPT models")
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not configured")

    client = OpenAI(api_key=settings.openai_api_key)
    prompt = format_prompt(question, contexts)
    response = client.responses.create(
        model=settings.model_generation,
        input=prompt,
    )
    answer_text = response.output[0].content[0].text if response.output else ""
    return {
        "answer": answer_text.strip(),
        "sources": [
            {
                "document": item["document"],
                "chunk": item["text"],
                "page_number": item["page_number"],
            }
            for item in contexts
        ],
    }


__all__ = ["generate_answer", "format_prompt"]

