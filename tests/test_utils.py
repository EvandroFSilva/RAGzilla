"""Lightweight tests for helper utilities."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ingestion import DocumentChunk, chunk_text  # noqa: E402
from backend.query import format_prompt  # noqa: E402


def test_chunk_text_respects_overlap() -> None:
    text = """Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua"""
    chunks = chunk_text(text, chunk_size=10, overlap=2)
    assert len(chunks) == 3
    assert chunks[0].endswith("sed do")
    assert chunks[1].startswith("sed do")


def test_format_prompt_renders_sources() -> None:
    contexts = [
        {
            "document": "contrato_A.pdf",
            "page_number": 3,
            "text": "Prazo de vigência: 24 meses a partir da assinatura.",
        }
    ]
    prompt = format_prompt("Qual o prazo de vigência?", contexts)
    assert "contrato_A.pdf" in prompt
    assert "24 meses" in prompt
    assert prompt.strip().endswith("Resposta:")


def test_document_chunk_serialization() -> None:
    chunk = DocumentChunk(
        document_id="contrato.pdf",
        document_name="contrato.pdf",
        text="Cláusula", 
        page_number=1,
        chunk_id="short_1_0",
        embedding_id=123,
    )
    assert chunk.document_id == "contrato.pdf"
    assert chunk.embedding_id == 123
