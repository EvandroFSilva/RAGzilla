"""Ferramentas para ingestão e consulta de documentos utilizando Ollama offline.

Este módulo oferece funções reutilizáveis que são utilizadas tanto pela aplicação
Streamlit quanto por utilitários de linha de comando. Ele implementa duas etapas
de indexação:

1. **Índice de descoberta** com chunks curtos (~50 tokens, 15 de sobreposição)
   dedicado à etapa de busca de documentos.
2. **Índice de contexto** com chunks maiores (~500 tokens, 50 de sobreposição)
   utilizado para responder perguntas sobre os documentos selecionados.

Os embeddings e as respostas são gerados via modelos locais expostos pelo
servidor Ollama.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import faiss
import numpy as np
import requests

try:
    import fitz  # type: ignore  # PyMuPDF para leitura de PDFs
except Exception:  # pragma: no cover - import opcional
    fitz = None


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuração de caminhos
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = Path(os.getenv("RAG_DOCUMENTS_DIR", BASE_DIR / "data" / "documents"))
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", BASE_DIR / "data" / "indexes"))

SMALL_INDEX_FILE = INDEX_DIR / "discovery.index"
SMALL_METADATA_FILE = INDEX_DIR / "discovery_metadata.jsonl"
LARGE_INDEX_FILE = INDEX_DIR / "context.index"
LARGE_METADATA_FILE = INDEX_DIR / "context_metadata.jsonl"


# ---------------------------------------------------------------------------
# Configuração de modelos Ollama
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "embeddinggemma")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3:8b")


# ---------------------------------------------------------------------------
# Utilidades para comunicação com Ollama
# ---------------------------------------------------------------------------


class OllamaError(RuntimeError):
    """Erro específico para comunicação com Ollama."""


def _ollama_request(endpoint: str, payload: Dict) -> Dict:
    url = f"{OLLAMA_HOST}{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - dependente do ambiente
        raise OllamaError(f"Falha ao comunicar com Ollama em {url}: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - resposta inesperada
        raise OllamaError(f"Resposta inválida de Ollama em {url}") from exc


def get_embedding(text: str) -> np.ndarray:
    if not text.strip():
        return np.zeros(1536, dtype="float32")

    data = _ollama_request("/api/embeddings", {"model": EMBED_MODEL, "prompt": text})
    embedding = data.get("embedding")
    if not embedding:
        raise OllamaError("Resposta de embedding vazia de Ollama")
    vector = np.asarray(embedding, dtype="float32")
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def generate_answer(prompt: str) -> str:
    data = _ollama_request(
        "/api/generate",
        {"model": LLM_MODEL, "prompt": prompt, "stream": False},
    )
    response = data.get("response")
    if not response:
        raise OllamaError("Resposta vazia do modelo de linguagem Ollama")
    return response.strip()


# ---------------------------------------------------------------------------
# Leitura e chunking de documentos
# ---------------------------------------------------------------------------


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def _read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) não está instalado para leitura de PDFs.")
        doc = fitz.open(str(path))
        try:
            texts = [page.get_text("text") for page in doc]
        finally:
            doc.close()
        return "\n".join(texts)
    raise ValueError(f"Extensão não suportada: {path.suffix}")


def iter_documents(directory: Path) -> Iterable[Tuple[str, Path, str]]:
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            doc_id = file_path.stem
            try:
                text = _read_document(file_path)
            except Exception as exc:  # pragma: no cover - apenas logamos
                LOGGER.warning("Não foi possível ler %s: %s", file_path, exc)
                continue
            yield doc_id, file_path, text


def chunk_text(words: Sequence[str], size: int, overlap: int) -> Iterable[Tuple[int, int]]:
    if size <= 0:
        raise ValueError("size deve ser maior que zero")
    if overlap >= size:
        raise ValueError("overlap deve ser menor que size")

    start = 0
    total = len(words)
    while start < total:
        end = min(start + size, total)
        yield start, end
        if end == total:
            break
        start = end - overlap


def build_chunks(text: str, size: int, overlap: int) -> List[Dict]:
    words = text.split()
    spans = list(chunk_text(words, size, overlap))
    chunks: List[Dict] = []
    for idx, (start, end) in enumerate(spans):
        snippet_words = words[start:end]
        snippet = " ".join(snippet_words)
        chunks.append(
            {
                "chunk_index": idx,
                "start": start,
                "end": end,
                "text": snippet,
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Indexação com FAISS
# ---------------------------------------------------------------------------


@dataclass
class VectorIndex:
    index: faiss.Index
    metadata: List[Dict]


def _write_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def _write_metadata(metadata: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for item in metadata:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")


def _read_metadata(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


def indexes_exist() -> bool:
    return all(
        path.exists()
        for path in (SMALL_INDEX_FILE, SMALL_METADATA_FILE, LARGE_INDEX_FILE, LARGE_METADATA_FILE)
    )


def ingest_documents(documents_dir: Path = DOCUMENTS_DIR) -> Dict[str, int]:
    documents = list(iter_documents(documents_dir))
    if not documents:
        raise RuntimeError(
            f"Nenhum documento encontrado em {documents_dir}. Adicione arquivos .txt, .md ou .pdf."
        )

    discovery_vectors: List[np.ndarray] = []
    discovery_metadata: List[Dict] = []
    context_vectors: List[np.ndarray] = []
    context_metadata: List[Dict] = []

    for doc_id, file_path, text in documents:
        words = text.split()
        if not words:
            continue

        discovery_chunks = build_chunks(text, size=50, overlap=15)
        context_chunks = build_chunks(text, size=500, overlap=50)

        for chunk in discovery_chunks:
            embedding = get_embedding(chunk["text"])
            discovery_vectors.append(embedding)
            discovery_metadata.append(
                {
                    "doc_id": doc_id,
                    "document_name": file_path.name,
                    "document_path": str(file_path),
                    "chunk_index": chunk["chunk_index"],
                    "snippet": chunk["text"][:400],
                }
            )

        for chunk in context_chunks:
            embedding = get_embedding(chunk["text"])
            context_vectors.append(embedding)
            context_metadata.append(
                {
                    "doc_id": doc_id,
                    "document_name": file_path.name,
                    "document_path": str(file_path),
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                }
            )

    if not discovery_vectors or not context_vectors:
        raise RuntimeError("Não foi possível gerar embeddings para os documentos fornecidos.")

    discovery_matrix = np.vstack(discovery_vectors).astype("float32")
    context_matrix = np.vstack(context_vectors).astype("float32")

    discovery_index = faiss.IndexFlatIP(discovery_matrix.shape[1])
    context_index = faiss.IndexFlatIP(context_matrix.shape[1])
    discovery_index.add(discovery_matrix)
    context_index.add(context_matrix)

    _write_index(discovery_index, SMALL_INDEX_FILE)
    _write_index(context_index, LARGE_INDEX_FILE)
    _write_metadata(discovery_metadata, SMALL_METADATA_FILE)
    _write_metadata(context_metadata, LARGE_METADATA_FILE)

    load_discovery_index.cache_clear()
    load_context_index.cache_clear()

    return {
        "documents": len(documents),
        "discovery_chunks": len(discovery_metadata),
        "context_chunks": len(context_metadata),
    }


@lru_cache(maxsize=1)
def load_discovery_index() -> VectorIndex:
    index = faiss.read_index(str(SMALL_INDEX_FILE))
    metadata = _read_metadata(SMALL_METADATA_FILE)
    return VectorIndex(index=index, metadata=metadata)


@lru_cache(maxsize=1)
def load_context_index() -> VectorIndex:
    index = faiss.read_index(str(LARGE_INDEX_FILE))
    metadata = _read_metadata(LARGE_METADATA_FILE)
    return VectorIndex(index=index, metadata=metadata)


def search_documents(query: str, top_k: int = 10) -> List[Dict]:
    vector = get_embedding(query)
    dataset = load_discovery_index()
    distances, indices = dataset.index.search(np.expand_dims(vector, axis=0), top_k)

    results: List[Dict] = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        metadata = dataset.metadata[idx]
        results.append(
            {
                "score": float(distance),
                "doc_id": metadata["doc_id"],
                "document_name": metadata["document_name"],
                "document_path": metadata["document_path"],
                "chunk_index": metadata["chunk_index"],
                "snippet": metadata["snippet"],
            }
        )
    return results


def retrieve_context(question: str, doc_ids: Iterable[str], top_k: int = 6) -> List[Dict]:
    doc_ids_set = set(doc_ids)
    if not doc_ids_set:
        return []

    vector = get_embedding(question)
    dataset = load_context_index()

    multiplier = max(len(doc_ids_set), 1) * 10
    distances, indices = dataset.index.search(np.expand_dims(vector, axis=0), top_k * multiplier)

    contexts: List[Dict] = []
    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        metadata = dataset.metadata[idx]
        if metadata["doc_id"] not in doc_ids_set:
            continue
        contexts.append(
            {
                "score": float(distance),
                "doc_id": metadata["doc_id"],
                "document_name": metadata["document_name"],
                "chunk_index": metadata["chunk_index"],
                "text": metadata["text"],
            }
        )
        if len(contexts) >= top_k:
            break
    return contexts


def build_prompt(question: str, contexts: Sequence[Dict]) -> str:
    if not contexts:
        return (
            "Você é um assistente que não encontrou documentos relevantes para a pergunta. "
            f"Explique que não há dados disponíveis para responder: {question}"
        )

    formatted_contexts = []
    for idx, ctx in enumerate(contexts, start=1):
        formatted_contexts.append(
            f"Trecho {idx} — Documento: {ctx['document_name']} (chunk {ctx['chunk_index']})\n{ctx['text']}"
        )
    context_block = "\n\n".join(formatted_contexts)

    return (
        "Você é um assistente jurídico que responde utilizando exclusivamente os trechos "
        "fornecidos. Caso a resposta não esteja nos trechos, informe que as informações "
        "não foram encontradas.\n\n"
        f"Trechos selecionados:\n{context_block}\n\n"
        f"Pergunta: {question}\n"
        "Responda em português de forma direta e, quando possível, referencie o documento "
        "utilizado (ex: 'Documento X')."
    )


def answer_question(question: str, doc_ids: Iterable[str], top_k: int = 6) -> Dict[str, object]:
    contexts = retrieve_context(question, doc_ids, top_k=top_k)
    prompt = build_prompt(question, contexts)
    answer = generate_answer(prompt)
    return {"answer": answer, "contexts": contexts}


# ---------------------------------------------------------------------------
# Linha de comando utilitária
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG offline com Ollama")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Reprocessa todos os documentos e cria os índices vetoriais",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Realiza uma busca simples nos índices de descoberta",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Responde uma pergunta utilizando os documentos selecionados",
    )
    parser.add_argument(
        "--docs",
        type=str,
        default=str(DOCUMENTS_DIR),
        help="Diretório contendo os documentos a serem indexados",
    )
    parser.add_argument(
        "--selected",
        nargs="*",
        help="Lista de doc_ids a serem utilizados na resposta",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Quantidade de trechos para compor o contexto da resposta",
    )
    return parser


def main() -> None:  # pragma: no cover - função de CLI
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.ingest:
        summary = ingest_documents(Path(args.docs))
        LOGGER.info("Indexação concluída: %s", summary)
        return

    if args.search:
        if not indexes_exist():
            raise RuntimeError("Índices inexistentes. Execute com --ingest primeiro.")
        hits = search_documents(args.search)
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return

    if args.question:
        if not indexes_exist():
            raise RuntimeError("Índices inexistentes. Execute com --ingest primeiro.")
        if not args.selected:
            raise RuntimeError("Informe ao menos um doc_id através de --selected.")
        result = answer_question(args.question, args.selected, top_k=args.top_k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":  # pragma: no cover - entrada CLI
    main()
