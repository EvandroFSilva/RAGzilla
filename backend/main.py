"""FastAPI application exposing RAG endpoints."""
from __future__ import annotations

from typing import List

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from .logging_config import configure_logging, get_logger
from .query import generate_answer
from .retrieval import DualIndexRetriever
from .settings import Settings, get_settings

from fastapi.responses import FileResponse
from pathlib import Path
from fastapi import Path as FastAPIPath

configure_logging()
LOGGER = get_logger(__name__)
app = FastAPI(title="RAG Web Contracts", version="1.0.0")


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    document_id: str
    document_name: str
    score: float


class QueryRequest(BaseModel):
    query: str
    selected_docs: List[str]


class SourceResponse(BaseModel):
    document: str
    chunk: str
    page_number: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceResponse]


def get_retriever(settings: Settings = Depends(get_settings)) -> DualIndexRetriever:
    if not hasattr(app.state, "retriever"):
        app.state.retriever = DualIndexRetriever(settings=settings)
    return app.state.retriever  # type: ignore[return-value]


@app.on_event("startup")
async def startup_event() -> None:  # pragma: no cover - side effect
    settings = get_settings()
    LOGGER.info("RAG Web Contracts API initialised with data dir %s", settings.data_dir)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

@app.get("/document/{document_id}/file")
async def get_document_file(
    document_id: str = FastAPIPath(..., description="Nome do arquivo (ex: contrato.pdf)"),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """
    Retorna o PDF correspondente ao document_id.
    document_id deve ser o nome do ficheiro tal como aparece em data_dir (ex: 'contrato.pdf').
    """
    # Proteção básica: document_id não pode conter navegação de diretórios
    if ".." in document_id or document_id.startswith("/") or "/" in document_id:
        raise HTTPException(status_code=400, detail="document_id inválido")

    # Primeiro, tente ler metadata para obter caminho exato (mais seguro)
    metadata_path = settings.metadata_path
    file_path = settings.data_dir / document_id  # fallback default

    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = __import__("json").loads(line)
                    except Exception:
                        continue
                    if rec.get("document_id") == document_id or rec.get("name") == document_id:
                        # usa path salvo no metadata se presente
                        rec_path = rec.get("path")
                        if rec_path:
                            candidate = Path(rec_path)
                            # garante que o arquivo reside dentro do data_dir
                            try:
                                if settings.data_dir.resolve() in candidate.resolve().parents or candidate.resolve().parent == settings.data_dir.resolve():
                                    file_path = candidate
                                else:
                                    # se o path metadata é estranho, fallback para data_dir/document_id
                                    file_path = settings.data_dir / document_id
                            except Exception:
                                file_path = settings.data_dir / document_id
                        break
        except Exception:
            # falha ao ler metadata -> continuamos com fallback
            file_path = settings.data_dir / document_id

    # Normalize e verifique confinamento no data_dir
    try:
        file_path_resolved = file_path.resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Caminho do documento inválido")

    if settings.data_dir.resolve() not in file_path_resolved.parents and file_path_resolved.parent != settings.data_dir.resolve():
        # Não permitimos servir arquivos fora da pasta de dados
        raise HTTPException(status_code=403, detail="Acesso ao arquivo negado")

    if not file_path_resolved.exists() or not file_path_resolved.is_file():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")

    # Retorna o PDF inline (o navegador / frontend pode exibir)
    return FileResponse(path=str(file_path_resolved), media_type="application/pdf", filename=file_path_resolved.name)


@app.get("/document/{document_id}/metadata")
async def get_document_metadata(
    document_id: str = FastAPIPath(..., description="Nome do arquivo (ex: contrato.pdf)"),
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Retorna o registro de metadata para o document_id (se existir) — útil para o frontend.
    """
    metadata_path = settings.metadata_path
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata não encontrada")

    with metadata_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = __import__("json").loads(line)
            except Exception:
                continue
            if rec.get("document_id") == document_id or rec.get("name") == document_id:
                return rec
    raise HTTPException(status_code=404, detail="Metadata do documento não encontrada")


@app.post("/search", response_model=List[SearchResponse])
async def search_documents(request: SearchRequest, retriever: DualIndexRetriever = Depends(get_retriever)) -> List[dict]:
    try:
        results = retriever.search_documents(request.query)
        return results
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except Exception as exc:
        LOGGER.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    retriever: DualIndexRetriever = Depends(get_retriever),
    settings: Settings = Depends(get_settings),
) -> dict:
    if not request.selected_docs:
        raise HTTPException(status_code=400, detail="Ao menos um documento deve ser selecionado")
    try:
        contexts = retriever.retrieve_context(request.query, request.selected_docs, top_k=5)
        response = generate_answer(request.query, contexts, settings=settings)
        return response
    except FileNotFoundError as error:
        raise HTTPException(status_code=500, detail=str(error))
    except Exception as exc:
        LOGGER.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


__all__ = ["app"]
