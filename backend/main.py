"""FastAPI application exposing RAG endpoints."""
from __future__ import annotations

from typing import List

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from .logging_config import configure_logging, get_logger
from .query import generate_answer
from .retrieval import DualIndexRetriever
from .settings import Settings, get_settings

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
