"""Document ingestion and indexing pipeline."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing helper
    import numpy as np  # noqa: F401
try:  # pragma: no cover - optional dependency
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore

from .drive_utils import GoogleDriveClient, compute_md5
from .embeddings import OpenAIEmbeddingClient
from .logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .settings import Settings

LOGGER = get_logger(__name__)


@dataclass
class DocumentChunk:
    """Represent a chunk of text extracted from a document."""

    document_id: str
    document_name: str
    text: str
    page_number: int
    chunk_id: str
    embedding_id: int


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping chunks."""

    tokens = text.split()
    if not tokens:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break
        chunks.append(" ".join(chunk_tokens))
        if end >= len(tokens):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_text_with_ocr(page: fitz.Page) -> str:
    """Extract text from a page, falling back to OCR when needed."""

    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required to extract text from PDFs")

    text = page.get_text().strip()
    if text:
        return text
    if Image is None or pytesseract is None:
        raise RuntimeError("pytesseract and Pillow are required to perform OCR")

    pix = page.get_pixmap()
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return pytesseract.image_to_string(image)


def _chunk_identifier(document_id: str, chunk_id: str) -> int:
    """Return a deterministic numeric identifier for a chunk."""

    return abs(hash(f"{document_id}:{chunk_id}")) % (2**63)


def extract_document_chunks(path: Path, settings: "Settings") -> Dict[str, List[DocumentChunk]]:
    """Produce short and long chunks for the provided document."""

    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) is required to extract text from PDFs")

    doc = fitz.open(path)
    short_chunks: List[DocumentChunk] = []
    long_chunks: List[DocumentChunk] = []

    for page_number, page in enumerate(doc, start=1):
        text = extract_text_with_ocr(page)

        if page_number == 1:
            short_source = text
        else:
            short_source = ""

        if short_source:
            for idx, chunk in enumerate(
                chunk_text(short_source, settings.short_chunk_size, settings.short_chunk_overlap)
            ):
                chunk_id = f"short_{page_number}_{idx}"
                short_chunks.append(
                    DocumentChunk(
                        document_id=path.name,
                        document_name=path.name,
                        text=chunk,
                        page_number=page_number,
                        chunk_id=chunk_id,
                        embedding_id=_chunk_identifier(path.name, chunk_id),
                    )
                )

        for idx, chunk in enumerate(
            chunk_text(text, settings.long_chunk_size, settings.long_chunk_overlap)
        ):
            chunk_id = f"long_{page_number}_{idx}"
            long_chunks.append(
                DocumentChunk(
                    document_id=path.name,
                    document_name=path.name,
                    text=chunk,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    embedding_id=_chunk_identifier(path.name, chunk_id),
                )
            )

    return {"short": short_chunks, "long": long_chunks}


def load_metadata(path: Path) -> Dict[str, dict]:
    """Load existing metadata from a JSONL file."""

    if not path.exists():
        return {}
    metadata: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            record = json.loads(line)
            metadata[record["document_id"]] = record
    return metadata


def save_metadata(path: Path, metadata: Dict[str, dict]) -> None:
    """Persist metadata to a JSONL file."""

    with path.open("w", encoding="utf-8") as file_obj:
        for record in metadata.values():
            json.dump(record, file_obj, ensure_ascii=False)
            file_obj.write("\n")


def save_chunks(path: Path, chunks: List[DocumentChunk]) -> None:
    """Persist chunk metadata to disk."""

    with path.open("w", encoding="utf-8") as file_obj:
        for chunk in chunks:
            json.dump(asdict(chunk), file_obj, ensure_ascii=False)
            file_obj.write("\n")


def save_embedding_vectors(
    path: Path,
    embeddings: "np.ndarray",
    ids: List[int],
) -> None:
    """Persist raw embedding vectors and ids to disk."""

    if np is None:
        raise RuntimeError("numpy is required to store embeddings")

    if embeddings.size == 0:
        LOGGER.warning("No embeddings were provided, skipping save for %s", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embeddings.astype("float32"),
        ids=np.array(ids, dtype="int64"),
    )
    LOGGER.info("Persisted %d vectors to %s", embeddings.shape[0], path)


def load_embedding_vectors(path: Path) -> tuple["np.ndarray", List[int]]:
    """Load previously stored embedding vectors and ids."""

    if np is None:
        raise RuntimeError("numpy is required to load embeddings")

    if not path.exists():
        raise FileNotFoundError(f"Embedding store not found at {path}")

    data = np.load(path)
    embeddings = data["embeddings"].astype("float32")
    ids = data["ids"].astype("int64").tolist()
    return embeddings, ids


def build_faiss_index(
    embeddings: "np.ndarray",
    ids: List[int],
    index_path: Path,
    factory: str,
    metric: str | None = None,
) -> None:
    """Persist embeddings into a FAISS index, replacing previous data."""

    if np is None:
        raise RuntimeError("numpy is required to build FAISS indexes")

    if embeddings.size == 0:
        LOGGER.warning("No embeddings were provided, skipping index creation for %s", index_path)
        return

    try:
        import faiss  # type: ignore
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("faiss-cpu needs to be installed to build indexes") from error

    # Determine metric, allowing legacy suffixes like "FlatIP" / "IVF1024,FlatL2"
    factory_str = factory
    metric_val = None
    if metric:
        m = metric.lower()
        if m in ("ip", "inner_product", "cosine"):
            metric_val = faiss.METRIC_INNER_PRODUCT
        elif m in ("l2", "euclidean"):
            metric_val = faiss.METRIC_L2
    if metric_val is None:
        upper = factory_str.upper()
        if upper.endswith("IP"):
            metric_val = faiss.METRIC_INNER_PRODUCT
            factory_str = factory_str[:-2]
        elif upper.endswith("L2"):
            metric_val = faiss.METRIC_L2
            factory_str = factory_str[:-2]
        else:
            # Default to inner product when embeddings are normalized
            metric_val = faiss.METRIC_INNER_PRODUCT

    dimension = embeddings.shape[1]
    base_index = faiss.index_factory(dimension, factory_str, metric_val)
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(embeddings.astype("float32"), np.array(ids, dtype="int64"))
    faiss.write_index(index, str(index_path))
    LOGGER.info("Persisted %d vectors to %s", embeddings.shape[0], index_path)


def ingest_documents(
    source_dir: Path,
    settings: Optional["Settings"] = None,
    drive_client: Optional[GoogleDriveClient] = None,
    embedding_client: Optional[OpenAIEmbeddingClient] = None,
    recreate_index: bool = False,
) -> None:
    """Synchronise documents, extract content, and update indexes."""

    if settings is None:
        from .settings import get_settings

        settings = get_settings()
    metadata = load_metadata(settings.metadata_path) if not recreate_index else {}

    if drive_client is not None:
        LOGGER.info("Synchronising documents from Google Drive folder %s", drive_client.folder_id)
        drive_files = drive_client.list_pdfs()
        for file_info in drive_files:
            target_path = source_dir / file_info["name"]
            if target_path.exists():
                current_md5 = compute_md5(target_path)
                if current_md5 == file_info.get("md5Checksum"):
                    LOGGER.info("Skipping %s (unchanged)", target_path.name)
                    continue
            drive_client.download_file(file_info["id"], target_path)

    embedding_client = embedding_client or OpenAIEmbeddingClient(api_key=settings.openai_api_key)

    short_chunk_records: List[DocumentChunk] = []
    long_chunk_records: List[DocumentChunk] = []

    for document_path in sorted(source_dir.glob("*.pdf")):
        LOGGER.info("Processing %s", document_path.name)
        chunks = extract_document_chunks(document_path, settings)
        document_id = document_path.name
        document_metadata = {
            "document_id": document_id,
            "name": document_path.name,
            "md5": compute_md5(document_path),
            "path": str(document_path),
        }
        metadata[document_id] = document_metadata

        short_chunk_records.extend(chunks["short"])
        long_chunk_records.extend(chunks["long"])

    if short_chunk_records:
        short_embeddings = embedding_client.normalize(
            embedding_client.embed_texts(
                (chunk.text for chunk in short_chunk_records),
                model=settings.model_embedding_short,
            )
        )
        short_ids = [chunk.embedding_id for chunk in short_chunk_records]
        build_faiss_index(
            short_embeddings,
            short_ids,
            settings.short_index_dir / "index.faiss",
            settings.faiss_factory,
            settings.faiss_metric,
        )
        save_embedding_vectors(
            settings.short_index_dir / "embeddings.npz",
            short_embeddings,
            short_ids,
        )
        save_chunks(settings.short_chunks_path, short_chunk_records)
    else:
        LOGGER.warning("No short chunks generated during ingestion")

    if long_chunk_records:
        long_embeddings = embedding_client.normalize(
            embedding_client.embed_texts(
                (chunk.text for chunk in long_chunk_records),
                model=settings.model_embedding_long,
            )
        )
        long_ids = [chunk.embedding_id for chunk in long_chunk_records]
        build_faiss_index(
            long_embeddings,
            long_ids,
            settings.long_index_dir / "index.faiss",
            settings.faiss_factory,
            settings.faiss_metric,
        )
        save_embedding_vectors(
            settings.long_index_dir / "embeddings.npz",
            long_embeddings,
            long_ids,
        )
        save_chunks(settings.long_chunks_path, long_chunk_records)
    else:
        LOGGER.warning("No long chunks generated during ingestion")

    save_metadata(settings.metadata_path, metadata)
    LOGGER.info("Ingestion completed. Metadata stored at %s", settings.metadata_path)


def cli_sync_drive() -> None:
    """CLI helper to ingest and sync documents."""

    from .settings import get_settings

    settings = get_settings()
    credentials_path = Path(
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")
    )
    drive_client = None
    if settings.google_drive_folder_id:
        drive_client = GoogleDriveClient(credentials_path=credentials_path, folder_id=settings.google_drive_folder_id)
    ingest_documents(settings.data_dir, settings=settings, drive_client=drive_client)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Google Drive PDFs into FAISS indexes")
    parser.add_argument("--sync-drive", action="store_true", help="Synchronise PDFs from Google Drive")
    parser.add_argument("--recreate", action="store_true", help="Recreate FAISS indexes from scratch")
    args = parser.parse_args()

    from .settings import get_settings

    settings = get_settings()
    drive_client = None
    if args.sync_drive and settings.google_drive_folder_id:
        credentials_path = Path(
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")
        )
        drive_client = GoogleDriveClient(credentials_path=credentials_path, folder_id=settings.google_drive_folder_id)
    ingest_documents(
        settings.data_dir,
        settings=settings,
        drive_client=drive_client,
        recreate_index=args.recreate,
    )
