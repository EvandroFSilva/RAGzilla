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


@dataclass
class IngestionStats:
    """Track operational metrics for an ingestion execution."""

    drive_files_checked: int = 0
    new_drive_files: int = 0
    updated_drive_files: int = 0
    downloaded_files: int = 0
    documents_processed: int = 0
    short_chunks: int = 0
    long_chunks: int = 0

    def log_summary(self) -> None:
        """Emit an aggregated view of the ingestion activity."""

        LOGGER.info(
            "Resumo da ingestão -> verificados: %d, novos: %d, atualizados: %d, "
            "baixados: %d, lidos: %d, short_chunks: %d, long_chunks: %d",
            self.drive_files_checked,
            self.new_drive_files,
            self.updated_drive_files,
            self.downloaded_files,
            self.documents_processed,
            self.short_chunks,
            self.long_chunks,
        )


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


def load_chunks(path: Path) -> List[DocumentChunk]:
    """Load previously stored chunk metadata."""

    if not path.exists():
        return []

    chunks: List[DocumentChunk] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            payload = json.loads(line)
            chunks.append(DocumentChunk(**payload))
    return chunks


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
    if metric_val == faiss.METRIC_INNER_PRODUCT:
        faiss.normalize_L2(embeddings)
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
    stats = IngestionStats()

    if drive_client is not None:
        LOGGER.info("Synchronising documents from Google Drive folder %s", drive_client.folder_id)
        drive_files = drive_client.list_pdfs()
        stats.drive_files_checked = len(drive_files)
        for file_info in drive_files:
            is_new_file = False
            target_path = source_dir / file_info["name"]
            if target_path.exists():
                current_md5 = compute_md5(target_path)
                if current_md5 == file_info.get("md5Checksum"):
                    LOGGER.info("Skipping %s (unchanged)", target_path.name)
                    continue
            else:
                is_new_file = True

            drive_client.download_file(file_info["id"], target_path)
            stats.downloaded_files += 1
            if is_new_file:
                stats.new_drive_files += 1
            else:
                stats.updated_drive_files += 1

    document_info: Dict[str, tuple[Path, str]] = {}
    documents_to_process: List[str] = []
    for document_path in sorted(source_dir.glob("*.pdf")):
        document_id = document_path.name
        file_md5 = compute_md5(document_path)
        document_info[document_id] = (document_path, file_md5)
        existing = metadata.get(document_id)
        should_process = recreate_index or existing is None or existing.get("md5") != file_md5
        if should_process:
            documents_to_process.append(document_id)
        else:
            LOGGER.info("Skipping %s (no local changes detected)", document_path.name)

    doc_ids_to_refresh = set(documents_to_process)

    reused_short_chunks: List[DocumentChunk] = []
    reused_long_chunks: List[DocumentChunk] = []
    reused_short_embeddings: Optional["np.ndarray"] = None
    reused_long_embeddings: Optional["np.ndarray"] = None
    reused_short_ids: List[int] = []
    reused_long_ids: List[int] = []

    force_full_rebuild = False

    if not recreate_index:
        existing_short_chunks = load_chunks(settings.short_chunks_path)
        existing_long_chunks = load_chunks(settings.long_chunks_path)
        short_embeddings_path = settings.short_index_dir / "embeddings.npz"
        long_embeddings_path = settings.long_index_dir / "embeddings.npz"
        try:
            existing_short_embeddings, existing_short_ids = load_embedding_vectors(short_embeddings_path)
            short_id_to_idx = {emb_id: idx for idx, emb_id in enumerate(existing_short_ids)}
        except FileNotFoundError:
            existing_short_embeddings = None
            short_id_to_idx = {}
            if existing_short_chunks:
                LOGGER.warning("Short embedding store not found; forcing full index rebuild.")
                force_full_rebuild = True
        try:
            existing_long_embeddings, existing_long_ids = load_embedding_vectors(long_embeddings_path)
            long_id_to_idx = {emb_id: idx for idx, emb_id in enumerate(existing_long_ids)}
        except FileNotFoundError:
            existing_long_embeddings = None
            long_id_to_idx = {}
            if existing_long_chunks:
                LOGGER.warning("Long embedding store not found; forcing full index rebuild.")
                force_full_rebuild = True

        short_indexes: List[int] = []
        for chunk in existing_short_chunks:
            if chunk.document_id in doc_ids_to_refresh:
                continue
            if not short_id_to_idx:
                continue
            idx = short_id_to_idx.get(chunk.embedding_id)
            if idx is None:
                LOGGER.warning(
                    "Short embedding id %s missing from store; scheduling %s for reprocessing",
                    chunk.embedding_id,
                    chunk.document_id,
                )
                if chunk.document_id not in doc_ids_to_refresh:
                    documents_to_process.append(chunk.document_id)
                    doc_ids_to_refresh.add(chunk.document_id)
                continue
            reused_short_chunks.append(chunk)
            reused_short_ids.append(chunk.embedding_id)
            short_indexes.append(idx)
        if short_indexes and existing_short_embeddings is not None:
            reused_short_embeddings = existing_short_embeddings[short_indexes]

        long_indexes: List[int] = []
        for chunk in existing_long_chunks:
            if chunk.document_id in doc_ids_to_refresh:
                continue
            if not long_id_to_idx:
                continue
            idx = long_id_to_idx.get(chunk.embedding_id)
            if idx is None:
                LOGGER.warning(
                    "Long embedding id %s missing from store; scheduling %s for reprocessing",
                    chunk.embedding_id,
                    chunk.document_id,
                )
                if chunk.document_id not in doc_ids_to_refresh:
                    documents_to_process.append(chunk.document_id)
                    doc_ids_to_refresh.add(chunk.document_id)
                continue
            reused_long_chunks.append(chunk)
            reused_long_ids.append(chunk.embedding_id)
            long_indexes.append(idx)
        if long_indexes and existing_long_embeddings is not None:
            reused_long_embeddings = existing_long_embeddings[long_indexes]

    if force_full_rebuild:
        documents_to_process = list(document_info.keys())
        doc_ids_to_refresh = set(documents_to_process)
        reused_short_chunks = []
        reused_long_chunks = []
        reused_short_ids = []
        reused_long_ids = []
        reused_short_embeddings = None
        reused_long_embeddings = None
        LOGGER.info("Forcing full ingestion run to rebuild missing artifacts.")

    if not documents_to_process:
        LOGGER.info("No new or updated documents. Keeping previous embeddings untouched.")
        stats.log_summary()
        return

    embedding_client = embedding_client or OpenAIEmbeddingClient(api_key=settings.openai_api_key)

    short_chunk_records: List[DocumentChunk] = []
    long_chunk_records: List[DocumentChunk] = []

    for document_id in documents_to_process:
        document_path, file_md5 = document_info[document_id]
        LOGGER.info("Processing %s", document_path.name)
        chunks = extract_document_chunks(document_path, settings)
        document_metadata = {
            "document_id": document_id,
            "name": document_path.name,
            "md5": file_md5,
            "path": str(document_path),
        }
        metadata[document_id] = document_metadata

        short_chunk_records.extend(chunks["short"])
        long_chunk_records.extend(chunks["long"])
        stats.documents_processed += 1
        stats.short_chunks += len(chunks["short"])
        stats.long_chunks += len(chunks["long"])

    if reused_short_chunks:
        LOGGER.info("Reusing %d short chunks from previous runs", len(reused_short_chunks))
    if reused_long_chunks:
        LOGGER.info("Reusing %d long chunks from previous runs", len(reused_long_chunks))

    short_embeddings = None
    short_ids: List[int] = []
    if short_chunk_records:
        short_embeddings = embedding_client.normalize(
            embedding_client.embed_texts(
                (chunk.text for chunk in short_chunk_records),
                model=settings.model_embedding_short,
            )
        )
        short_ids = [chunk.embedding_id for chunk in short_chunk_records]

    if reused_short_embeddings is not None:
        short_embeddings = (
            np.concatenate([reused_short_embeddings, short_embeddings], axis=0)
            if short_embeddings is not None
            else reused_short_embeddings
        )
        short_ids = reused_short_ids + short_ids

    if short_embeddings is not None and short_ids:
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
        save_chunks(settings.short_chunks_path, reused_short_chunks + short_chunk_records)
    else:
        LOGGER.warning("No short chunks generated during ingestion")

    long_embeddings = None
    long_ids: List[int] = []
    if long_chunk_records:
        long_embeddings = embedding_client.normalize(
            embedding_client.embed_texts(
                (chunk.text for chunk in long_chunk_records),
                model=settings.model_embedding_long,
            )
        )
        long_ids = [chunk.embedding_id for chunk in long_chunk_records]

    if reused_long_embeddings is not None:
        long_embeddings = (
            np.concatenate([reused_long_embeddings, long_embeddings], axis=0)
            if long_embeddings is not None
            else reused_long_embeddings
        )
        long_ids = reused_long_ids + long_ids

    if long_embeddings is not None and long_ids:
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
        save_chunks(settings.long_chunks_path, reused_long_chunks + long_chunk_records)
    else:
        LOGGER.warning("No long chunks generated during ingestion")

    save_metadata(settings.metadata_path, metadata)
    LOGGER.info("Ingestion completed. Metadata stored at %s", settings.metadata_path)
    stats.log_summary()


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
