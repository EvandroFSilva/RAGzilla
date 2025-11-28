"""Application configuration helpers."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central application settings loaded from environment variables."""

    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    google_drive_folder_id: str = Field(default="", env="GOOGLE_DRIVE_FOLDER_ID")
    # Optional path to Google service account credentials JSON
    google_application_credentials: Path | None = Field(
        default=None, env="GOOGLE_APPLICATION_CREDENTIALS"
    )

    data_dir: Path = Field(default=Path("data"))
    short_index_dir: Path = Field(default=Path("data/faiss_index_short"))
    long_index_dir: Path = Field(default=Path("data/faiss_index_long"))
    metadata_path: Path = Field(default=Path("data/metadata.jsonl"))
    short_chunks_path: Path = Field(default=Path("data/chunks_short.jsonl"))
    long_chunks_path: Path = Field(default=Path("data/chunks_long.jsonl"))

    short_chunk_size: int = 20
    short_chunk_overlap: int = 5
    long_chunk_size: int = 500
    long_chunk_overlap: int = 50

    faiss_factory: str = Field(
        default="Flat",
        description="FAISS index factory string (without metric suffix)",
    )
    faiss_metric: str = Field(
        default="cosine",
        description="Similarity metric: cosine via inner product + L2 normalisation",
    )

    model_embedding_short: str = Field(default="text-embedding-3-small")
    model_embedding_long: str = Field(default="text-embedding-3-small")
    model_generation: str = Field(default="gpt-4o-mini")

    allowed_extensions: List[str] = Field(default_factory=lambda: [".pdf"])

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""

    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.short_index_dir.mkdir(parents=True, exist_ok=True)
    settings.long_index_dir.mkdir(parents=True, exist_ok=True)
    settings.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    settings.short_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    settings.long_chunks_path.parent.mkdir(parents=True, exist_ok=True)
    return settings


__all__ = ["Settings", "get_settings"]
