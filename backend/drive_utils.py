"""Utilities to interact with Google Drive."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .logging_config import get_logger

try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
except Exception:  # pragma: no cover - optional dependency
    Credentials = None  # type: ignore
    build = None  # type: ignore
    MediaIoBaseDownload = None  # type: ignore


LOGGER = get_logger(__name__)


@dataclass
class GoogleDriveClient:
    """Minimal client for downloading PDF files from a Drive folder."""

    credentials_path: Optional[Path]
    folder_id: str
    scopes: Iterable[str] = ("https://www.googleapis.com/auth/drive.readonly",)

    def _build_service(self):
        if not self.folder_id:
            raise ValueError("GOOGLE_DRIVE_FOLDER_ID is required for Drive sync")
        if Credentials is None or build is None:
            raise RuntimeError(
                "google-api-python-client is not installed. Install it to sync Drive files."
            )
        if self.credentials_path is None or not self.credentials_path.exists():
            raise FileNotFoundError(
                "Service account credentials JSON not found. Provide GOOGLE_APPLICATION_CREDENTIALS."
            )
        creds = Credentials.from_service_account_file(str(self.credentials_path), scopes=self.scopes)
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    def list_pdfs(self) -> List[dict]:
        """Return metadata for PDF files inside the configured folder."""

        service = self._build_service()
        query = f"'{self.folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        results = (
            service.files()
            .list(q=query, fields="files(id, name, modifiedTime, md5Checksum)")
            .execute()
        )
        return results.get("files", [])

    def download_file(self, file_id: str, destination: Path) -> None:
        """Download a file from Drive into the destination path."""

        service = self._build_service()
        request = service.files().get_media(fileId=file_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fh = io.FileIO(destination, "wb")
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:  # pragma: no cover - network loop
            status, done = downloader.next_chunk()
            if status:
                LOGGER.info("Download %s: %.1f%%", destination.name, status.progress() * 100)
        fh.close()


def compute_md5(path: Path) -> str:
    """Return the MD5 checksum for the provided file."""

    hash_md5 = hashlib.md5()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


__all__ = ["GoogleDriveClient", "compute_md5"]
