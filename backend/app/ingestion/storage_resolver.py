from __future__ import annotations

from app.ingestion.content_resolver import InputContentResolver
from app.ingestion.schemas import IngestionRequest
from app.storage.service import StorageService


class StorageBackedContentResolver(InputContentResolver):
    """Resolve inline or storage-backed ingestion content."""

    def __init__(self, storage: StorageService) -> None:
        self.storage = storage

    async def resolve(self, request: IngestionRequest) -> bytes:
        # Inline binary content
        if isinstance(request.content, bytes):
            return request.content

        # Storage-backed content
        if request.storage_key:
            return await self.storage.download(request.storage_key)

        raise ValueError(
            "No binary content or storage reference is available."
        )