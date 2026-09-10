from __future__ import annotations

from typing import BinaryIO
from uuid import UUID

from app.storage.provider import StorageProvider
from app.storage.schemas import (
    StorageObject,
    StorageObjectType,
    StorageUpload,
)


class StorageService:
    """Application-level service for storage operations.

    This service provides a stable interface for the rest of
    the application while hiding the underlying storage provider.
    """

    def __init__(
        self,
        provider: StorageProvider,
    ) -> None:
        self.provider = provider

    async def upload_source(
        self,
        *,
        object_id: UUID,
        filename: str,
        content_type: str,
        content: bytes,
        content_hash: str | None = None,
        metadata: dict | None = None,
    ) -> StorageObject:
        """Upload an original source file."""

        upload = StorageUpload(
            object_id=object_id,
            object_type=StorageObjectType.SOURCE_FILE,
            filename=filename,
            content_type=content_type,
            content_hash=content_hash,
            metadata=metadata or {},
        )

        return await self.provider.upload(
            upload,
            content,
        )

    async def upload_intermediate(
        self,
        *,
        object_id: UUID,
        filename: str,
        content_type: str,
        content: bytes,
        content_hash: str | None = None,
        metadata: dict | None = None,
    ) -> StorageObject:
        """Upload an intermediate processing file."""

        upload = StorageUpload(
            object_id=object_id,
            object_type=StorageObjectType.INTERMEDIATE,
            filename=filename,
            content_type=content_type,
            content_hash=content_hash,
            metadata=metadata or {},
        )

        return await self.provider.upload(
            upload,
            content,
        )

    async def upload_artifact(
        self,
        *,
        object_id: UUID,
        filename: str,
        content_type: str,
        content: bytes,
        content_hash: str | None = None,
        metadata: dict | None = None,
    ) -> StorageObject:
        """Upload a generated communication artifact."""

        upload = StorageUpload(
            object_id=object_id,
            object_type=StorageObjectType.ARTIFACT,
            filename=filename,
            content_type=content_type,
            content_hash=content_hash,
            metadata=metadata or {},
        )

        return await self.provider.upload(
            upload,
            content,
        )

    async def download(
        self,
        storage_key: str,
    ) -> bytes:
        """Download an object using its storage key."""

        return await self.provider.download(
            storage_key,
        )

    async def exists(
        self,
        storage_key: str,
    ) -> bool:
        """Check whether an object exists."""

        return await self.provider.exists(
            storage_key,
        )

    async def delete(
        self,
        storage_key: str,
    ) -> None:
        """Delete an object."""

        await self.provider.delete(
            storage_key,
        )

    async def get_uri(
        self,
        storage_key: str,
    ) -> str:
        """Get the URI for an existing object."""

        return await self.provider.get_uri(
            storage_key,
        )