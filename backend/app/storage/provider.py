from typing import Protocol

from app.storage.schemas import (
    StorageObject,
    StorageUpload,
)


class StorageProvider(Protocol):
    """Backend-independent contract for object storage."""

    async def upload(
        self,
        upload: StorageUpload,
        content: bytes,
    ) -> StorageObject:
        """Store content and return its storage metadata."""
        ...

    async def download(
        self,
        storage_key: str,
    ) -> bytes:
        """Retrieve stored content."""
        ...

    async def exists(
        self,
        storage_key: str,
    ) -> bool:
        """Check whether an object exists."""
        ...

    async def delete(
        self,
        storage_key: str,
    ) -> None:
        """Delete a stored object."""
        ...

    async def get_uri(
        self,
        storage_key: str,
    ) -> str:
        """Return a backend-specific URI for an object."""
        ...