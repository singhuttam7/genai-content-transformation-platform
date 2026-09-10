from __future__ import annotations

from app.core.config import settings
from app.storage.files.local import LocalStorageProvider
from app.storage.provider import StorageProvider


def create_storage_provider() -> StorageProvider:
    """Create the configured storage provider.

    The provider is selected through the STORAGE_BACKEND
    application setting.

    Supported backends:
        - local

    Future backends:
        - s3
        - minio
        - r2
    """

    backend = settings.storage_backend.strip().lower()

    if backend == "local":
        return LocalStorageProvider()

    raise ValueError(
        f"Unsupported storage backend: {settings.storage_backend}"
    )