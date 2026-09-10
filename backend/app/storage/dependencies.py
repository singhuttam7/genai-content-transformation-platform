from __future__ import annotations

from functools import lru_cache

from app.storage.factory import create_storage_provider
from app.storage.service import StorageService


@lru_cache
def get_storage_service() -> StorageService:
    """Return the application-wide storage service.

    The storage provider is created from the configured
    storage backend and reused across the application.
    """

    provider = create_storage_provider()

    return StorageService(provider)