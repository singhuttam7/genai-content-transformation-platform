from __future__ import annotations

from functools import lru_cache

from app.storage.compensation import StorageCompensationService
from app.storage.dependencies import get_storage_service


@lru_cache
def get_storage_compensation_service() -> StorageCompensationService:
    """
    Return the application-level storage compensation service.
    """

    return StorageCompensationService(
        storage_service=get_storage_service(),
    )