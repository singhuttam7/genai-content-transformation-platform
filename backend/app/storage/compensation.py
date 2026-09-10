from __future__ import annotations

from typing import Protocol


class StorageCompensator(Protocol):
    """
    Contract for compensating storage operations.

    A compensation operation is used when a storage object was
    successfully created but a later operation failed.

    Example:
        Storage upload -> SUCCESS
        Database       -> FAILURE
        Compensation   -> delete uploaded object
    """

    async def compensate_upload(
        self,
        storage_key: str,
    ) -> None:
        """
        Compensate for a previously successful upload.

        Args:
            storage_key: Storage key of the object that must be
                removed.

        Raises:
            Exception: If the compensation operation itself fails.
        """
        ...


class StorageCompensationService:
    """
    Application-level service for storage compensation.

    This service deliberately depends on the StorageProvider contract
    through StorageService rather than directly accessing the filesystem
    or an external object-storage SDK.
    """

    def __init__(self, storage_service) -> None:
        self.storage_service = storage_service

    async def compensate_upload(
        self,
        storage_key: str,
    ) -> None:
        """
        Delete a previously uploaded storage object.
        """

        if not storage_key:
            raise ValueError(
                "Storage key is required for compensation."
            )

        await self.storage_service.delete(storage_key)