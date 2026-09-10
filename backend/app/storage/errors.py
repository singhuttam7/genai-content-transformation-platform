from __future__ import annotations


class StorageCompensationError(RuntimeError):
    """
    Raised when a storage compensation operation fails.

    This indicates that an earlier operation succeeded, but its
    compensating cleanup operation could not be completed.
    """

    def __init__(
        self,
        *,
        storage_key: str,
        original_error: Exception,
        compensation_error: Exception,
    ) -> None:
        self.storage_key = storage_key
        self.original_error = original_error
        self.compensation_error = compensation_error

        message = (
            "Storage compensation failed. "
            f"Storage key: {storage_key}. "
            f"Original error: {original_error}. "
            f"Compensation error: {compensation_error}."
        )

        super().__init__(message)