from __future__ import annotations


class EmbeddingError(Exception):
    """Base exception for embedding operations."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when embedding configuration is invalid."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when an embedding provider fails."""


class EmbeddingDimensionError(EmbeddingError):
    """Raised when embedding dimensions are inconsistent."""


class EmbeddingInputError(EmbeddingError):
    """Raised when embedding input is invalid."""


class EmbeddingBatchError(EmbeddingError):
    """Raised when a batch embedding operation fails."""