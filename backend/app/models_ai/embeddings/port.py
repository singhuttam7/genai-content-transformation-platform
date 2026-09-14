from __future__ import annotations

from abc import ABC, abstractmethod

from app.models_ai.embeddings.schemas import (
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingRequest,
    EmbeddingVector,
)


class EmbeddingPort(ABC):
    """
    Provider-independent interface for embedding generation.

    Implementations may use local models, hosted APIs, or other
    embedding backends. Consumers must depend only on this interface
    and the embedding contracts.
    """

    @abstractmethod
    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        """
        Generate an embedding for a single input.

        Parameters
        ----------
        request:
            Provider-independent embedding request.

        Returns
        -------
        EmbeddingVector
            The generated embedding and model information.

        Raises
        ------
        EmbeddingError
            Implementations may raise provider-independent embedding
            exceptions for configuration, input, provider, or
            dimension failures.
        """

    @abstractmethod
    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        """
        Generate embeddings for a batch of inputs.

        Parameters
        ----------
        request:
            Provider-independent batch embedding request.

        Returns
        -------
        EmbeddingBatchResult
            Embeddings together with model and batch metadata.

        Raises
        ------
        EmbeddingError
            Implementations may raise provider-independent embedding
            exceptions for configuration, input, provider, or
            dimension failures.
        """