from __future__ import annotations

from abc import ABC, abstractmethod

from app.models_ai.embeddings.exceptions import (
    EmbeddingErrorClassification,
)


class EmbeddingErrorClassifier(ABC):
    """Provider-independent interface for embedding error classification."""

    @abstractmethod
    def classify(
        self,
        error: Exception,
    ) -> EmbeddingErrorClassification:
        """Classify an embedding exception.

        Implementations must:

        - be deterministic;
        - perform no network I/O;
        - not mutate the supplied exception;
        - return a provider-independent classification;
        - never raise provider-specific exceptions as part of
          normal classification.
        """
        raise NotImplementedError