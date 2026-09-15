from __future__ import annotations

from app.models_ai.embeddings.exceptions import (
    EmbeddingErrorClassification,
    EmbeddingProviderError,
)
from app.models_ai.embeddings.retry import (
    EmbeddingRetryPolicy,
    RetryDecision,
)


class EmbeddingFailureService:
    """Coordinate provider-error handling and retry evaluation.

    This service does not execute retries.

    It provides the integration boundary between:

        provider error
        classification
        retry policy
    """

    def __init__(
        self,
        retry_policy: EmbeddingRetryPolicy | None = None,
    ) -> None:
        self.retry_policy = (
            retry_policy
            or EmbeddingRetryPolicy()
        )

    def evaluate_retry(
        self,
        *,
        error: EmbeddingProviderError,
    ) -> RetryDecision:
        """Evaluate retryability for a normalized provider error."""

        classification = self._get_classification(
            error
        )

        return self.retry_policy.evaluate(
            classification
        )

    @staticmethod
    def _get_classification(
        error: EmbeddingProviderError,
    ) -> EmbeddingErrorClassification:
        """Return the canonical classification from the error."""

        if error.classification is None:
            raise ValueError(
                "EmbeddingProviderError does not contain "
                "an error classification."
            )

        return error.classification