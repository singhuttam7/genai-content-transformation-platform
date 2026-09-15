from __future__ import annotations

from app.models_ai.embeddings.exceptions import (
    EmbeddingErrorCategory,
    EmbeddingErrorClassification,
)
from app.models_ai.embeddings.retry.schemas import (
    RetryDecision,
)


class EmbeddingRetryPolicy:
    """Determine retryability from provider-independent error categories.

    The policy intentionally does not depend on provider SDKs, HTTP
    clients, network libraries, or retry execution mechanisms.

    Retryability is determined from the classified error category,
    not from the classifier's retryable field.
    """

    RETRYABLE_CATEGORIES = frozenset(
        {
            EmbeddingErrorCategory.RATE_LIMIT,
            EmbeddingErrorCategory.TIMEOUT,
            EmbeddingErrorCategory.NETWORK,
            EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
        }
    )

    def evaluate(
        self,
        classification: EmbeddingErrorClassification,
    ) -> RetryDecision:
        """Return a deterministic retry decision."""

        retryable = (
            classification.category
            in self.RETRYABLE_CATEGORIES
        )

        if retryable:
            reason = (
                f"Category '{classification.category.value}' "
                "is considered transient by the embedding "
                "retry policy."
            )
        else:
            reason = (
                f"Category '{classification.category.value}' "
                "is considered non-retryable by the embedding "
                "retry policy."
            )

        return RetryDecision(
            retryable=retryable,
            category=classification.category,
            reason=reason,
            metadata={
                "policy": "embedding_default",
                "policy_version": "1.0",
            },
        )