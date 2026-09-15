from __future__ import annotations

from typing import Any

from app.models_ai.embeddings.exceptions import (
    EmbeddingErrorCategory,
    EmbeddingErrorClassification,
)
from app.models_ai.embeddings.heuristic_classification import (
    HeuristicEmbeddingErrorMapper,
)


class StructuredEmbeddingErrorMapper:
    """Classify embedding errors using reliable structured signals."""

    STATUS_CATEGORY_MAP: dict[int, EmbeddingErrorCategory] = {
        400: EmbeddingErrorCategory.INPUT,
        401: EmbeddingErrorCategory.AUTHENTICATION,
        403: EmbeddingErrorCategory.AUTHORIZATION,
        404: EmbeddingErrorCategory.MODEL_NOT_FOUND,
        408: EmbeddingErrorCategory.TIMEOUT,
        429: EmbeddingErrorCategory.RATE_LIMIT,
        500: EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
        502: EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
        503: EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
        504: EmbeddingErrorCategory.TIMEOUT,
    }

    RETRYABLE_CATEGORIES = frozenset(
        {
            EmbeddingErrorCategory.RATE_LIMIT,
            EmbeddingErrorCategory.TIMEOUT,
            EmbeddingErrorCategory.NETWORK,
            EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
        }
    )

    def classify(
        self,
        error: Exception,
    ) -> EmbeddingErrorClassification | None:
        """Classify an exception when a supported structured signal exists.

        Returns None when the exception does not contain a supported
        structured signal. Message-based heuristic classification is
        intentionally outside this mapper.
        """

        status_code = self._extract_status_code(error)

        if status_code is None:
            return None

        category = self.STATUS_CATEGORY_MAP.get(status_code)

        if category is None:
            return None

        metadata: dict[str, Any] = {
            "classification_source": "structured",
            "status_code": status_code,
        }

        error_code = self._extract_optional_string(
            error,
            "error_code",
        )

        if error_code is not None:
            metadata["error_code"] = error_code

        error_type = self._extract_optional_string(
            error,
            "error_type",
        )

        if error_type is not None:
            metadata["error_type"] = error_type

        return EmbeddingErrorClassification(
            category=category,
            retryable=category in self.RETRYABLE_CATEGORIES,
            confidence=1.0,
            reason=(
                f"Structured status code {status_code} "
                f"mapped to {category.value}."
            ),
            metadata=metadata,
        )

    @staticmethod
    def _extract_status_code(
        error: Exception,
    ) -> int | None:
        """Extract a valid HTTP-like status code if available.

        Provider exceptions are treated as untrusted boundary objects.
        Attribute access itself may fail, so structured fields must be
        read defensively.
        """

        try:
            value = getattr(
                error,
                "status_code",
                None,
            )
        except Exception:
            return None

        if isinstance(value, bool):
            return None

        if not isinstance(value, int):
            return None

        if not 100 <= value <= 599:
            return None

        return value

    @staticmethod
    def _extract_optional_string(
        error: Exception,
        attribute_name: str,
    ) -> str | None:
        """Safely extract a non-empty string attribute.

        Provider exception objects are treated as untrusted inputs.
        Attribute access may itself raise an exception.
        """

        try:
            value = getattr(
                error,
                attribute_name,
                None,
            )
        except Exception:
            return None

        if not isinstance(value, str):
            return None

        normalized = value.strip()

        return normalized or None


class CompositeEmbeddingErrorClassifier:
    """Classify embedding errors using structured and heuristic signals.

    Classification precedence:

        1. Structured provider signals
        2. Heuristic signals
        3. UNKNOWN fallback

    The classifier is deterministic and provider-independent.
    """

    def __init__(
        self,
        *,
        structured_mapper: StructuredEmbeddingErrorMapper | None = None,
        heuristic_mapper: HeuristicEmbeddingErrorMapper | None = None,
    ) -> None:
        self.structured_mapper = (
            structured_mapper
            or StructuredEmbeddingErrorMapper()
        )

        self.heuristic_mapper = (
            heuristic_mapper
            or HeuristicEmbeddingErrorMapper()
        )

    def classify(
        self,
        error: Exception,
    ) -> EmbeddingErrorClassification:
        """Return the highest-confidence supported classification."""

        structured_result = self.structured_mapper.classify(
            error
        )

        if structured_result is not None:
            return structured_result

        heuristic_result = self.heuristic_mapper.classify(
            error
        )

        if heuristic_result is not None:
            return heuristic_result

        return EmbeddingErrorClassification(
            category=EmbeddingErrorCategory.UNKNOWN,
            retryable=False,
            confidence=0.0,
            reason=(
                "No supported structured or heuristic "
                "signal was found."
            ),
            metadata={
                "classification_source": "fallback",
            },
        )