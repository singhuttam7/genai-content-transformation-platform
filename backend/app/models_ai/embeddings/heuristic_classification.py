from __future__ import annotations

import re
from typing import Final

from app.models_ai.embeddings.exceptions import (
    EmbeddingErrorCategory,
    EmbeddingErrorClassification,
)


class HeuristicEmbeddingErrorMapper:
    """Classify embedding errors using conservative heuristics."""

    _PATTERNS: Final[
        tuple[
            tuple[
                EmbeddingErrorCategory,
                float,
                bool,
                tuple[str, ...],
            ],
            ...,
        ]
    ] = (
        (
            EmbeddingErrorCategory.TIMEOUT,
            0.90,
            True,
            (
                r"\btimeout\b",
                r"\btimed\s*out\b",
                r"\brequest\s+timed\s*out\b",
                r"\bdeadline\s+exceeded\b",
            ),
        ),
        (
            EmbeddingErrorCategory.NETWORK,
            0.90,
            True,
            (
                r"\bconnection\s+refused\b",
                r"\bconnection\s+reset\b",
                r"\bnetwork\s+unreachable\b",
                r"\bnetwork\s+error\b",
                r"\bconnection\s+error\b",
                r"\bconnection\s+failed\b",
                r"\bname\s+resolution\b",
                r"\bdns\s+error\b",
            ),
        ),
        (
            EmbeddingErrorCategory.RATE_LIMIT,
            0.85,
            True,
            (
                r"\brate[\s_-]*limit\b",
                r"\btoo\s+many\s+requests\b",
                r"\bquota\s+exceeded\b",
                r"\brate\s+exceeded\b",
                r"\bthrottl(?:ed|ing|e)\b",
            ),
        ),
        (
            EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
            0.80,
            True,
            (
                r"\bservice\s+unavailable\b",
                r"\btemporarily\s+unavailable\b",
                r"\bserver\s+unavailable\b",
                r"\bbackend\s+unavailable\b",
                r"\bservice\s+overloaded\b",
                r"\btemporarily\s+overloaded\b",
            ),
        ),
        (
            EmbeddingErrorCategory.AUTHENTICATION,
            0.90,
            False,
            (
                r"\bauthentication\s+failed\b",
                r"\bunauthenticated\b",
                r"\binvalid\s+api[\s_-]*key\b",
                r"\bapi[\s_-]*key\s+invalid\b",
                r"\bcredential(?:s)?\s+invalid\b",
                r"\bmissing\s+api[\s_-]*key\b",
            ),
        ),
        (
            EmbeddingErrorCategory.AUTHORIZATION,
            0.90,
            False,
            (
                r"\bauthorization\s+failed\b",
                r"\bpermission\s+denied\b",
                r"\baccess\s+denied\b",
                r"\bforbidden\b",
                r"\bnot\s+authorized\b",
                r"\bunauthorized\s+access\b",
            ),
        ),
        (
            EmbeddingErrorCategory.MODEL_NOT_FOUND,
            0.90,
            False,
            (
                r"\bmodel\s+not\s+found\b",
                r"\bmodel\s+does\s+not\s+exist\b",
                r"\bunknown\s+model\b",
                r"\binvalid\s+model\b",
                r"\bmodel\s+unavailable\b",
            ),
        ),
        (
            EmbeddingErrorCategory.INPUT,
            0.80,
            False,
            (
                r"\binvalid\s+input\b",
                r"\binvalid\s+request\b",
                r"\binvalid\s+text\b",
                r"\bvalidation\s+error\b",
                r"\binput\s+validation\b",
                r"\bmalformed\s+request\b",
            ),
        ),
        (
            EmbeddingErrorCategory.CONFIGURATION,
            0.85,
            False,
            (
                r"\bconfiguration\s+error\b",
                r"\bconfiguration\s+invalid\b",
                r"\bmisconfigured\b",
                r"\bmissing\s+configuration\b",
                r"\binvalid\s+configuration\b",
            ),
        ),
        (
            EmbeddingErrorCategory.DIMENSION,
            0.95,
            False,
            (
                r"\bdimension\s+mismatch\b",
                r"\binvalid\s+embedding\s+dimension\b",
                r"\bembedding\s+dimension\b",
                r"\bvector\s+dimension\b",
            ),
        ),
    )

    def classify(
        self,
        error: Exception,
    ) -> EmbeddingErrorClassification | None:
        """Classify an exception using its type and message.

        Returns None when no supported heuristic matches.
        Structured signals such as status_code are intentionally
        ignored by this mapper.
        """

        message = self._normalize_message(error)

        if not message:
            return self._classify_exception_type(error)

        for (
            category,
            confidence,
            retryable,
            patterns,
        ) in self._PATTERNS:
            if any(
                re.search(
                    pattern,
                    message,
                    flags=re.IGNORECASE,
                )
                for pattern in patterns
            ):
                return EmbeddingErrorClassification(
                    category=category,
                    retryable=retryable,
                    confidence=confidence,
                    reason=(
                        f"Heuristic message pattern matched "
                        f"{category.value}."
                    ),
                    metadata={
                        "classification_source": "heuristic",
                        "signal": "message",
                    },
                )

        return self._classify_exception_type(error)

    @staticmethod
    def _normalize_message(
        error: Exception,
    ) -> str:
        """Normalize an exception message for matching."""

        try:
            message = str(error)
        except Exception:
            return ""

        return " ".join(
            message.strip().split()
        )

    @staticmethod
    def _classify_exception_type(
        error: Exception,
    ) -> EmbeddingErrorClassification | None:
        """Use conservative built-in exception-type signals."""

        if isinstance(error, TimeoutError):
            return EmbeddingErrorClassification(
                category=EmbeddingErrorCategory.TIMEOUT,
                retryable=True,
                confidence=0.90,
                reason=(
                    "Built-in TimeoutError indicates a timeout."
                ),
                metadata={
                    "classification_source": "heuristic",
                    "signal": "exception_type",
                },
            )

        if isinstance(error, ConnectionError):
            return EmbeddingErrorClassification(
                category=EmbeddingErrorCategory.NETWORK,
                retryable=True,
                confidence=0.90,
                reason=(
                    "Built-in ConnectionError indicates "
                    "a network failure."
                ),
                metadata={
                    "classification_source": "heuristic",
                    "signal": "exception_type",
                },
            )

        return None