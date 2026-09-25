from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LLMRetryPolicy:
    """
    Provider-independent retry policy for LLM operations.

    max_attempts represents the total number of provider attempts,
    including the initial attempt.
    """

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.5
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least 1."
            )

        if self.initial_backoff_seconds < 0:
            raise ValueError(
                "initial_backoff_seconds must be non-negative."
            )

        if self.backoff_multiplier < 1:
            raise ValueError(
                "backoff_multiplier must be at least 1."
            )

        if self.max_backoff_seconds < 0:
            raise ValueError(
                "max_backoff_seconds must be non-negative."
            )

        if (
            self.max_backoff_seconds
            < self.initial_backoff_seconds
        ):
            raise ValueError(
                "max_backoff_seconds must be greater than or "
                "equal to initial_backoff_seconds."
            )

    @property
    def max_retries(self) -> int:
        """
        Return the maximum number of retries after the
        initial provider attempt.
        """

        return self.max_attempts - 1

    def should_retry(
        self,
        *,
        attempt_number: int,
        retryable: bool,
    ) -> bool:
        """
        Determine whether another attempt is permitted.

        attempt_number is one-based and represents the attempt
        that has just failed.
        """

        if not retryable:
            return False

        if attempt_number < 1:
            raise ValueError(
                "attempt_number must be at least 1."
            )

        return attempt_number < self.max_attempts

    def backoff_seconds(
        self,
        *,
        attempt_number: int,
    ) -> float:
        """
        Calculate the bounded backoff after a failed attempt.

        attempt_number is one-based and represents the attempt
        that has just failed.
        """

        if attempt_number < 1:
            raise ValueError(
                "attempt_number must be at least 1."
            )

        delay = (
            self.initial_backoff_seconds
            * (
                self.backoff_multiplier
                ** (attempt_number - 1)
            )
        )

        return min(
            delay,
            self.max_backoff_seconds,
        )