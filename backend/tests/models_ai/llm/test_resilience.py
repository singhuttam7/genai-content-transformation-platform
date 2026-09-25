import pytest

from app.models_ai.llm.resilience import LLMRetryPolicy


def test_default_policy_values() -> None:
    policy = LLMRetryPolicy()

    assert policy.max_attempts == 3
    assert policy.initial_backoff_seconds == 0.5
    assert policy.backoff_multiplier == 2.0
    assert policy.max_backoff_seconds == 8.0
    assert policy.max_retries == 2


def test_max_retries_is_total_attempts_minus_one() -> None:
    policy = LLMRetryPolicy(
        max_attempts=5,
    )

    assert policy.max_retries == 4


@pytest.mark.parametrize(
    "attempt_number, expected",
    [
        (1, True),
        (2, True),
        (3, False),
    ],
)
def test_retry_decision_for_retryable_failure(
    attempt_number: int,
    expected: bool,
) -> None:
    policy = LLMRetryPolicy(
        max_attempts=3,
    )

    assert (
        policy.should_retry(
            attempt_number=attempt_number,
            retryable=True,
        )
        is expected
    )


@pytest.mark.parametrize(
    "attempt_number",
    [1, 2, 3],
)
def test_non_retryable_failure_is_never_retried(
    attempt_number: int,
) -> None:
    policy = LLMRetryPolicy(
        max_attempts=3,
    )

    assert (
        policy.should_retry(
            attempt_number=attempt_number,
            retryable=False,
        )
        is False
    )


def test_single_attempt_policy_disables_retry() -> None:
    policy = LLMRetryPolicy(
        max_attempts=1,
    )

    assert policy.max_retries == 0

    assert (
        policy.should_retry(
            attempt_number=1,
            retryable=True,
        )
        is False
    )


@pytest.mark.parametrize(
    "attempt_number, expected",
    [
        (1, 0.5),
        (2, 1.0),
        (3, 2.0),
        (4, 4.0),
    ],
)
def test_exponential_backoff(
    attempt_number: int,
    expected: float,
) -> None:
    policy = LLMRetryPolicy(
        initial_backoff_seconds=0.5,
        backoff_multiplier=2.0,
        max_backoff_seconds=8.0,
    )

    assert (
        policy.backoff_seconds(
            attempt_number=attempt_number,
        )
        == expected
    )


def test_backoff_is_capped() -> None:
    policy = LLMRetryPolicy(
        initial_backoff_seconds=1.0,
        backoff_multiplier=2.0,
        max_backoff_seconds=3.0,
    )

    assert (
        policy.backoff_seconds(
            attempt_number=1,
        )
        == 1.0
    )

    assert (
        policy.backoff_seconds(
            attempt_number=2,
        )
        == 2.0
    )

    assert (
        policy.backoff_seconds(
            attempt_number=3,
        )
        == 3.0
    )

    assert (
        policy.backoff_seconds(
            attempt_number=4,
        )
        == 3.0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "max_attempts": 0,
        },
        {
            "max_attempts": -1,
        },
        {
            "initial_backoff_seconds": -0.1,
        },
        {
            "backoff_multiplier": 0.5,
        },
        {
            "max_backoff_seconds": -1.0,
        },
        {
            "initial_backoff_seconds": 2.0,
            "max_backoff_seconds": 1.0,
        },
    ],
)
def test_invalid_policy_values_are_rejected(
    kwargs: dict,
) -> None:
    with pytest.raises(ValueError):
        LLMRetryPolicy(**kwargs)


@pytest.mark.parametrize(
    "attempt_number",
    [0, -1],
)
def test_invalid_attempt_number_is_rejected(
    attempt_number: int,
) -> None:
    policy = LLMRetryPolicy()

    with pytest.raises(ValueError):
        policy.should_retry(
            attempt_number=attempt_number,
            retryable=True,
        )

    with pytest.raises(ValueError):
        policy.backoff_seconds(
            attempt_number=attempt_number,
        )


def test_policy_is_immutable() -> None:
    policy = LLMRetryPolicy()

    with pytest.raises(AttributeError):
        policy.max_attempts = 5