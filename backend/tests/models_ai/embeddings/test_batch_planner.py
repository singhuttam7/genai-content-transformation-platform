from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    EmbeddingInputError,
    EmbeddingRequest,
)
from app.models_ai.embeddings.batching import (
    EmbeddingBatchPlan,
    EmbeddingBatchPlanner,
)


def make_requests(count: int) -> list[EmbeddingRequest]:
    return [
        EmbeddingRequest(
            text=f"document chunk {index}",
            metadata={
                "original_index": index,
            },
        )
        for index in range(count)
    ]


def test_planner_creates_single_batch_when_input_fits() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(3)

    plan = planner.plan(
        requests,
        batch_size=10,
    )

    assert isinstance(plan, EmbeddingBatchPlan)
    assert plan.input_count == 3
    assert plan.batch_count == 1
    assert len(plan.batches) == 1
    assert len(plan.batches[0]) == 3


def test_planner_splits_requests_into_configured_batch_size() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(10)

    plan = planner.plan(
        requests,
        batch_size=3,
    )

    assert plan.batch_count == 4
    assert [len(batch) for batch in plan.batches] == [
        3,
        3,
        3,
        1,
    ]


def test_planner_handles_exact_batch_boundary() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(9)

    plan = planner.plan(
        requests,
        batch_size=3,
    )

    assert plan.batch_count == 3
    assert [len(batch) for batch in plan.batches] == [
        3,
        3,
        3,
    ]


def test_planner_handles_single_item_batches() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(4)

    plan = planner.plan(
        requests,
        batch_size=1,
    )

    assert plan.batch_count == 4
    assert [len(batch) for batch in plan.batches] == [
        1,
        1,
        1,
        1,
    ]


def test_planner_preserves_request_order() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(7)

    plan = planner.plan(
        requests,
        batch_size=3,
    )

    flattened = [
        request
        for batch in plan.batches
        for request in batch
    ]

    assert flattened == requests


def test_planner_preserves_metadata() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(5)

    plan = planner.plan(
        requests,
        batch_size=2,
    )

    flattened = [
        request
        for batch in plan.batches
        for request in batch
    ]

    assert [
        request.metadata["original_index"]
        for request in flattened
    ] == [0, 1, 2, 3, 4]


def test_planner_preserves_input_count() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(17)

    plan = planner.plan(
        requests,
        batch_size=5,
    )

    assert plan.input_count == len(requests)


def test_planner_preserves_batch_count() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(17)

    plan = planner.plan(
        requests,
        batch_size=5,
    )

    assert plan.batch_count == len(plan.batches)


def test_planner_does_not_mutate_input_list() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(8)
    original = list(requests)

    planner.plan(
        requests,
        batch_size=3,
    )

    assert requests == original


def test_planner_is_deterministic() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(11)

    first = planner.plan(
        requests,
        batch_size=4,
    )

    second = planner.plan(
        requests,
        batch_size=4,
    )

    assert first == second


def test_planner_rejects_empty_requests() -> None:
    planner = EmbeddingBatchPlanner()

    with pytest.raises(
        EmbeddingInputError,
        match="At least one embedding request is required",
    ):
        planner.plan(
            [],
            batch_size=4,
        )


@pytest.mark.parametrize(
    "batch_size",
    [
        0,
        -1,
        -10,
    ],
)
def test_planner_rejects_invalid_batch_size(
    batch_size: int,
) -> None:
    planner = EmbeddingBatchPlanner()

    with pytest.raises(
        EmbeddingInputError,
        match="batch_size must be greater than or equal to 1",
    ):
        planner.plan(
            make_requests(3),
            batch_size=batch_size,
        )


def test_planner_supports_large_input() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(1000)

    plan = planner.plan(
        requests,
        batch_size=32,
    )

    assert plan.input_count == 1000
    assert plan.batch_count == 32
    assert len(plan.batches[-1]) == 8


def test_planner_supports_unicode_content() -> None:
    planner = EmbeddingBatchPlanner()

    requests = [
        EmbeddingRequest(text="नमस्ते"),
        EmbeddingRequest(text="こんにちは"),
        EmbeddingRequest(text="你好"),
        EmbeddingRequest(text="Hello"),
    ]

    plan = planner.plan(
        requests,
        batch_size=2,
    )

    flattened = [
        request
        for batch in plan.batches
        for request in batch
    ]

    assert [request.text for request in flattened] == [
        "नमस्ते",
        "こんにちは",
        "你好",
        "Hello",
    ]


def test_plan_is_immutable() -> None:
    planner = EmbeddingBatchPlanner()

    plan = planner.plan(
        make_requests(3),
        batch_size=2,
    )

    with pytest.raises(Exception):
        plan.batch_size = 10  # type: ignore[misc]