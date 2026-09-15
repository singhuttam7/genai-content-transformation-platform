from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    EmbeddingInputError,
    EmbeddingRequest,
)
from app.models_ai.embeddings.batching import (
    EmbeddingBatchAggregator,
    EmbeddingBatchPlanner,
)


def make_requests(count: int) -> list[EmbeddingRequest]:
    return [
        EmbeddingRequest(
            text=f"chunk {index}",
            metadata={"index": index},
        )
        for index in range(count)
    ]


def test_single_request_creates_single_batch() -> None:
    planner = EmbeddingBatchPlanner()

    plan = planner.plan(
        make_requests(1),
        batch_size=32,
    )

    assert plan.input_count == 1
    assert plan.batch_count == 1
    assert len(plan.batches[0]) == 1


def test_batch_size_one_creates_one_batch_per_request() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(5)

    plan = planner.plan(
        requests,
        batch_size=1,
    )

    assert plan.batch_count == 5
    assert [len(batch) for batch in plan.batches] == [
        1,
        1,
        1,
        1,
        1,
    ]


def test_batch_size_larger_than_input_creates_single_batch() -> None:
    planner = EmbeddingBatchPlanner()

    plan = planner.plan(
        make_requests(4),
        batch_size=100,
    )

    assert plan.batch_count == 1
    assert len(plan.batches[0]) == 4


def test_exact_batch_boundary_has_no_partial_batch() -> None:
    planner = EmbeddingBatchPlanner()

    plan = planner.plan(
        make_requests(12),
        batch_size=4,
    )

    assert plan.batch_count == 3
    assert [len(batch) for batch in plan.batches] == [
        4,
        4,
        4,
    ]


def test_non_exact_batch_boundary_has_correct_remainder() -> None:
    planner = EmbeddingBatchPlanner()

    plan = planner.plan(
        make_requests(13),
        batch_size=4,
    )

    assert plan.batch_count == 4
    assert [len(batch) for batch in plan.batches] == [
        4,
        4,
        4,
        1,
    ]


def test_duplicate_texts_remain_distinct_requests() -> None:
    planner = EmbeddingBatchPlanner()

    requests = [
        EmbeddingRequest(
            text="same text",
            metadata={"index": 0},
        ),
        EmbeddingRequest(
            text="same text",
            metadata={"index": 1},
        ),
        EmbeddingRequest(
            text="same text",
            metadata={"index": 2},
        ),
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

    assert len(flattened) == 3
    assert [request.metadata["index"] for request in flattened] == [
        0,
        1,
        2,
    ]


def test_empty_text_is_preserved_by_batch_planner() -> None:
    planner = EmbeddingBatchPlanner()

    requests = [
        EmbeddingRequest(text=""),
        EmbeddingRequest(text="normal text"),
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
        "",
        "normal text",
    ]


def test_unicode_text_is_preserved() -> None:
    planner = EmbeddingBatchPlanner()

    requests = [
        EmbeddingRequest(text="नमस्ते दुनिया"),
        EmbeddingRequest(text="こんにちは世界"),
        EmbeddingRequest(text="你好世界"),
        EmbeddingRequest(text="مرحبا بالعالم"),
        EmbeddingRequest(text="Hello world"),
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
        "नमस्ते दुनिया",
        "こんにちは世界",
        "你好世界",
        "مرحبا بالعالم",
        "Hello world",
    ]


def test_metadata_is_preserved_without_modification() -> None:
    planner = EmbeddingBatchPlanner()

    requests = [
        EmbeddingRequest(
            text="first",
            metadata={
                "chunk_index": 0,
                "language": "en",
            },
        ),
        EmbeddingRequest(
            text="second",
            metadata={
                "chunk_index": 1,
                "language": "hi",
            },
        ),
    ]

    original_metadata = [
        dict(request.metadata)
        for request in requests
    ]

    plan = planner.plan(
        requests,
        batch_size=1,
    )

    flattened = [
        request
        for batch in plan.batches
        for request in batch
    ]

    assert [
        dict(request.metadata)
        for request in flattened
    ] == original_metadata


def test_large_request_collection_is_partitioned_correctly() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(10_000)

    plan = planner.plan(
        requests,
        batch_size=128,
    )

    assert plan.input_count == 10_000
    assert sum(
        len(batch)
        for batch in plan.batches
    ) == 10_000

    assert len(plan.batches[-1]) == 16


def test_large_request_collection_preserves_order() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(5000)

    plan = planner.plan(
        requests,
        batch_size=127,
    )

    flattened = [
        request
        for batch in plan.batches
        for request in batch
    ]

    assert flattened == requests


def test_planner_rejects_empty_collection() -> None:
    planner = EmbeddingBatchPlanner()

    with pytest.raises(
        EmbeddingInputError,
        match="At least one embedding request is required",
    ):
        planner.plan(
            [],
            batch_size=32,
        )


def test_planner_rejects_zero_batch_size() -> None:
    planner = EmbeddingBatchPlanner()

    with pytest.raises(
        EmbeddingInputError,
        match="batch_size must be greater than or equal to 1",
    ):
        planner.plan(
            make_requests(3),
            batch_size=0,
        )


def test_planner_rejects_negative_batch_size() -> None:
    planner = EmbeddingBatchPlanner()

    with pytest.raises(
        EmbeddingInputError,
        match="batch_size must be greater than or equal to 1",
    ):
        planner.plan(
            make_requests(3),
            batch_size=-5,
        )


def test_planner_is_repeatable_for_same_input() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(100)

    first = planner.plan(
        requests,
        batch_size=17,
    )

    second = planner.plan(
        requests,
        batch_size=17,
    )

    assert first == second


def test_planner_does_not_modify_request_order() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(20)

    original_texts = [
        request.text
        for request in requests
    ]

    planner.plan(
        requests,
        batch_size=6,
    )

    assert [
        request.text
        for request in requests
    ] == original_texts


def test_batch_plan_reports_correct_input_count() -> None:
    planner = EmbeddingBatchPlanner()

    plan = planner.plan(
        make_requests(37),
        batch_size=8,
    )

    assert plan.input_count == 37


def test_batch_plan_reports_correct_batch_count() -> None:
    planner = EmbeddingBatchPlanner()

    plan = planner.plan(
        make_requests(37),
        batch_size=8,
    )

    assert plan.batch_count == len(plan.batches)


def test_every_input_belongs_to_exactly_one_batch() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(101)

    plan = planner.plan(
        requests,
        batch_size=10,
    )

    flattened = [
        request
        for batch in plan.batches
        for request in batch
    ]

    assert len(flattened) == len(requests)
    assert flattened == requests


def test_batch_size_never_exceeds_configured_limit() -> None:
    planner = EmbeddingBatchPlanner()

    plan = planner.plan(
        make_requests(103),
        batch_size=11,
    )

    assert all(
        len(batch) <= 11
        for batch in plan.batches
    )


def test_batch_size_is_recorded_in_plan() -> None:
    planner = EmbeddingBatchPlanner()

    plan = planner.plan(
        make_requests(10),
        batch_size=4,
    )

    assert plan.batch_size == 4


def test_empty_text_is_not_rejected_by_planner() -> None:
    planner = EmbeddingBatchPlanner()

    plan = planner.plan(
        [EmbeddingRequest(text="")],
        batch_size=1,
    )

    assert plan.input_count == 1
    assert plan.batches[0][0].text == ""


def test_planner_preserves_complex_metadata() -> None:
    planner = EmbeddingBatchPlanner()

    request = EmbeddingRequest(
        text="complex metadata",
        metadata={
            "page_number": 12,
            "section_path": [
                "Chapter 1",
                "Introduction",
            ],
            "provenance": {
                "source_id": "source-1",
                "document_id": "document-1",
            },
            "media": {
                "start_time": 12.5,
                "end_time": 19.2,
            },
        },
    )

    plan = planner.plan(
        [request],
        batch_size=1,
    )

    result = plan.batches[0][0]

    assert result.metadata == request.metadata


def test_planner_does_not_alias_batch_container_with_input_list() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(5)

    plan = planner.plan(
        requests,
        batch_size=2,
    )

    assert plan.batches is not requests


def test_batch_containers_are_independent() -> None:
    planner = EmbeddingBatchPlanner()

    requests = make_requests(6)

    plan = planner.plan(
        requests,
        batch_size=2,
    )

    assert plan.batches[0] is not plan.batches[1]


def test_aggregator_handles_single_batch_boundary() -> None:
    planner = EmbeddingBatchPlanner()

    plan = planner.plan(
        make_requests(1),
        batch_size=1,
    )

    assert plan.input_count == 1
    assert plan.batch_count == 1


def test_aggregator_import_remains_available() -> None:
    aggregator = EmbeddingBatchAggregator()

    assert aggregator is not None