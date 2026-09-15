from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    EmbeddingBatchError,
    EmbeddingBatchResult,
    EmbeddingDimensionError,
    EmbeddingModelInfo,
    EmbeddingRequest,
    EmbeddingVector,
)
from app.models_ai.embeddings.batching import (
    EmbeddingBatchAggregator,
    EmbeddingBatchPlanner,
)


def make_requests(count: int) -> list[EmbeddingRequest]:
    return [
        EmbeddingRequest(
            text=f"chunk {index}",
            metadata={"original_index": index},
        )
        for index in range(count)
    ]


def make_model(dimension: int = 3) -> EmbeddingModelInfo:
    return EmbeddingModelInfo(
        provider="fake",
        model_name="fake-model",
        dimension=dimension,
        normalized=True,
    )


def make_result(
    count: int,
    *,
    dimension: int = 3,
    model: EmbeddingModelInfo | None = None,
) -> EmbeddingBatchResult:
    model = model or make_model(dimension)

    embeddings = [
        EmbeddingVector(
            values=[
                float(index + value)
                for value in range(dimension)
            ],
            model=model,
            input_index=index,
        )
        for index in range(count)
    ]

    return EmbeddingBatchResult(
        embeddings=embeddings,
        model=model,
        input_count=count,
        dimension=dimension,
    )


def make_plan(
    input_count: int,
    batch_size: int,
):
    planner = EmbeddingBatchPlanner()

    return planner.plan(
        make_requests(input_count),
        batch_size=batch_size,
    )


def test_aggregator_flattens_single_batch() -> None:
    plan = make_plan(3, 10)

    results = [
        make_result(3),
    ]

    aggregator = EmbeddingBatchAggregator()

    vectors = aggregator.aggregate(
        plan=plan,
        results=results,
    )

    assert len(vectors) == 3
    assert [vector.input_index for vector in vectors] == [
        0,
        1,
        2,
    ]


def test_aggregator_flattens_multiple_batches() -> None:
    plan = make_plan(7, 3)

    results = [
        make_result(3),
        make_result(3),
        make_result(1),
    ]

    aggregator = EmbeddingBatchAggregator()

    vectors = aggregator.aggregate(
        plan=plan,
        results=results,
    )

    assert len(vectors) == 7
    assert [vector.input_index for vector in vectors] == list(range(7))


def test_aggregator_reconstructs_global_indices() -> None:
    plan = make_plan(7, 3)

    results = [
        make_result(3),
        make_result(3),
        make_result(1),
    ]

    aggregator = EmbeddingBatchAggregator()

    vectors = aggregator.aggregate(
        plan=plan,
        results=results,
    )

    assert [vector.input_index for vector in vectors] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    ]


def test_aggregator_preserves_vector_values() -> None:
    plan = make_plan(4, 2)

    results = [
        make_result(2),
        make_result(2),
    ]

    original_values = [
        list(vector.values)
        for result in results
        for vector in result.embeddings
    ]

    aggregator = EmbeddingBatchAggregator()

    vectors = aggregator.aggregate(
        plan=plan,
        results=results,
    )

    assert [
        vector.values
        for vector in vectors
    ] == original_values


def test_aggregator_preserves_model_information() -> None:
    plan = make_plan(4, 2)

    results = [
        make_result(2),
        make_result(2),
    ]

    aggregator = EmbeddingBatchAggregator()

    vectors = aggregator.aggregate(
        plan=plan,
        results=results,
    )

    assert all(
        vector.model.provider == "fake"
        for vector in vectors
    )

    assert all(
        vector.model.model_name == "fake-model"
        for vector in vectors
    )


def test_aggregator_rejects_wrong_result_count() -> None:
    plan = make_plan(5, 2)

    results = [
        make_result(2),
        make_result(2),
    ]

    aggregator = EmbeddingBatchAggregator()

    with pytest.raises(
        EmbeddingBatchError,
        match="does not match the batch plan",
    ):
        aggregator.aggregate(
            plan=plan,
            results=results,
        )


def test_aggregator_rejects_wrong_batch_input_count() -> None:
    plan = make_plan(5, 2)

    results = [
        make_result(2),
        make_result(1),
        make_result(1),
    ]

    aggregator = EmbeddingBatchAggregator()

    with pytest.raises(
        EmbeddingBatchError,
        match="unexpected input count",
    ):
        aggregator.aggregate(
            plan=plan,
            results=results,
        )


def test_aggregator_rejects_wrong_embedding_count() -> None:
    plan = make_plan(5, 2)

    result = EmbeddingBatchResult(
        embeddings=[
            EmbeddingVector(
                values=[0.1, 0.2, 0.3],
                model=make_model(),
                input_index=0,
            )
        ],
        model=make_model(),
        input_count=2,
        dimension=3,
    )

    aggregator = EmbeddingBatchAggregator()

    with pytest.raises(
        EmbeddingBatchError,
        match="unexpected number of vectors",
    ):
        aggregator.aggregate(
            plan=plan,
            results=[
                result,
                make_result(2),
                make_result(1),
            ],
        )


def test_aggregator_rejects_inconsistent_batch_dimensions() -> None:
    plan = make_plan(4, 2)

    results = [
        make_result(2, dimension=3),
        make_result(2, dimension=4),
    ]

    aggregator = EmbeddingBatchAggregator()

    with pytest.raises(
        EmbeddingDimensionError,
        match="differ across batches",
    ):
        aggregator.aggregate(
            plan=plan,
            results=results,
        )


def test_aggregator_rejects_model_dimension_mismatch() -> None:
    plan = make_plan(2, 2)

    model = make_model(4)

    result = EmbeddingBatchResult(
        embeddings=[
            EmbeddingVector(
                values=[0.1, 0.2, 0.3],
                model=model,
                input_index=0,
            ),
            EmbeddingVector(
                values=[0.4, 0.5, 0.6],
                model=model,
                input_index=1,
            ),
        ],
        model=model,
        input_count=2,
        dimension=3,
    )

    aggregator = EmbeddingBatchAggregator()

    with pytest.raises(
        EmbeddingDimensionError,
        match="does not match its model dimension",
    ):
        aggregator.aggregate(
            plan=plan,
            results=[result],
        )


def test_aggregator_rejects_vector_dimension_mismatch() -> None:
    plan = make_plan(2, 2)

    model = make_model(3)

    result = EmbeddingBatchResult(
        embeddings=[
            EmbeddingVector(
                values=[0.1, 0.2],
                model=model,
                input_index=0,
            ),
            EmbeddingVector(
                values=[0.3, 0.4],
                model=model,
                input_index=1,
            ),
        ],
        model=model,
        input_count=2,
        dimension=3,
    )

    aggregator = EmbeddingBatchAggregator()

    with pytest.raises(
        EmbeddingDimensionError,
        match="does not match the batch dimension",
    ):
        aggregator.aggregate(
            plan=plan,
            results=[result],
        )


def test_aggregator_rejects_duplicate_local_indices() -> None:
    plan = make_plan(2, 2)

    model = make_model()

    result = EmbeddingBatchResult(
        embeddings=[
            EmbeddingVector(
                values=[0.1, 0.2, 0.3],
                model=model,
                input_index=0,
            ),
            EmbeddingVector(
                values=[0.4, 0.5, 0.6],
                model=model,
                input_index=0,
            ),
        ],
        model=model,
        input_count=2,
        dimension=3,
    )

    aggregator = EmbeddingBatchAggregator()

    with pytest.raises(
        EmbeddingBatchError,
        match="Duplicate input_index",
    ):
        aggregator.aggregate(
            plan=plan,
            results=[result],
        )


def test_aggregator_rejects_out_of_range_local_index() -> None:
    plan = make_plan(2, 2)

    model = make_model()

    result = EmbeddingBatchResult(
        embeddings=[
            EmbeddingVector(
                values=[0.1, 0.2, 0.3],
                model=model,
                input_index=2,
            ),
            EmbeddingVector(
                values=[0.4, 0.5, 0.6],
                model=model,
                input_index=1,
            ),
        ],
        model=model,
        input_count=2,
        dimension=3,
    )

    aggregator = EmbeddingBatchAggregator()

    with pytest.raises(
        EmbeddingBatchError,
        match="outside batch",
    ):
        aggregator.aggregate(
            plan=plan,
            results=[result],
        )


def test_aggregator_does_not_mutate_results() -> None:
    plan = make_plan(5, 2)

    results = [
        make_result(2),
        make_result(2),
        make_result(1),
    ]

    original_indices = [
        [
            vector.input_index
            for vector in result.embeddings
        ]
        for result in results
    ]

    aggregator = EmbeddingBatchAggregator()

    aggregator.aggregate(
        plan=plan,
        results=results,
    )

    assert [
        [
            vector.input_index
            for vector in result.embeddings
        ]
        for result in results
    ] == original_indices


def test_aggregator_is_deterministic() -> None:
    plan = make_plan(8, 3)

    results = [
        make_result(3),
        make_result(3),
        make_result(2),
    ]

    aggregator = EmbeddingBatchAggregator()

    first = aggregator.aggregate(
        plan=plan,
        results=results,
    )

    second = aggregator.aggregate(
        plan=plan,
        results=results,
    )

    assert first == second


def test_aggregator_returns_global_order_even_when_local_results_are_reordered() -> None:
    plan = make_plan(5, 3)

    model = make_model()

    first_batch = EmbeddingBatchResult(
        embeddings=[
            EmbeddingVector(
                values=[0.4, 0.5, 0.6],
                model=model,
                input_index=1,
            ),
            EmbeddingVector(
                values=[0.1, 0.2, 0.3],
                model=model,
                input_index=0,
            ),
            EmbeddingVector(
                values=[0.7, 0.8, 0.9],
                model=model,
                input_index=2,
            ),
        ],
        model=model,
        input_count=3,
        dimension=3,
    )

    second_batch = make_result(2)

    aggregator = EmbeddingBatchAggregator()

    vectors = aggregator.aggregate(
        plan=plan,
        results=[
            first_batch,
            second_batch,
        ],
    )

    assert [vector.input_index for vector in vectors] == [
        0,
        1,
        2,
        3,
        4,
    ]