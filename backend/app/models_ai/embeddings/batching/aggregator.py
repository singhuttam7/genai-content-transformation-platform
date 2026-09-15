from __future__ import annotations

from app.models_ai.embeddings.batching.schemas import EmbeddingBatchPlan
from app.models_ai.embeddings.exceptions import (
    EmbeddingBatchError,
    EmbeddingDimensionError,
)
from app.models_ai.embeddings.schemas import (
    EmbeddingBatchResult,
    EmbeddingVector,
)


class EmbeddingBatchAggregator:
    """
    Aggregate embedding results from multiple provider batches.

    The aggregator validates dimensions and reconstructs global input
    indices from the original batch plan.
    """

    def aggregate(
        self,
        *,
        plan: EmbeddingBatchPlan,
        results: list[EmbeddingBatchResult],
    ) -> list[EmbeddingVector]:
        """
        Aggregate batch results into one globally ordered vector list.

        The provider's input_index is interpreted as the position inside
        its individual batch. The original batch plan is used to map
        that local position to the global input index.

        Raises
        ------
        EmbeddingBatchError
            If batch/result counts, input counts, or indices are invalid.

        EmbeddingDimensionError
            If embedding dimensions are inconsistent.
        """

        if len(results) != plan.batch_count:
            raise EmbeddingBatchError(
                "Number of embedding results does not match the batch plan."
            )

        aggregated: list[EmbeddingVector] = []
        expected_dimension: int | None = None

        for batch_index, (batch, result) in enumerate(
            zip(plan.batches, results, strict=True)
        ):
            if result.input_count != len(batch):
                raise EmbeddingBatchError(
                    f"Embedding result for batch {batch_index} contains "
                    "an unexpected input count."
                )

            if result.dimension != result.model.dimension:
                raise EmbeddingDimensionError(
                    f"Batch {batch_index} dimension does not match "
                    "its model dimension."
                )

            if expected_dimension is None:
                expected_dimension = result.dimension
            elif result.dimension != expected_dimension:
                raise EmbeddingDimensionError(
                    "Embedding dimensions differ across batches."
                )

            if len(result.embeddings) != len(batch):
                raise EmbeddingBatchError(
                    f"Embedding result for batch {batch_index} contains "
                    "an unexpected number of vectors."
                )

            seen_local_indices: set[int] = set()

            for vector in result.embeddings:
                local_index = vector.input_index

                if local_index in seen_local_indices:
                    raise EmbeddingBatchError(
                        f"Duplicate input_index {local_index} "
                        f"in batch {batch_index}."
                    )

                seen_local_indices.add(local_index)

                if local_index >= len(batch):
                    raise EmbeddingBatchError(
                        f"input_index {local_index} is outside batch "
                        f"{batch_index}."
                    )

                if len(vector.values) != result.dimension:
                    raise EmbeddingDimensionError(
                        f"Vector dimension in batch {batch_index} "
                        "does not match the batch dimension."
                    )

                global_index = (
                    sum(len(previous_batch) for previous_batch in plan.batches[:batch_index])
                    + local_index
                )

                aggregated.append(
                    vector.model_copy(
                        update={
                            "input_index": global_index,
                        }
                    )
                )

        if len(aggregated) != plan.input_count:
            raise EmbeddingBatchError(
                "Aggregated embedding count does not match input count."
            )

        aggregated.sort(key=lambda vector: vector.input_index)

        expected_indices = list(range(plan.input_count))
        actual_indices = [
            vector.input_index
            for vector in aggregated
        ]

        if actual_indices != expected_indices:
            raise EmbeddingBatchError(
                "Aggregated embedding indices are not contiguous "
                "from zero."
            )

        return aggregated