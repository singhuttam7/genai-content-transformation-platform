from __future__ import annotations

from app.models_ai.embeddings.exceptions import EmbeddingInputError
from app.models_ai.embeddings.schemas import EmbeddingRequest
from app.models_ai.embeddings.batching.schemas import EmbeddingBatchPlan


class EmbeddingBatchPlanner:
    """
    Deterministically divide embedding requests into batches.

    The planner performs no provider calls and does not mutate the
    supplied requests.
    """

    def plan(
        self,
        requests: list[EmbeddingRequest],
        *,
        batch_size: int,
    ) -> EmbeddingBatchPlan:
        """
        Create a deterministic batch plan.

        Parameters
        ----------
        requests:
            Embedding requests to divide into batches.

        batch_size:
            Maximum number of requests per batch.

        Returns
        -------
        EmbeddingBatchPlan
            Planned batches preserving original request order.

        Raises
        ------
        EmbeddingInputError
            If requests are empty or batch_size is invalid.
        """

        if not requests:
            raise EmbeddingInputError(
                "At least one embedding request is required."
            )

        if batch_size < 1:
            raise EmbeddingInputError(
                "batch_size must be greater than or equal to 1."
            )

        batches = [
            requests[start : start + batch_size]
            for start in range(0, len(requests), batch_size)
        ]

        return EmbeddingBatchPlan(
            batches=batches,
            batch_size=batch_size,
            input_count=len(requests),
            batch_count=len(batches),
        )