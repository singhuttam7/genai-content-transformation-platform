from __future__ import annotations

from app.models_ai.embeddings.batching.schemas import EmbeddingBatchPlan
from app.models_ai.embeddings.exceptions import (
    EmbeddingBatchError,
    EmbeddingErrorContext,
    EmbeddingOperation,
)
from app.models_ai.embeddings.port import EmbeddingPort
from app.models_ai.embeddings.schemas import (
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingModelConfig,
)


class EmbeddingBatchExecutor:
    """
    Execute a deterministic embedding batch plan through an
    EmbeddingPort.

    Failure semantics:

    - batches execute sequentially;
    - execution stops at the first failed batch;
    - successful intermediate results remain internal only;
    - partial results are never returned;
    - existing EmbeddingBatchError exceptions are re-raised;
    - unexpected exceptions are wrapped in EmbeddingBatchError;
    - the original unexpected exception is preserved as the cause;
    - no retry is performed;
    - provider-independent error context is attached when
      model configuration is available.
    """

    def __init__(
        self,
        embedding_port: EmbeddingPort,
        model_config: EmbeddingModelConfig | None = None,
    ) -> None:
        self.embedding_port = embedding_port
        self.model_config = model_config

    def _build_error_context(
        self,
        *,
        batch_index: int,
        input_count: int,
    ) -> EmbeddingErrorContext:
        provider: str | None = None
        model: str | None = None
        metadata: dict[str, object] = {}

        if self.model_config is not None:
            provider = self.model_config.provider
            model = self.model_config.model_name

            if self.model_config.options:
                metadata["model_options"] = dict(
                    self.model_config.options
                )

        return EmbeddingErrorContext(
            operation=EmbeddingOperation.EMBED_BATCH,
            provider=provider,
            model=model,
            batch_index=batch_index,
            input_count=input_count,
            metadata=metadata,
        )

    async def execute(
        self,
        plan: EmbeddingBatchPlan,
    ) -> list[EmbeddingBatchResult]:
        results: list[EmbeddingBatchResult] = []

        for batch_index, batch in enumerate(plan.batches):
            request = EmbeddingBatchRequest(
                requests=list(batch),
            )

            try:
                result = await self.embedding_port.embed_batch(
                    request
                )

            except EmbeddingBatchError:
                raise

            except Exception as exc:
                context = self._build_error_context(
                    batch_index=batch_index,
                    input_count=len(batch),
                )

                raise EmbeddingBatchError(
                    f"Embedding batch {batch_index} failed.",
                    batch_index=batch_index,
                    context=context,
                ) from exc

            results.append(result)

        return results