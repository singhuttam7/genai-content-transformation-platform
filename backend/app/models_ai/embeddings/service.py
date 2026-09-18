from __future__ import annotations

from copy import deepcopy

from app.models_ai.embeddings.batching import (
    EmbeddingBatchAggregator,
    EmbeddingBatchExecutor,
    EmbeddingBatchPlanner,
)
from app.models_ai.embeddings.port import EmbeddingPort
from app.models_ai.embeddings.schemas import (
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingModelConfig,
    EmbeddingRequest,
    EmbeddingVector,
)


class EmbeddingService:
    """Application-facing service for provider-independent embeddings.

    The service depends on the provider-independent EmbeddingPort and
    coordinates the existing embedding batching infrastructure.

    Responsibilities:
    - validate the provider dependency;
    - validate application-facing inputs;
    - translate application inputs into embedding contracts;
    - execute single embedding operations;
    - plan, execute, and aggregate batch operations;
    - preserve provider-independent results and errors.

    Non-responsibilities:
    - provider/model construction;
    - provider SDK handling;
    - persistence;
    - vector-store operations;
    - chunking;
    - retrieval;
    - retry execution.
    """

    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        port: EmbeddingPort,
        model_config: EmbeddingModelConfig | None = None,
        *,
        planner: EmbeddingBatchPlanner | None = None,
        executor: EmbeddingBatchExecutor | None = None,
        aggregator: EmbeddingBatchAggregator | None = None,
    ) -> None:
        if not isinstance(port, EmbeddingPort):
            raise TypeError(
                "port must implement EmbeddingPort."
            )

        if model_config is not None and not isinstance(
            model_config,
            EmbeddingModelConfig,
        ):
            raise TypeError(
                "model_config must be an EmbeddingModelConfig."
            )

        self._port = port
        self._model_config = model_config

        self._planner = (
            planner
            if planner is not None
            else EmbeddingBatchPlanner()
        )

        self._executor = (
            executor
            if executor is not None
            else EmbeddingBatchExecutor(
                embedding_port=port,
                model_config=model_config,
            )
        )

        self._aggregator = (
            aggregator
            if aggregator is not None
            else EmbeddingBatchAggregator()
        )

        self._validate_batch_components()

    def _validate_batch_components(self) -> None:
        """Validate batching infrastructure dependencies."""

        if not isinstance(
            self._planner,
            EmbeddingBatchPlanner,
        ):
            raise TypeError(
                "planner must be an EmbeddingBatchPlanner."
            )

        if not isinstance(
            self._executor,
            EmbeddingBatchExecutor,
        ):
            raise TypeError(
                "executor must be an EmbeddingBatchExecutor."
            )

        if not isinstance(
            self._aggregator,
            EmbeddingBatchAggregator,
        ):
            raise TypeError(
                "aggregator must be an EmbeddingBatchAggregator."
            )

    @property
    def port(self) -> EmbeddingPort:
        """Return the configured embedding port."""

        return self._port

    @property
    def model_config(self) -> EmbeddingModelConfig | None:
        """Return the configured embedding model configuration."""

        return self._model_config

    @property
    def planner(self) -> EmbeddingBatchPlanner:
        """Return the configured batch planner."""

        return self._planner

    @property
    def executor(self) -> EmbeddingBatchExecutor:
        """Return the configured batch executor."""

        return self._executor

    @property
    def aggregator(self) -> EmbeddingBatchAggregator:
        """Return the configured batch aggregator."""

        return self._aggregator

    async def embed_text(
        self,
        text: str,
        *,
        metadata: dict[str, object] | None = None,
    ) -> EmbeddingVector:
        """Generate an embedding for one text input.

        Args:
            text: Text to embed.
            metadata: Optional caller-provided metadata.

        Returns:
            Provider-independent EmbeddingVector.

        Raises:
            TypeError: If text is not a string.
            ValueError: If text is blank.
            EmbeddingError: If the underlying provider fails.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        if not text.strip():
            raise ValueError("text cannot be blank.")

        request = EmbeddingRequest(
            text=text,
            metadata=(
                {}
                if metadata is None
                else deepcopy(metadata)
            ),
        )

        return await self._port.embed(request)

    async def embed_texts(
        self,
        texts: list[str],
        *,
        metadata: list[dict[str, object]] | None = None,
    ) -> EmbeddingBatchResult:
        """Generate embeddings for multiple text inputs.

        The existing batching infrastructure is used for every batch
        operation. The configured model batch size determines how the
        requests are partitioned.

        Args:
            texts: Non-empty list of texts to embed.
            metadata: Optional metadata corresponding positionally
                to each text.

        Returns:
            Provider-independent EmbeddingBatchResult.

        Raises:
            TypeError: If texts or metadata have invalid types.
            ValueError: If the input is invalid.
            EmbeddingError: If the underlying embedding operation fails.
            EmbeddingBatchError: If batch execution or aggregation fails.
        """

        self._validate_texts(texts)
        self._validate_metadata(texts, metadata)

        requests = [
            EmbeddingRequest(
                text=text,
                metadata=(
                    {}
                    if metadata is None
                    else deepcopy(metadata[index])
                ),
            )
            for index, text in enumerate(texts)
        ]

        batch_size = self._resolve_batch_size()

        plan = self._planner.plan(
            requests,
            batch_size=batch_size,
        )

        results = await self._executor.execute(plan)

        embeddings = self._aggregator.aggregate(
            plan=plan,
            results=results,
        )

        if not embeddings:
            raise ValueError(
                "Embedding aggregation returned no embeddings."
            )

        model = embeddings[0].model
        dimension = len(embeddings[0].values)

        return EmbeddingBatchResult(
            embeddings=embeddings,
            model=model,
            input_count=len(embeddings),
            dimension=dimension,
        )

    @staticmethod
    def _validate_texts(texts: list[str]) -> None:
        """Validate the application-level text collection."""

        if not isinstance(texts, list):
            raise TypeError("texts must be a list.")

        if not texts:
            raise ValueError("texts cannot be empty.")

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"texts[{index}] must be a string."
                )

            if not text.strip():
                raise ValueError(
                    f"texts[{index}] cannot be blank."
                )

    @staticmethod
    def _validate_metadata(
        texts: list[str],
        metadata: list[dict[str, object]] | None,
    ) -> None:
        """Validate positional metadata for a text collection."""

        if metadata is None:
            return

        if not isinstance(metadata, list):
            raise TypeError("metadata must be a list.")

        if len(metadata) != len(texts):
            raise ValueError(
                "metadata length must match texts length."
            )

        for index, item in enumerate(metadata):
            if not isinstance(item, dict):
                raise TypeError(
                    f"metadata[{index}] must be a dictionary."
                )

    def _resolve_batch_size(self) -> int:
        """Resolve the batch size used by the batching infrastructure."""

        if self._model_config is None:
            return self.DEFAULT_BATCH_SIZE

        return self._model_config.batch_size