from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from app.models_ai.embeddings.adapters.base import BaseEmbeddingAdapter
from app.models_ai.embeddings.exceptions import (
    EmbeddingDimensionError,
    EmbeddingOperation,
)
from app.models_ai.embeddings.schemas import (
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingModelInfo,
    EmbeddingRequest,
    EmbeddingVector,
)


class SentenceTransformerAdapter(BaseEmbeddingAdapter):
    """Embedding adapter backed by sentence-transformers."""

    DEFAULT_PROVIDER = "sentence-transformers"

    # Options consumed while constructing SentenceTransformer.
    _CONSTRUCTOR_OPTION_KEYS = {
        "device",
    }

    # Supported legacy flat encode options.
    #
    # Preferred configuration:
    #
    # options={
    #     "device": "cpu",
    #     "encode": {
    #         "show_progress_bar": False,
    #         "batch_size": 8,
    #     },
    # }
    #
    # Flat options are retained for backward compatibility.
    _ENCODE_OPTION_KEYS = {
        "batch_size",
        "normalize_embeddings",
        "show_progress_bar",
        "convert_to_numpy",
        "convert_to_tensor",
        "precision",
        "truncate_dim",
        "prompt_name",
        "prompt",
    }

    def __init__(
        self,
        model_config,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(
            model_config=model_config,
        )

        self._model_factory = model_factory
        self._model: Any | None = None

    @property
    def model(self) -> Any | None:
        """
        Return the currently loaded model.

        This property intentionally does not trigger lazy loading.
        The model is loaded only when an embedding operation requires it.
        """

        return self._model

    @property
    def is_loaded(self) -> bool:
        """Return whether the SentenceTransformer model is loaded."""

        return self._model is not None

    def _get_model_factory(self) -> Callable[..., Any]:
        """
        Resolve the SentenceTransformer implementation lazily.

        The third-party dependency is imported only when the model
        is actually required.
        """

        if self._model_factory is not None:
            return self._model_factory

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise self.translate_and_classify_provider_exception(
                error=exc,
                operation=EmbeddingOperation.EMBED,
                input_count=0,
            ) from exc

        self._model_factory = SentenceTransformer

        return self._model_factory

    def _constructor_options(self) -> dict[str, Any]:
        """
        Build options passed to SentenceTransformer(...).

        Constructor-specific options are stored at the top level of
        model_config.options.

        Currently supported constructor option:

            device
        """

        options = deepcopy(
            self.model_config.options
        )

        constructor_options: dict[str, Any] = {}

        for key in self._CONSTRUCTOR_OPTION_KEYS:
            if key in options:
                constructor_options[key] = deepcopy(
                    options[key]
                )

        return constructor_options

    def _load_model(self) -> Any:
        """
        Load and cache the SentenceTransformer model.

        Loading is performed exactly when required and the resulting
        model instance is reused for subsequent operations.
        """

        if self._model is not None:
            return self._model

        factory = self._get_model_factory()

        try:
            self._model = factory(
                self.model_config.model_name,
                **self._constructor_options(),
            )
        except Exception as exc:
            raise self.translate_and_classify_provider_exception(
                error=exc,
                operation=EmbeddingOperation.EMBED,
                input_count=0,
            ) from exc

        return self._model

    def _build_model_info(
        self,
        dimension: int,
    ) -> EmbeddingModelInfo:
        """Build model metadata for an embedding result."""

        return EmbeddingModelInfo(
            provider=self.model_config.provider,
            model_name=self.model_config.model_name,
            dimension=dimension,
            normalized=self.model_config.normalized,
            metadata={
                "runtime": "sentence-transformers",
            },
        )

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        """Generate one embedding."""

        model = self._load_model()

        try:
            raw_vector = model.encode(
                request.text,
                **self._encode_options(),
            )
        except Exception as exc:
            raise self.translate_and_classify_provider_exception(
                error=exc,
                operation=EmbeddingOperation.EMBED,
                input_count=1,
            ) from exc

        vector = self._convert_vector(
            raw_vector
        )

        self._validate_dimension(
            len(vector)
        )

        return EmbeddingVector(
            values=vector,
            model=self._build_model_info(
                len(vector)
            ),
            input_index=0,
        )

    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        """Generate embeddings for a batch."""

        model = self._load_model()

        texts = [
            item.text
            for item in request.requests
        ]

        input_count = len(texts)

        try:
            raw_vectors = model.encode(
                texts,
                **self._encode_options(),
            )
        except Exception as exc:
            raise self.translate_and_classify_provider_exception(
                error=exc,
                operation=EmbeddingOperation.EMBED_BATCH,
                input_count=input_count,
            ) from exc

        vectors = self._convert_batch(
            raw_vectors
        )

        if len(vectors) != input_count:
            raise EmbeddingDimensionError(
                (
                    "SentenceTransformer returned an unexpected "
                    "number of embeddings: "
                    f"expected {input_count}, "
                    f"got {len(vectors)}."
                )
            )

        dimensions = {
            len(vector)
            for vector in vectors
        }

        if len(dimensions) != 1:
            raise EmbeddingDimensionError(
                (
                    "SentenceTransformer returned embeddings "
                    "with inconsistent dimensions."
                )
            )

        dimension = len(vectors[0])

        self._validate_dimension(
            dimension
        )

        model_info = self._build_model_info(
            dimension
        )

        embeddings = [
            EmbeddingVector(
                values=vector,
                model=model_info,
                input_index=index,
            )
            for index, vector in enumerate(vectors)
        ]

        return EmbeddingBatchResult(
            embeddings=embeddings,
            model=model_info,
            input_count=input_count,
            dimension=dimension,
        )

    def _encode_options(self) -> dict[str, Any]:
        """
        Build options passed to model.encode().

        Preferred configuration:

            options={
                "device": "cpu",
                "encode": {
                    "show_progress_bar": False,
                    "batch_size": 8,
                },
            }

        Legacy flat configuration remains supported:

            options={
                "show_progress_bar": False,
                "batch_size": 8,
            }

        Precedence:

            nested encode option
                >
            flat encode option
                >
            generic configuration default

        Constructor-only options such as "device" are never passed
        to model.encode().
        """

        options = deepcopy(
            self.model_config.options
        )

        encode_options: dict[str, Any] = {}

        # ---------------------------------------------------------
        # 1. Legacy flat encode options
        # ---------------------------------------------------------

        for key in self._ENCODE_OPTION_KEYS:
            if key in options:
                encode_options[key] = deepcopy(
                    options[key]
                )

        # ---------------------------------------------------------
        # 2. Preferred nested encode options
        # ---------------------------------------------------------

        nested_encode = options.get(
            "encode",
            {},
        )

        if not isinstance(
            nested_encode,
            dict,
        ):
            raise EmbeddingDimensionError(
                (
                    "SentenceTransformer 'encode' options "
                    "must be a dictionary."
                )
            )

        # Nested options override flat options.
        encode_options.update(
            deepcopy(nested_encode)
        )

        # ---------------------------------------------------------
        # 3. Generic configuration defaults
        # ---------------------------------------------------------

        if self.model_config.normalized:
            encode_options.setdefault(
                "normalize_embeddings",
                True,
            )

        encode_options.setdefault(
            "batch_size",
            self.model_config.batch_size,
        )

        # ---------------------------------------------------------
        # 4. Constructor-only options never reach encode().
        # ---------------------------------------------------------

        for key in self._CONSTRUCTOR_OPTION_KEYS:
            encode_options.pop(
                key,
                None,
            )

        return encode_options

    def _convert_vector(
        self,
        raw_vector: Any,
    ) -> list[float]:
        """Convert one provider vector into canonical format."""

        if hasattr(
            raw_vector,
            "tolist",
        ):
            raw_vector = raw_vector.tolist()

        if not isinstance(
            raw_vector,
            (list, tuple),
        ):
            raise EmbeddingDimensionError(
                (
                    "SentenceTransformer returned a "
                    "non-sequence embedding."
                )
            )

        if not raw_vector:
            raise EmbeddingDimensionError(
                (
                    "SentenceTransformer returned "
                    "an empty embedding."
                )
            )

        try:
            return [
                float(value)
                for value in raw_vector
            ]
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise EmbeddingDimensionError(
                (
                    "SentenceTransformer returned a "
                    "non-numeric embedding."
                )
            ) from exc

    def _convert_batch(
        self,
        raw_vectors: Any,
    ) -> list[list[float]]:
        """Convert provider batch output into canonical vectors."""

        if hasattr(
            raw_vectors,
            "tolist",
        ):
            raw_vectors = raw_vectors.tolist()

        if not isinstance(
            raw_vectors,
            (list, tuple),
        ):
            raise EmbeddingDimensionError(
                (
                    "SentenceTransformer returned a "
                    "non-sequence batch."
                )
            )

        return [
            self._convert_vector(vector)
            for vector in raw_vectors
        ]

    def _validate_dimension(
        self,
        dimension: int,
    ) -> None:
        """Validate the returned embedding dimension."""

        expected_dimension = (
            self.model_config.expected_dimension
        )

        if expected_dimension is None:
            return

        if dimension != expected_dimension:
            raise EmbeddingDimensionError(
                (
                    "SentenceTransformer embedding dimension "
                    "mismatch: "
                    f"expected {expected_dimension}, "
                    f"got {dimension}."
                )
            )