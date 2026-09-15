from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy

from app.models_ai.embeddings.classification import (
    CompositeEmbeddingErrorClassifier,
)
from app.models_ai.embeddings.exceptions import (
    EmbeddingErrorCategory,
    EmbeddingErrorClassification,
    EmbeddingErrorContext,
    EmbeddingOperation,
    EmbeddingProviderError,
)
from app.models_ai.embeddings.port import EmbeddingPort
from app.models_ai.embeddings.schemas import (
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingModelConfig,
    EmbeddingRequest,
    EmbeddingVector,
)


class BaseEmbeddingAdapter(
    EmbeddingPort,
    ABC,
):
    """Base boundary for provider-specific embedding adapters.

    Provider SDK exceptions are translated into the
    provider-independent embedding error model at this boundary.

    Provider and caller metadata are defensively copied so that
    error-context state cannot accidentally alias mutable objects
    owned by callers or model configuration.
    """

    def __init__(
        self,
        model_config: EmbeddingModelConfig,
        *,
        error_classifier: CompositeEmbeddingErrorClassifier | None = None,
    ) -> None:
        self.model_config = model_config
        self.error_classifier = (
            error_classifier
            or CompositeEmbeddingErrorClassifier()
        )

    @abstractmethod
    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        """Generate one embedding."""

        raise NotImplementedError

    @abstractmethod
    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        """Generate embeddings for a batch."""

        raise NotImplementedError

    def build_error_context(
        self,
        *,
        operation: EmbeddingOperation,
        input_count: int,
        batch_index: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> EmbeddingErrorContext:
        """Build provider-independent error context.

        All mutable metadata is defensively deep-copied before being
        stored in the error context.

        This prevents later mutation of caller-owned metadata or model
        configuration from changing an already-created error.
        """

        context_metadata: dict[str, object] = {}

        if self.model_config.options:
            context_metadata["model_options"] = deepcopy(
                self.model_config.options
            )

        if metadata:
            context_metadata.update(
                deepcopy(metadata)
            )

        return EmbeddingErrorContext(
            operation=operation,
            provider=self.model_config.provider,
            model=self.model_config.model_name,
            batch_index=batch_index,
            input_count=input_count,
            metadata=context_metadata,
        )

    def build_provider_error(
        self,
        *,
        message: str,
        category: EmbeddingErrorCategory,
        operation: EmbeddingOperation,
        input_count: int,
        batch_index: int | None = None,
        metadata: dict[str, object] | None = None,
        classification: EmbeddingErrorClassification | None = None,
    ) -> EmbeddingProviderError:
        """Create a provider-independent provider error.

        Metadata is isolated from caller-owned mutable objects.
        Classification remains a first-class runtime attribute while
        its serialized representation may also be stored in context
        metadata.
        """

        context = self.build_error_context(
            operation=operation,
            input_count=input_count,
            batch_index=batch_index,
            metadata=metadata,
        )

        return EmbeddingProviderError(
            message,
            category=category,
            context=context,
            classification=classification,
        )

    def translate_provider_exception(
        self,
        *,
        error: Exception,
        operation: EmbeddingOperation,
        input_count: int,
        category: EmbeddingErrorCategory,
        batch_index: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> EmbeddingProviderError:
        """Translate a provider exception into the common error model.

        The original provider exception is not modified.

        The caller should preserve the original exception through
        Python exception chaining:

            translated = self.translate_provider_exception(...)
            raise translated from error
        """

        message = str(error)

        if not message:
            message = "Embedding provider operation failed."

        return self.build_provider_error(
            message=message,
            category=category,
            operation=operation,
            input_count=input_count,
            batch_index=batch_index,
            metadata=metadata,
        )

    def classify_provider_exception(
        self,
        error: Exception,
    ) -> EmbeddingErrorClassification:
        """Classify a provider exception using the configured classifier."""

        return self.error_classifier.classify(
            error
        )

    def translate_and_classify_provider_exception(
        self,
        *,
        error: Exception,
        operation: EmbeddingOperation,
        input_count: int,
        batch_index: int | None = None,
        metadata: dict[str, object] | None = None,
    ) -> EmbeddingProviderError:
        """Classify and translate a provider exception.

        Classification precedence is:

        1. structured provider evidence;
        2. heuristic evidence;
        3. UNKNOWN fallback.

        The resulting error contains:

        - provider-independent category;
        - complete classification;
        - provider/model identity;
        - operation;
        - input count;
        - batch index when applicable;
        - serialized classification metadata.

        The original provider exception remains untouched.
        """

        classification = self.classify_provider_exception(
            error
        )

        classification_metadata = {
            "classification": classification.model_dump(
                mode="json"
            )
        }

        if metadata:
            classification_metadata.update(
                deepcopy(metadata)
            )

        message = str(error)

        if not message:
            message = "Embedding provider operation failed."

        return self.build_provider_error(
            message=message,
            category=classification.category,
            operation=operation,
            input_count=input_count,
            batch_index=batch_index,
            metadata=classification_metadata,
            classification=classification,
        )