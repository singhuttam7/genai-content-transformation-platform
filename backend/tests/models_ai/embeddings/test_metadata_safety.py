from __future__ import annotations

from app.models_ai.embeddings.adapters.base import (
    BaseEmbeddingAdapter,
)
from app.models_ai.embeddings.exceptions import (
    EmbeddingErrorCategory,
    EmbeddingOperation,
)
from app.models_ai.embeddings.schemas import (
    EmbeddingModelConfig,
)


class _FakeEmbeddingAdapter(BaseEmbeddingAdapter):
    """Minimal adapter used to test the base error boundary."""

    async def embed(self, request):
        raise NotImplementedError

    async def embed_batch(self, request):
        raise NotImplementedError


def create_adapter() -> _FakeEmbeddingAdapter:
    return _FakeEmbeddingAdapter(
        EmbeddingModelConfig(
            provider="test-provider",
            model_name="test-model",
            options={
                "temperature": 0,
                "nested": {
                    "enabled": True,
                    "values": [1, 2, 3],
                },
            },
        )
    )


def test_error_metadata_does_not_mutate_input_metadata() -> None:
    adapter = create_adapter()

    metadata = {
        "request_id": "req-001",
        "attempt": 1,
    }

    original_metadata = dict(metadata)

    error = adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.NETWORK,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        metadata=metadata,
    )

    assert metadata == original_metadata
    assert error.context is not None


def test_error_metadata_contains_supplied_values() -> None:
    adapter = create_adapter()

    metadata = {
        "request_id": "req-002",
        "region": "ap-south-1",
    }

    error = adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.NETWORK,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        metadata=metadata,
    )

    assert error.context is not None
    assert error.context.metadata["request_id"] == "req-002"
    assert error.context.metadata["region"] == "ap-south-1"


def test_nested_metadata_is_copied_into_error_context() -> None:
    adapter = create_adapter()

    nested = {
        "provider": {
            "region": "ap-south-1",
            "features": ["embeddings", "batching"],
        }
    }

    error = adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.NETWORK,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        metadata=nested,
    )

    assert error.context is not None

    nested["provider"]["region"] = "changed"
    nested["provider"]["features"].append("unexpected")

    assert (
        error.context.metadata["provider"]["region"]
        == "ap-south-1"
    )

    assert error.context.metadata["provider"]["features"] == [
        "embeddings",
        "batching",
    ]


def test_nested_metadata_does_not_alias_input_dictionary() -> None:
    adapter = create_adapter()

    provider_metadata = {
        "provider": {
            "limits": {
                "requests_per_minute": 100,
            }
        }
    }

    error = adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.RATE_LIMIT,
        operation=EmbeddingOperation.EMBED_BATCH,
        input_count=4,
        metadata=provider_metadata,
    )

    assert error.context is not None

    provider_metadata["provider"]["limits"][
        "requests_per_minute"
    ] = 1

    assert (
        error.context.metadata["provider"]["limits"][
            "requests_per_minute"
        ]
        == 100
    )


def test_model_options_are_not_aliased_into_error_context() -> None:
    model_options = {
        "temperature": 0,
        "nested": {
            "enabled": True,
            "values": [1, 2, 3],
        },
    }

    adapter = _FakeEmbeddingAdapter(
        EmbeddingModelConfig(
            provider="test-provider",
            model_name="test-model",
            options=model_options,
        )
    )

    error = adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.NETWORK,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert error.context is not None

    model_options["temperature"] = 100
    model_options["nested"]["enabled"] = False
    model_options["nested"]["values"].append(4)

    stored_options = error.context.metadata["model_options"]

    assert stored_options["temperature"] == 0
    assert stored_options["nested"]["enabled"] is True
    assert stored_options["nested"]["values"] == [
        1,
        2,
        3,
    ]


def test_error_context_metadata_does_not_mutate_model_options() -> None:
    model_options = {
        "temperature": 0,
        "nested": {
            "enabled": True,
        },
    }

    adapter = _FakeEmbeddingAdapter(
        EmbeddingModelConfig(
            provider="test-provider",
            model_name="test-model",
            options=model_options,
        )
    )

    adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.TIMEOUT,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert model_options == {
        "temperature": 0,
        "nested": {
            "enabled": True,
        },
    }


def test_classification_metadata_is_serialized_in_context() -> None:
    adapter = create_adapter()

    error = adapter.translate_and_classify_provider_exception(
        error=TimeoutError("request timeout"),
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert error.classification is not None
    assert error.context is not None

    serialized = error.context.metadata[
        "classification"
    ]

    assert serialized["category"] == (
        EmbeddingErrorCategory.TIMEOUT.value
    )

    assert serialized["retryable"] is True
    assert serialized["confidence"] == 0.90


def test_classification_metadata_is_independent_from_runtime_object() -> None:
    adapter = create_adapter()

    error = adapter.translate_and_classify_provider_exception(
        error=TimeoutError("request timeout"),
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert error.classification is not None
    assert error.context is not None

    serialized = error.context.metadata[
        "classification"
    ]

    assert serialized is not error.classification

    assert serialized["category"] == (
        error.classification.category.value
    )


def test_provider_and_model_identity_are_preserved() -> None:
    adapter = create_adapter()

    error = adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
        operation=EmbeddingOperation.EMBED_BATCH,
        input_count=8,
        batch_index=2,
    )

    assert error.context is not None

    assert error.context.provider == "test-provider"
    assert error.context.model == "test-model"


def test_batch_context_is_preserved() -> None:
    adapter = create_adapter()

    error = adapter.build_provider_error(
        message="batch failed",
        category=EmbeddingErrorCategory.RATE_LIMIT,
        operation=EmbeddingOperation.EMBED_BATCH,
        input_count=16,
        batch_index=3,
    )

    assert error.context is not None

    assert error.context.operation == (
        EmbeddingOperation.EMBED_BATCH
    )

    assert error.context.batch_index == 3
    assert error.context.input_count == 16

    assert error.batch_index == 3


def test_unicode_metadata_is_preserved() -> None:
    adapter = create_adapter()

    metadata = {
        "language": "हिन्दी",
        "region": "भारत",
        "message": "请求超时",
    }

    error = adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.TIMEOUT,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        metadata=metadata,
    )

    assert error.context is not None

    assert error.context.metadata["language"] == "हिन्दी"
    assert error.context.metadata["region"] == "भारत"
    assert error.context.metadata["message"] == "请求超时"


def test_metadata_is_json_serializable() -> None:
    adapter = create_adapter()

    error = adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.NETWORK,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        metadata={
            "request_id": "req-100",
            "attempt": 2,
            "features": ["batch", "normalized"],
            "nested": {
                "enabled": True,
                "limit": 100,
            },
        },
    )

    assert error.context is not None

    dumped = error.context.model_dump(
        mode="json"
    )

    assert dumped["provider"] == "test-provider"
    assert dumped["model"] == "test-model"
    assert dumped["metadata"]["request_id"] == "req-100"
    assert dumped["metadata"]["attempt"] == 2


def test_classification_context_is_json_serializable() -> None:
    adapter = create_adapter()

    error = adapter.translate_and_classify_provider_exception(
        error=ConnectionError("connection refused"),
        operation=EmbeddingOperation.EMBED_BATCH,
        input_count=8,
        batch_index=1,
    )

    assert error.context is not None

    dumped = error.context.model_dump(
        mode="json"
    )

    classification = dumped["metadata"][
        "classification"
    ]

    assert classification["category"] == (
        EmbeddingErrorCategory.NETWORK.value
    )

    assert classification["retryable"] is True


def test_error_context_is_not_shared_between_errors() -> None:
    adapter = create_adapter()

    first = adapter.build_provider_error(
        message="first",
        category=EmbeddingErrorCategory.NETWORK,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        metadata={
            "request_id": "first",
        },
    )

    second = adapter.build_provider_error(
        message="second",
        category=EmbeddingErrorCategory.TIMEOUT,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        metadata={
            "request_id": "second",
        },
    )

    assert first.context is not None
    assert second.context is not None

    assert first.context is not second.context
    assert first.context.metadata is not second.context.metadata

    assert (
        first.context.metadata["request_id"]
        == "first"
    )

    assert (
        second.context.metadata["request_id"]
        == "second"
    )


def test_model_options_are_not_shared_between_errors() -> None:
    adapter = create_adapter()

    first = adapter.build_provider_error(
        message="first",
        category=EmbeddingErrorCategory.NETWORK,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    second = adapter.build_provider_error(
        message="second",
        category=EmbeddingErrorCategory.TIMEOUT,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert first.context is not None
    assert second.context is not None

    first_options = first.context.metadata[
        "model_options"
    ]

    second_options = second.context.metadata[
        "model_options"
    ]

    assert first_options is not second_options
    assert first_options["nested"] is not second_options[
        "nested"
    ]

    first_options["nested"]["enabled"] = False

    assert second_options["nested"]["enabled"] is True


def test_supplied_metadata_is_not_modified_by_model_options() -> None:
    adapter = create_adapter()

    metadata = {
        "request_id": "req-200",
        "nested": {
            "value": 42,
        },
    }

    original = {
        "request_id": "req-200",
        "nested": {
            "value": 42,
        },
    }

    adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.NETWORK,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        metadata=metadata,
    )

    assert metadata == original


def test_error_metadata_remains_consistent_after_input_mutation() -> None:
    adapter = create_adapter()

    metadata = {
        "request": {
            "id": "req-300",
            "tags": ["a", "b"],
        }
    }

    error = adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.NETWORK,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        metadata=metadata,
    )

    assert error.context is not None

    metadata["request"]["id"] = "changed"
    metadata["request"]["tags"].clear()
    metadata["request"]["new_field"] = "unexpected"

    stored = error.context.metadata["request"]

    assert stored["id"] == "req-300"
    assert stored["tags"] == ["a", "b"]
    assert "new_field" not in stored


def test_error_metadata_is_independent_when_mutated_after_creation() -> None:
    adapter = create_adapter()

    metadata = {
        "request_id": "req-400",
    }

    error = adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.NETWORK,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        metadata=metadata,
    )

    assert error.context is not None

    error.context.metadata["request_id"] = "changed"

    assert metadata["request_id"] == "req-400"


def test_error_classification_and_category_remain_consistent() -> None:
    adapter = create_adapter()

    error = adapter.translate_and_classify_provider_exception(
        error=TimeoutError("timeout"),
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert error.classification is not None

    assert error.category == (
        error.classification.category
    )

    assert error.context is not None

    serialized_category = error.context.metadata[
        "classification"
    ]["category"]

    assert serialized_category == (
        error.classification.category.value
    )


def test_model_options_are_present_only_when_configured() -> None:
    adapter = _FakeEmbeddingAdapter(
        EmbeddingModelConfig(
            provider="test-provider",
            model_name="test-model",
            options={},
        )
    )

    error = adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.UNKNOWN,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert error.context is not None
    assert "model_options" not in error.context.metadata


def test_metadata_values_with_falsey_values_are_preserved() -> None:
    adapter = create_adapter()

    metadata = {
        "empty_string": "",
        "zero": 0,
        "false": False,
        "empty_list": [],
        "empty_dict": {},
    }

    error = adapter.build_provider_error(
        message="provider failed",
        category=EmbeddingErrorCategory.INPUT,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        metadata=metadata,
    )

    assert error.context is not None

    stored = error.context.metadata

    assert stored["empty_string"] == ""
    assert stored["zero"] == 0
    assert stored["false"] is False
    assert stored["empty_list"] == []
    assert stored["empty_dict"] == {}