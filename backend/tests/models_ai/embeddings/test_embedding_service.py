from __future__ import annotations

import pytest

from app.models_ai.embeddings.exceptions import (
    EmbeddingBatchError,
    EmbeddingError,
)
from app.models_ai.embeddings.port import EmbeddingPort
from app.models_ai.embeddings.schemas import (
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingModelInfo,
    EmbeddingRequest,
    EmbeddingVector,
    EmbeddingModelConfig,
)
from app.models_ai.embeddings.service import EmbeddingService


MODEL = EmbeddingModelInfo(
    provider="fake",
    model_name="fake-model",
    dimension=3,
    normalized=True,
)


class FakeEmbeddingPort(EmbeddingPort):
    """Deterministic provider used for service contract tests."""

    def __init__(self) -> None:
        self.single_requests: list[EmbeddingRequest] = []
        self.batch_requests: list[EmbeddingBatchRequest] = []
        self.error: Exception | None = None

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        self.single_requests.append(request)

        if self.error is not None:
            raise self.error

        return EmbeddingVector(
            values=[1.0, 2.0, 3.0],
            model=MODEL,
            input_index=0,
        )

    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        self.batch_requests.append(request)

        if self.error is not None:
            raise self.error

        embeddings = [
            EmbeddingVector(
                values=[
                    float(index),
                    float(index + 1),
                    float(index + 2),
                ],
                model=MODEL,
                input_index=index,
            )
            for index in range(len(request.requests))
        ]

        return EmbeddingBatchResult(
            embeddings=embeddings,
            model=MODEL,
            input_count=len(embeddings),
            dimension=3,
        )


class NotAnEmbeddingPort:
    pass


def create_service() -> tuple[EmbeddingService, FakeEmbeddingPort]:
    port = FakeEmbeddingPort()
    return EmbeddingService(port), port


def test_service_accepts_embedding_port() -> None:
    service, port = create_service()

    assert service.port is port


def test_service_rejects_invalid_port() -> None:
    with pytest.raises(TypeError, match="EmbeddingPort"):
        EmbeddingService(NotAnEmbeddingPort())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_embed_text_builds_embedding_request() -> None:
    service, port = create_service()

    result = await service.embed_text(
        "hello world",
        metadata={"source": "test"},
    )

    assert result.values == [1.0, 2.0, 3.0]
    assert len(port.single_requests) == 1
    assert port.single_requests[0].text == "hello world"
    assert port.single_requests[0].metadata == {
        "source": "test",
    }


@pytest.mark.asyncio
async def test_embed_text_without_metadata_uses_empty_metadata() -> None:
    service, port = create_service()

    await service.embed_text("hello")

    assert port.single_requests[0].metadata == {}


@pytest.mark.asyncio
async def test_embed_text_does_not_alias_metadata() -> None:
    service, port = create_service()

    metadata = {
        "nested": {
            "value": 1,
        }
    }

    await service.embed_text(
        "hello",
        metadata=metadata,
    )

    metadata["nested"]["value"] = 999

    assert port.single_requests[0].metadata["nested"]["value"] == 1


@pytest.mark.asyncio
async def test_embed_text_rejects_non_string() -> None:
    service, _ = create_service()

    with pytest.raises(TypeError, match="string"):
        await service.embed_text(123)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_embed_text_rejects_blank_text() -> None:
    service, _ = create_service()

    with pytest.raises(ValueError, match="blank"):
        await service.embed_text("   ")


@pytest.mark.asyncio
async def test_embed_text_preserves_provider_error() -> None:
    service, port = create_service()

    error = EmbeddingError("provider failed")
    port.error = error

    with pytest.raises(EmbeddingError) as exc_info:
        await service.embed_text("hello")

    assert exc_info.value is error


@pytest.mark.asyncio
async def test_embed_texts_builds_batch_request() -> None:
    service, port = create_service()

    result = await service.embed_texts(
        ["first", "second", "third"],
    )

    assert result.input_count == 3
    assert result.dimension == 3

    assert len(port.batch_requests) == 1

    requests = port.batch_requests[0].requests

    assert [request.text for request in requests] == [
        "first",
        "second",
        "third",
    ]


@pytest.mark.asyncio
async def test_embed_texts_preserves_input_order() -> None:
    service, _ = create_service()

    result = await service.embed_texts(
        ["first", "second", "third"],
    )

    assert [
        embedding.input_index
        for embedding in result.embeddings
    ] == [0, 1, 2]


@pytest.mark.asyncio
async def test_embed_texts_preserves_duplicate_inputs() -> None:
    service, port = create_service()

    await service.embed_texts(
        ["same", "same", "different"],
    )

    requests = port.batch_requests[0].requests

    assert [request.text for request in requests] == [
        "same",
        "same",
        "different",
    ]


@pytest.mark.asyncio
async def test_embed_texts_passes_positional_metadata() -> None:
    service, port = create_service()

    await service.embed_texts(
        ["first", "second"],
        metadata=[
            {"id": 1},
            {"id": 2},
        ],
    )

    requests = port.batch_requests[0].requests

    assert requests[0].metadata == {"id": 1}
    assert requests[1].metadata == {"id": 2}


@pytest.mark.asyncio
async def test_embed_texts_without_metadata_uses_empty_metadata() -> None:
    service, port = create_service()

    await service.embed_texts(
        ["first", "second"],
    )

    assert [
        request.metadata
        for request in port.batch_requests[0].requests
    ] == [{}, {}]


@pytest.mark.asyncio
async def test_embed_texts_rejects_empty_batch() -> None:
    service, _ = create_service()

    with pytest.raises(ValueError, match="empty"):
        await service.embed_texts([])


@pytest.mark.asyncio
async def test_embed_texts_rejects_non_list() -> None:
    service, _ = create_service()

    with pytest.raises(TypeError, match="list"):
        await service.embed_texts(  # type: ignore[arg-type]
            ("first", "second")
        )


@pytest.mark.asyncio
async def test_embed_texts_rejects_non_string_item() -> None:
    service, _ = create_service()

    with pytest.raises(TypeError, match="texts\\[1\\]"):
        await service.embed_texts(
            ["first", 123],  # type: ignore[list-item]
        )


@pytest.mark.asyncio
async def test_embed_texts_rejects_blank_item() -> None:
    service, _ = create_service()

    with pytest.raises(ValueError, match="texts\\[1\\]"):
        await service.embed_texts(
            ["first", "   "],
        )


@pytest.mark.asyncio
async def test_embed_texts_rejects_metadata_length_mismatch() -> None:
    service, _ = create_service()

    with pytest.raises(
        ValueError,
        match="metadata length",
    ):
        await service.embed_texts(
            ["first", "second"],
            metadata=[{"id": 1}],
        )


@pytest.mark.asyncio
async def test_embed_texts_rejects_invalid_metadata_type() -> None:
    service, _ = create_service()

    with pytest.raises(TypeError, match="metadata"):
        await service.embed_texts(
            ["first"],
            metadata={"id": 1},  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_embed_texts_rejects_invalid_metadata_item() -> None:
    service, _ = create_service()

    with pytest.raises(TypeError, match="metadata\\[1\\]"):
        await service.embed_texts(
            ["first", "second"],
            metadata=[
                {"id": 1},
                "invalid",  # type: ignore[list-item]
            ],
        )


@pytest.mark.asyncio
async def test_embed_texts_wraps_unexpected_provider_error() -> None:
    service, port = create_service()

    error = EmbeddingError("batch provider failed")
    port.error = error

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await service.embed_texts(
            ["first", "second"],
        )

    assert exc_info.value.__cause__ is error

@pytest.mark.asyncio
async def test_service_uses_injected_port_for_single_embedding() -> None:
    service, port = create_service()

    await service.embed_text("hello")

    assert len(port.single_requests) == 1
    assert port.single_requests[0].text == "hello"


@pytest.mark.asyncio
async def test_service_uses_injected_port_for_batch_embedding() -> None:
    service, port = create_service()

    await service.embed_texts(
        ["first", "second", "third"],
    )

    assert len(port.batch_requests) == 1
    assert [
        request.text
        for request in port.batch_requests[0].requests
    ] == [
        "first",
        "second",
        "third",
    ]


@pytest.mark.asyncio
async def test_service_does_not_construct_provider() -> None:
    port = FakeEmbeddingPort()

    service = EmbeddingService(port)

    assert service.port is port
    assert len(port.single_requests) == 0
    assert len(port.batch_requests) == 0


@pytest.mark.asyncio
async def test_service_can_use_different_port_instance() -> None:
    first_port = FakeEmbeddingPort()
    second_port = FakeEmbeddingPort()

    first_service = EmbeddingService(first_port)
    second_service = EmbeddingService(second_port)

    await first_service.embed_text("first")
    await second_service.embed_text("second")

    assert [request.text for request in first_port.single_requests] == [
        "first"
    ]

    assert [request.text for request in second_port.single_requests] == [
        "second"
    ]


@pytest.mark.asyncio
async def test_service_has_no_provider_specific_configuration_requirement() -> None:
    port = FakeEmbeddingPort()
    service = EmbeddingService(port)

    await service.embed_text("provider independent")

    request = port.single_requests[0]

    assert request.text == "provider independent"
    assert request.metadata == {}

@pytest.mark.asyncio
async def test_embed_text_returns_provider_vector_unchanged() -> None:
    service, _ = create_service()

    result = await service.embed_text("hello")

    assert result.values == [1.0, 2.0, 3.0]
    assert result.model == MODEL
    assert result.input_index == 0


@pytest.mark.asyncio
async def test_embed_text_preserves_original_text() -> None:
    service, port = create_service()

    text = "  Hello, world!  "

    await service.embed_text(text)

    assert port.single_requests[0].text == text


@pytest.mark.asyncio
async def test_embed_text_preserves_metadata_values() -> None:
    service, port = create_service()

    metadata = {
        "source": "document.pdf",
        "page": 4,
        "tags": ["important", "technical"],
    }

    await service.embed_text(
        "hello",
        metadata=metadata,
    )

    assert port.single_requests[0].metadata == metadata


@pytest.mark.asyncio
async def test_embed_text_does_not_modify_caller_metadata() -> None:
    service, _ = create_service()

    metadata = {
        "source": "document.pdf",
        "nested": {
            "page": 4,
        },
    }

    original = {
        "source": "document.pdf",
        "nested": {
            "page": 4,
        },
    }

    await service.embed_text(
        "hello",
        metadata=metadata,
    )

    assert metadata == original


@pytest.mark.asyncio
async def test_embed_text_calls_provider_exactly_once() -> None:
    service, port = create_service()

    await service.embed_text("hello")

    assert len(port.single_requests) == 1


@pytest.mark.asyncio
async def test_embed_text_does_not_call_batch_provider() -> None:
    service, port = create_service()

    await service.embed_text("hello")

    assert len(port.batch_requests) == 0


@pytest.mark.asyncio
async def test_embed_text_accepts_empty_metadata_dictionary() -> None:
    service, port = create_service()

    await service.embed_text(
        "hello",
        metadata={},
    )

    assert port.single_requests[0].metadata == {}

@pytest.mark.asyncio
async def test_embed_texts_returns_provider_batch_result() -> None:
    service, _ = create_service()

    result = await service.embed_texts(
        ["first", "second", "third"],
    )

    assert result.model == MODEL
    assert result.input_count == 3
    assert result.dimension == 3
    assert len(result.embeddings) == 3


@pytest.mark.asyncio
async def test_embed_texts_calls_batch_provider_exactly_once() -> None:
    service, port = create_service()

    await service.embed_texts(
        ["first", "second", "third"],
    )

    assert len(port.batch_requests) == 1


@pytest.mark.asyncio
async def test_embed_texts_does_not_call_single_provider() -> None:
    service, port = create_service()

    await service.embed_texts(
        ["first", "second"],
    )

    assert len(port.single_requests) == 0


@pytest.mark.asyncio
async def test_embed_texts_preserves_result_order() -> None:
    service, _ = create_service()

    result = await service.embed_texts(
        ["alpha", "beta", "gamma", "delta"],
    )

    assert [
        embedding.input_index
        for embedding in result.embeddings
    ] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_embed_texts_preserves_duplicate_inputs() -> None:
    service, port = create_service()

    texts = [
        "duplicate",
        "duplicate",
        "unique",
        "duplicate",
    ]

    await service.embed_texts(texts)

    assert [
        request.text
        for request in port.batch_requests[0].requests
    ] == texts


@pytest.mark.asyncio
async def test_embed_texts_preserves_positional_metadata_order() -> None:
    service, port = create_service()

    texts = ["first", "second", "third"]
    metadata = [
        {"position": 0},
        {"position": 1},
        {"position": 2},
    ]

    await service.embed_texts(
        texts,
        metadata=metadata,
    )

    requests = port.batch_requests[0].requests

    assert [
        request.metadata["position"]
        for request in requests
    ] == [0, 1, 2]


@pytest.mark.asyncio
async def test_embed_texts_does_not_alias_nested_metadata() -> None:
    service, port = create_service()

    metadata = [
        {
            "nested": {
                "value": 10,
            }
        },
        {
            "nested": {
                "value": 20,
            }
        },
    ]

    await service.embed_texts(
        ["first", "second"],
        metadata=metadata,
    )

    metadata[0]["nested"]["value"] = 999
    metadata[1]["nested"]["value"] = 888

    requests = port.batch_requests[0].requests

    assert requests[0].metadata["nested"]["value"] == 10
    assert requests[1].metadata["nested"]["value"] == 20


@pytest.mark.asyncio
async def test_embed_texts_does_not_modify_caller_texts() -> None:
    service, _ = create_service()

    texts = ["first", "second"]
    original = texts.copy()

    await service.embed_texts(texts)

    assert texts == original


@pytest.mark.asyncio
async def test_embed_texts_does_not_modify_caller_metadata() -> None:
    service, _ = create_service()

    metadata = [
        {"source": "one"},
        {"source": "two"},
    ]

    original = [
        {"source": "one"},
        {"source": "two"},
    ]

    await service.embed_texts(
        ["first", "second"],
        metadata=metadata,
    )

    assert metadata == original


@pytest.mark.asyncio
async def test_embed_texts_preserves_embedding_batch_error() -> None:
    service, port = create_service()

    error = EmbeddingBatchError(
        "batch embedding failed",
        batch_index=0,
    )
    port.error = error

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await service.embed_texts(
            ["first", "second"],
        )

    assert exc_info.value is error

@pytest.mark.asyncio
async def test_invalid_single_input_does_not_call_provider() -> None:
    service, port = create_service()

    with pytest.raises(TypeError):
        await service.embed_text(123)  # type: ignore[arg-type]

    assert port.single_requests == []
    assert port.batch_requests == []


@pytest.mark.asyncio
async def test_blank_single_input_does_not_call_provider() -> None:
    service, port = create_service()

    with pytest.raises(ValueError):
        await service.embed_text("   ")

    assert port.single_requests == []
    assert port.batch_requests == []


@pytest.mark.asyncio
async def test_invalid_batch_input_does_not_call_provider() -> None:
    service, port = create_service()

    with pytest.raises(ValueError):
        await service.embed_texts([])

    assert port.single_requests == []
    assert port.batch_requests == []


@pytest.mark.asyncio
async def test_blank_batch_item_does_not_call_provider() -> None:
    service, port = create_service()

    with pytest.raises(ValueError):
        await service.embed_texts(
            ["valid", "   ", "also valid"],
        )

    assert port.single_requests == []
    assert port.batch_requests == []


@pytest.mark.asyncio
async def test_metadata_length_error_does_not_call_provider() -> None:
    service, port = create_service()

    with pytest.raises(ValueError, match="metadata length"):
        await service.embed_texts(
            ["first", "second"],
            metadata=[{"id": 1}],
        )

    assert port.single_requests == []
    assert port.batch_requests == []


@pytest.mark.asyncio
async def test_invalid_metadata_type_does_not_call_provider() -> None:
    service, port = create_service()

    with pytest.raises(TypeError):
        await service.embed_texts(
            ["first"],
            metadata={"id": 1},  # type: ignore[arg-type]
        )

    assert port.single_requests == []
    assert port.batch_requests == []


@pytest.mark.asyncio
async def test_provider_error_identity_is_preserved() -> None:
    service, port = create_service()

    error = EmbeddingError("structured provider failure")
    port.error = error

    with pytest.raises(EmbeddingError) as exc_info:
        await service.embed_text("valid input")

    assert exc_info.value is error


@pytest.mark.asyncio
async def test_batch_provider_error_cause_is_preserved() -> None:
    service, port = create_service()

    error = EmbeddingError(
        "structured batch provider failure"
    )
    port.error = error

    with pytest.raises(EmbeddingBatchError) as exc_info:
        await service.embed_texts(
            ["first", "second"],
        )

    assert exc_info.value.__cause__ is error

@pytest.mark.asyncio
async def test_embed_texts_uses_batch_planner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _ = create_service()

    calls: list[int] = []

    original_plan = service.planner.plan

    def spy_plan(
        requests: list[EmbeddingRequest],
        *,
        batch_size: int,
    ):
        calls.append(batch_size)
        return original_plan(
            requests,
            batch_size=batch_size,
        )

    monkeypatch.setattr(
        service.planner,
        "plan",
        spy_plan,
    )

    await service.embed_texts(
        ["one", "two", "three"],
    )

    assert calls == [32]


@pytest.mark.asyncio
async def test_embed_texts_uses_configured_batch_size() -> None:
    port = FakeEmbeddingPort()

    config = EmbeddingModelConfig(
        provider="fake",
        model_name="fake-model",
        batch_size=2,
    )

    service = EmbeddingService(
        port,
        model_config=config,
    )

    result = await service.embed_texts(
        ["one", "two", "three", "four", "five"],
    )

    assert result.input_count == 5
    assert result.dimension == 3

    assert len(port.batch_requests) == 3

    assert [
        len(request.requests)
        for request in port.batch_requests
    ] == [2, 2, 1]


@pytest.mark.asyncio
async def test_embed_texts_aggregates_multiple_batches() -> None:
    port = FakeEmbeddingPort()

    config = EmbeddingModelConfig(
        provider="fake",
        model_name="fake-model",
        batch_size=2,
    )

    service = EmbeddingService(
        port,
        model_config=config,
    )

    result = await service.embed_texts(
        ["one", "two", "three", "four", "five"],
    )

    assert [
        vector.input_index
        for vector in result.embeddings
    ] == [0, 1, 2, 3, 4]


@pytest.mark.asyncio
async def test_embed_texts_preserves_metadata_across_batches() -> None:
    port = FakeEmbeddingPort()

    config = EmbeddingModelConfig(
        provider="fake",
        model_name="fake-model",
        batch_size=2,
    )

    service = EmbeddingService(
        port,
        model_config=config,
    )

    metadata = [
        {"id": 0},
        {"id": 1},
        {"id": 2},
        {"id": 3},
        {"id": 4},
    ]

    await service.embed_texts(
        ["one", "two", "three", "four", "five"],
        metadata=metadata,
    )

    actual_metadata = [
        request.metadata
        for batch in port.batch_requests
        for request in batch.requests
    ]

    assert actual_metadata == metadata


@pytest.mark.asyncio
async def test_embed_texts_preserves_duplicate_inputs_across_batches() -> None:
    port = FakeEmbeddingPort()

    config = EmbeddingModelConfig(
        provider="fake",
        model_name="fake-model",
        batch_size=2,
    )

    service = EmbeddingService(
        port,
        model_config=config,
    )

    texts = [
        "same",
        "same",
        "different",
        "same",
        "same",
    ]

    result = await service.embed_texts(texts)

    actual_texts = [
        request.text
        for batch in port.batch_requests
        for request in batch.requests
    ]

    assert actual_texts == texts

    assert [
        vector.input_index
        for vector in result.embeddings
    ] == [0, 1, 2, 3, 4]