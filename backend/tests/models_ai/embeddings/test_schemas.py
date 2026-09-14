from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models_ai.embeddings import (
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingModelInfo,
    EmbeddingRequest,
    EmbeddingVector,
)


def make_model_info() -> EmbeddingModelInfo:
    return EmbeddingModelInfo(
        provider="local",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        normalized=True,
        metadata={
            "framework": "sentence-transformers",
        },
    )


def make_request(text: str = "hello world") -> EmbeddingRequest:
    return EmbeddingRequest(
        text=text,
        metadata={
            "source": "test",
        },
    )


def test_embedding_model_info_accepts_valid_configuration() -> None:
    model = make_model_info()

    assert model.provider == "local"
    assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert model.dimension == 384
    assert model.normalized is True
    assert model.metadata["framework"] == "sentence-transformers"


def test_embedding_model_info_defaults_metadata() -> None:
    model = EmbeddingModelInfo(
        provider="test",
        model_name="test-model",
        dimension=3,
    )

    assert model.metadata == {}
    assert model.normalized is False


def test_embedding_model_info_rejects_empty_provider() -> None:
    with pytest.raises(ValidationError):
        EmbeddingModelInfo(
            provider="",
            model_name="test-model",
            dimension=3,
        )


def test_embedding_model_info_rejects_empty_model_name() -> None:
    with pytest.raises(ValidationError):
        EmbeddingModelInfo(
            provider="test",
            model_name="",
            dimension=3,
        )


def test_embedding_model_info_rejects_invalid_dimension() -> None:
    with pytest.raises(ValidationError):
        EmbeddingModelInfo(
            provider="test",
            model_name="test-model",
            dimension=0,
        )


def test_embedding_request_accepts_text_and_metadata() -> None:
    request = make_request()

    assert request.text == "hello world"
    assert request.metadata == {"source": "test"}


def test_embedding_request_allows_empty_text_at_contract_level() -> None:
    request = EmbeddingRequest(text="")

    assert request.text == ""


def test_embedding_request_defaults_metadata() -> None:
    request = EmbeddingRequest(text="hello")

    assert request.metadata == {}


def test_embedding_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        EmbeddingRequest(
            text="hello",
            unknown_field="invalid",
        )


def test_embedding_vector_accepts_valid_values() -> None:
    model = make_model_info()

    vector = EmbeddingVector(
        values=[0.1, 0.2, 0.3],
        model=model,
        input_index=0,
    )

    assert vector.values == [0.1, 0.2, 0.3]
    assert vector.model == model
    assert vector.input_index == 0


def test_embedding_vector_rejects_empty_values() -> None:
    with pytest.raises(ValidationError):
        EmbeddingVector(
            values=[],
            model=make_model_info(),
            input_index=0,
        )


def test_embedding_vector_rejects_negative_input_index() -> None:
    with pytest.raises(ValidationError):
        EmbeddingVector(
            values=[0.1],
            model=make_model_info(),
            input_index=-1,
        )


def test_embedding_batch_request_accepts_requests() -> None:
    request = EmbeddingBatchRequest(
        requests=[
            make_request("first"),
            make_request("second"),
        ]
    )

    assert len(request.requests) == 2
    assert request.requests[0].text == "first"
    assert request.requests[1].text == "second"


def test_embedding_batch_request_rejects_empty_batch() -> None:
    with pytest.raises(ValidationError):
        EmbeddingBatchRequest(requests=[])


def test_embedding_batch_result_accepts_valid_result() -> None:
    model = make_model_info()

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
        metadata={
            "batch_size": 2,
        },
    )

    assert len(result.embeddings) == 2
    assert result.input_count == 2
    assert result.dimension == 3
    assert result.metadata["batch_size"] == 2


def test_embedding_batch_result_rejects_empty_embeddings() -> None:
    with pytest.raises(ValidationError):
        EmbeddingBatchResult(
            embeddings=[],
            model=make_model_info(),
            input_count=1,
            dimension=384,
        )


def test_embedding_batch_result_rejects_invalid_input_count() -> None:
    with pytest.raises(ValidationError):
        EmbeddingBatchResult(
            embeddings=[
                EmbeddingVector(
                    values=[0.1],
                    model=EmbeddingModelInfo(
                        provider="test",
                        model_name="test",
                        dimension=1,
                    ),
                    input_index=0,
                )
            ],
            model=EmbeddingModelInfo(
                provider="test",
                model_name="test",
                dimension=1,
            ),
            input_count=0,
            dimension=1,
        )


def test_embedding_models_are_immutable() -> None:
    model = make_model_info()

    with pytest.raises(ValidationError):
        model.dimension = 768  # type: ignore[misc]


def test_embedding_request_is_immutable() -> None:
    request = make_request()

    with pytest.raises(ValidationError):
        request.text = "changed"  # type: ignore[misc]


def test_embedding_vector_is_immutable() -> None:
    vector = EmbeddingVector(
        values=[0.1, 0.2],
        model=EmbeddingModelInfo(
            provider="test",
            model_name="test",
            dimension=2,
        ),
        input_index=0,
    )

    with pytest.raises(ValidationError):
        vector.input_index = 1  # type: ignore[misc]


def test_embedding_contracts_support_json_serialization() -> None:
    model = make_model_info()

    result = EmbeddingBatchResult(
        embeddings=[
            EmbeddingVector(
                values=[0.1, 0.2, 0.3],
                model=model,
                input_index=0,
            )
        ],
        model=model,
        input_count=1,
        dimension=3,
    )

    serialized = result.model_dump(mode="json")

    assert serialized["model"]["provider"] == "local"
    assert serialized["model"]["dimension"] == 384
    assert serialized["embeddings"][0]["input_index"] == 0


def test_embedding_contracts_preserve_metadata() -> None:
    request = EmbeddingRequest(
        text="important text",
        metadata={
            "project_id": str(uuid4()),
            "chunk_index": 7,
            "language": "en",
        },
    )

    assert request.metadata["chunk_index"] == 7
    assert request.metadata["language"] == "en"