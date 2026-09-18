from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models_ai.embeddings.schemas import EmbeddingModelInfo
from app.vector_store.retrieval.schemas import (
    VectorRetrievalMatch,
    VectorRetrievalRequest,
    VectorRetrievalResult,
)


def build_model_info(
    *,
    provider: str = "sentence-transformers",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    dimension: int = 384,
    normalized: bool = True,
) -> EmbeddingModelInfo:
    return EmbeddingModelInfo(
        provider=provider,
        model_name=model_name,
        dimension=dimension,
        normalized=normalized,
    )


class TestVectorRetrievalRequest:
    def test_valid_request(self):
        request = VectorRetrievalRequest(
            query_vector=[0.1, 0.2, 0.3],
            model=build_model_info(),
        )

        assert request.query_vector == [0.1, 0.2, 0.3]
        assert request.top_k == 5
        assert request.similarity_threshold is None
        assert request.project_id is None
        assert request.metadata_filter == {}

    def test_custom_options(self):
        project_id = uuid4()

        request = VectorRetrievalRequest(
            query_vector=[0.1, 0.2],
            model=build_model_info(),
            top_k=10,
            similarity_threshold=0.75,
            project_id=project_id,
            metadata_filter={"language": "en"},
        )

        assert request.top_k == 10
        assert request.similarity_threshold == 0.75
        assert request.project_id == project_id
        assert request.metadata_filter == {"language": "en"}

    def test_empty_query_vector_rejected(self):
        with pytest.raises(ValidationError):
            VectorRetrievalRequest(
                query_vector=[],
                model=build_model_info(),
            )

    def test_non_numeric_query_vector_rejected(self):
        with pytest.raises(ValidationError):
            VectorRetrievalRequest(
                query_vector=[0.1, "invalid"],
                model=build_model_info(),
            )

    def test_top_k_must_be_positive(self):
        with pytest.raises(ValidationError):
            VectorRetrievalRequest(
                query_vector=[0.1],
                model=build_model_info(),
                top_k=0,
            )

    @pytest.mark.parametrize(
        "threshold",
        [-1.01, 1.01],
    )
    def test_similarity_threshold_bounds(self, threshold):
        with pytest.raises(ValidationError):
            VectorRetrievalRequest(
                query_vector=[0.1],
                model=build_model_info(),
                similarity_threshold=threshold,
            )

    def test_request_is_immutable(self):
        request = VectorRetrievalRequest(
            query_vector=[0.1],
            model=build_model_info(),
        )

        with pytest.raises(ValidationError):
            request.top_k = 10


class TestVectorRetrievalMatch:
    def test_valid_match(self):
        chunk_id = uuid4()

        match = VectorRetrievalMatch(
            chunk_id=chunk_id,
            similarity=0.91,
            model=build_model_info(),
            metadata={"chunk_index": 2},
        )

        assert match.chunk_id == chunk_id
        assert match.similarity == 0.91
        assert match.metadata == {"chunk_index": 2}

    def test_default_metadata(self):
        match = VectorRetrievalMatch(
            chunk_id=uuid4(),
            similarity=0.5,
            model=build_model_info(),
        )

        assert match.metadata == {}

    def test_match_is_immutable(self):
        match = VectorRetrievalMatch(
            chunk_id=uuid4(),
            similarity=0.5,
            model=build_model_info(),
        )

        with pytest.raises(ValidationError):
            match.similarity = 0.8


class TestVectorRetrievalResult:
    def test_valid_result(self):
        match = VectorRetrievalMatch(
            chunk_id=uuid4(),
            similarity=0.92,
            model=build_model_info(),
        )

        result = VectorRetrievalResult(
            matches=[match],
            query_model=build_model_info(),
            count=1,
        )

        assert result.matches == [match]
        assert result.count == 1
        assert result.metadata == {}

    def test_empty_result_is_valid(self):
        result = VectorRetrievalResult(
            matches=[],
            query_model=build_model_info(),
            count=0,
        )

        assert result.matches == []
        assert result.count == 0

    def test_negative_count_rejected(self):
        with pytest.raises(ValidationError):
            VectorRetrievalResult(
                matches=[],
                query_model=build_model_info(),
                count=-1,
            )

    def test_result_is_immutable(self):
        result = VectorRetrievalResult(
            matches=[],
            query_model=build_model_info(),
            count=0,
        )

        with pytest.raises(ValidationError):
            result.count = 1