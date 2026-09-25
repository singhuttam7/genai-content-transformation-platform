from __future__ import annotations

import inspect
from uuid import uuid4

from app.models.knowledge_chunk import KnowledgeChunk
from app.models.knowledge_chunk_embedding import KnowledgeChunkEmbedding
from app.models_ai.embeddings.schemas import (
    EmbeddingBatchResult,
    EmbeddingModelInfo,
    EmbeddingVector,
)
from app.vector_store import (
    VectorBatchPersistenceRequest,
    VectorDeleteRequest,
    VectorDeleteResult,
    VectorPersistenceRequest,
    VectorPersistenceResult,
    VectorQuery,
    VectorRecord,
    VectorStorePort,
)
from app.vector_store.adapters.postgres import PostgresVectorStore
from app.vector_store.retrieval import (
    VectorRetrievalMatch,
    VectorRetrievalPort,
    VectorRetrievalRequest,
    VectorRetrievalResult,
    VectorRetrievalService,
)
from app.vector_store.retrieval.adapters.postgres import (
    PostgresVectorRetrieval,
)
from app.vector_store.service import VectorPersistenceService


VECTOR_DIMENSION = 384


# ============================================================
# Public API / export audit
# ============================================================


def test_vector_store_public_exports_are_available():
    """
    The vector_store package must expose its public persistence
    contracts through its public package boundary.
    """
    public_symbols = {
        "VectorStorePort": VectorStorePort,
        "VectorRecord": VectorRecord,
        "VectorPersistenceRequest": VectorPersistenceRequest,
        "VectorBatchPersistenceRequest": VectorBatchPersistenceRequest,
        "VectorPersistenceResult": VectorPersistenceResult,
        "VectorQuery": VectorQuery,
        "VectorDeleteRequest": VectorDeleteRequest,
        "VectorDeleteResult": VectorDeleteResult,
    }

    for name, symbol in public_symbols.items():
        assert symbol is not None, (
            f"{name} is not exported correctly."
        )


def test_vector_retrieval_public_exports_are_available():
    """
    Retrieval consumers must be able to access the complete public
    retrieval API from app.vector_store.retrieval.
    """
    public_symbols = {
        "VectorRetrievalPort": VectorRetrievalPort,
        "VectorRetrievalRequest": VectorRetrievalRequest,
        "VectorRetrievalMatch": VectorRetrievalMatch,
        "VectorRetrievalResult": VectorRetrievalResult,
        "VectorRetrievalService": VectorRetrievalService,
    }

    for name, symbol in public_symbols.items():
        assert symbol is not None, (
            f"{name} is not exported correctly."
        )


# ============================================================
# Port / adapter boundary audit
# ============================================================


def test_postgres_vector_store_implements_persistence_port():
    """
    PostgreSQL persistence must implement the provider-independent
    VectorStorePort contract.
    """
    assert issubclass(
        PostgresVectorStore,
        VectorStorePort,
    )

    required_methods = {
        "persist",
        "persist_batch",
        "get_by_chunk_id",
        "exists",
        "delete",
    }

    for method_name in required_methods:
        method = getattr(
            PostgresVectorStore,
            method_name,
            None,
        )

        assert callable(method), (
            f"PostgresVectorStore.{method_name} must be callable."
        )

        assert method_name in PostgresVectorStore.__dict__, (
            f"PostgresVectorStore must implement {method_name}."
        )


def test_postgres_retrieval_implements_retrieval_port():
    """
    PostgreSQL retrieval must implement the provider-independent
    VectorRetrievalPort contract.
    """
    assert issubclass(
        PostgresVectorRetrieval,
        VectorRetrievalPort,
    )

    assert callable(PostgresVectorRetrieval.search)
    assert "search" in PostgresVectorRetrieval.__dict__


def test_persistence_service_depends_on_vector_store_port():
    """
    VectorPersistenceService must depend on the provider-independent
    VectorStorePort abstraction.

    The test intentionally does not assume a particular constructor
    parameter name because that is an implementation detail.
    """
    signature = inspect.signature(
        VectorPersistenceService.__init__
    )

    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.name != "self"
    ]

    assert parameters, (
        "VectorPersistenceService must accept a vector-store dependency."
    )

    dependency_parameter = parameters[0]

    annotation = dependency_parameter.annotation

    assert annotation in {
        VectorStorePort,
        "VectorStorePort",
    }, (
        "VectorPersistenceService must depend on VectorStorePort."
    )


def test_retrieval_service_depends_on_retrieval_port():
    """
    VectorRetrievalService must depend on the provider-independent
    VectorRetrievalPort abstraction.
    """
    signature = inspect.signature(
        VectorRetrievalService.__init__
    )

    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.name != "self"
    ]

    assert parameters, (
        "VectorRetrievalService must accept a retrieval dependency."
    )

    dependency_parameter = parameters[0]

    annotation = dependency_parameter.annotation

    assert annotation in {
        VectorRetrievalPort,
        "VectorRetrievalPort",
    }, (
        "VectorRetrievalService must depend on VectorRetrievalPort."
    )


# ============================================================
# Embedding / vector contract compatibility
# ============================================================


def test_embedding_vector_contains_required_vector_information():
    """
    EmbeddingVector must contain the information required for the
    application service to construct a VectorRecord.
    """
    model = EmbeddingModelInfo(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=VECTOR_DIMENSION,
        normalized=True,
        metadata={
            "runtime": "sentence-transformers",
        },
    )

    embedding = EmbeddingVector(
        values=[0.0] * VECTOR_DIMENSION,
        model=model,
        input_index=0,
    )

    assert len(embedding.values) == VECTOR_DIMENSION

    assert embedding.model.provider == (
        "sentence-transformers"
    )

    assert embedding.model.model_name == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    assert embedding.model.dimension == VECTOR_DIMENSION

    assert embedding.input_index == 0


def test_vector_record_can_construct_persistence_request():
    """
    VectorPersistenceRequest intentionally accepts VectorRecord.

    The application service is responsible for converting an
    EmbeddingVector into a VectorRecord before persistence.
    """
    model = EmbeddingModelInfo(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=VECTOR_DIMENSION,
        normalized=True,
        metadata={
            "runtime": "sentence-transformers",
        },
    )

    chunk_id = uuid4()

    record = VectorRecord(
        chunk_id=chunk_id,
        values=[0.0] * VECTOR_DIMENSION,
        model=model,
        metadata={
            "input_index": 0,
        },
    )

    request = VectorPersistenceRequest(
        chunk_id=chunk_id,
        embedding=record,
    )

    assert request.chunk_id == chunk_id
    assert request.embedding is record

    assert request.embedding.chunk_id == chunk_id

    assert request.embedding.model.provider == (
        "sentence-transformers"
    )

    assert request.embedding.model.model_name == (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    assert request.embedding.model.dimension == VECTOR_DIMENSION

    assert len(request.embedding.values) == VECTOR_DIMENSION


def test_batch_embedding_contract_preserves_dimensions():
    """
    EmbeddingBatchResult must preserve the dimensionality required
    by downstream vector persistence.
    """
    model = EmbeddingModelInfo(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=VECTOR_DIMENSION,
        normalized=True,
        metadata={},
    )

    embeddings = [
        EmbeddingVector(
            values=[0.0] * VECTOR_DIMENSION,
            model=model,
            input_index=0,
        ),
        EmbeddingVector(
            values=[1.0] + [0.0] * (VECTOR_DIMENSION - 1),
            model=model,
            input_index=1,
        ),
    ]

    batch = EmbeddingBatchResult(
        embeddings=embeddings,
        model=model,
        input_count=2,
        dimension=VECTOR_DIMENSION,
    )

    assert batch.input_count == 2
    assert batch.dimension == VECTOR_DIMENSION
    assert len(batch.embeddings) == 2

    assert all(
        len(embedding.values) == VECTOR_DIMENSION
        for embedding in batch.embeddings
    )


# ============================================================
# Transaction ownership audit
# ============================================================


def test_postgres_vector_store_does_not_own_transactions():
    """
    PostgresVectorStore must not expose transaction-control methods.

    Transaction ownership belongs to the caller/application layer.
    """
    forbidden_methods = {
        "commit",
        "rollback",
        "begin",
        "begin_nested",
    }

    adapter_methods = {
        name
        for name, value in inspect.getmembers(
            PostgresVectorStore,
            predicate=inspect.isfunction,
        )
    }

    assert adapter_methods.isdisjoint(
        forbidden_methods
    )


def test_vector_persistence_service_does_not_own_transactions():
    """
    VectorPersistenceService must not become a transaction manager.
    """
    forbidden_methods = {
        "commit",
        "rollback",
        "begin",
        "begin_nested",
    }

    service_methods = {
        name
        for name, value in inspect.getmembers(
            VectorPersistenceService,
            predicate=inspect.isfunction,
        )
    }

    assert service_methods.isdisjoint(
        forbidden_methods
    )


# ============================================================
# Retrieval result contract audit
# ============================================================


def test_retrieval_result_preserves_model_identity():
    """
    Retrieval results must retain provider/model/dimension identity
    so downstream RAG components know which vector space produced
    the result.
    """
    model = EmbeddingModelInfo(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=VECTOR_DIMENSION,
        normalized=True,
        metadata={
            "runtime": "sentence-transformers",
        },
    )

    chunk_id = uuid4()

    match = VectorRetrievalMatch(
    chunk_id=chunk_id,
    text="Integration test retrieval content.",
    similarity=0.95,
    model=model,
    metadata={
        "source": "integration-test",
    },
)

    result = VectorRetrievalResult(
        matches=[match],
        query_model=model,
        count=1,
        metadata={
            "similarity_metric": "cosine",
        },
    )

    assert result.count == 1
    assert len(result.matches) == 1

    assert result.query_model.provider == model.provider
    assert result.query_model.model_name == model.model_name
    assert result.query_model.dimension == model.dimension

    returned_model = result.matches[0].model

    assert returned_model.provider == model.provider
    assert returned_model.model_name == model.model_name
    assert returned_model.dimension == model.dimension

    assert result.matches[0].chunk_id == chunk_id


# ============================================================
# ORM relationship audit
# ============================================================


def test_knowledge_chunk_embedding_relationship_is_bidirectional():
    """
    KnowledgeChunk and KnowledgeChunkEmbedding must expose the
    expected bidirectional SQLAlchemy relationship.
    """
    chunk_relationship = (
        KnowledgeChunk.__mapper__.relationships["embeddings"]
    )

    embedding_relationship = (
        KnowledgeChunkEmbedding.__mapper__.relationships["chunk"]
    )

    assert chunk_relationship.back_populates == "chunk"
    assert embedding_relationship.back_populates == "embeddings"


def test_embedding_model_contains_required_identity_columns():
    """
    The persistence model must retain enough information to isolate
    vectors by provider, model, and dimension.
    """
    table = KnowledgeChunkEmbedding.__table__

    column_names = set(table.columns.keys())

    required_columns = {
        "chunk_id",
        "provider",
        "model_name",
        "dimension",
        "normalized",
        "embedding",
        "metadata",
    }

    assert required_columns.issubset(
        column_names
    )

    assert table.c.provider.nullable is False
    assert table.c.model_name.nullable is False
    assert table.c.dimension.nullable is False
    assert table.c.embedding.nullable is False


# ============================================================
# Identity / uniqueness audit
# ============================================================


def test_embedding_identity_constraint_exists():
    """
    A single chunk/provider/model identity must be unique.

    Different models may coexist for the same chunk, but the same
    provider/model combination must not be duplicated accidentally.
    """
    constraints = list(
        KnowledgeChunkEmbedding.__table__.constraints
    )

    unique_constraints = [
        constraint
        for constraint in constraints
        if constraint.__class__.__name__
        == "UniqueConstraint"
    ]

    matching_constraints = [
        constraint
        for constraint in unique_constraints
        if {
            column.name
            for column in constraint.columns
        }
        == {
            "chunk_id",
            "provider",
            "model_name",
        }
    ]

    assert len(matching_constraints) == 1


# ============================================================
# Retrieval boundary audit
# ============================================================


def test_retrieval_port_has_search_operation():
    """
    The provider-independent retrieval boundary must expose the
    search operation used by the application service.
    """
    assert callable(
        getattr(
            VectorRetrievalPort,
            "search",
            None,
        )
    )


def test_retrieval_service_exposes_search_operation():
    """
    VectorRetrievalService is the application-level retrieval
    entry point.
    """
    assert callable(
        VectorRetrievalService.search
    )

    signature = inspect.signature(
        VectorRetrievalService.search
    )

    parameters = [
        parameter
        for parameter in signature.parameters.values()
    ]

    assert parameters[0].name == "self"
    assert parameters[1].name == "request"


# ============================================================
# Persistence boundary audit
# ============================================================


def test_persistence_service_exposes_expected_operations():
    """
    VectorPersistenceService must expose the expected application
    operations without leaking infrastructure concerns.
    """
    required_methods = {
        "persist_embedding",
        "persist_batch",
        "get_embedding",
        "exists",
        "delete",
    }

    service_methods = {
        name
        for name, value in inspect.getmembers(
            VectorPersistenceService,
            predicate=inspect.isfunction,
        )
    }

    assert required_methods.issubset(
        service_methods
    )


def test_vector_store_port_exposes_expected_operations():
    """
    VectorStorePort defines the provider-independent persistence
    boundary.
    """
    required_methods = {
        "persist",
        "persist_batch",
        "get_by_chunk_id",
        "exists",
        "delete",
    }

    port_methods = {
        name
        for name, value in inspect.getmembers(
            VectorStorePort,
            predicate=inspect.isfunction,
        )
    }

    assert required_methods.issubset(
        port_methods
    )


# ============================================================
# Model isolation contract
# ============================================================


def test_model_identity_contains_provider_model_and_dimension():
    """
    EmbeddingModelInfo must carry the three pieces of identity
    required to keep vector spaces isolated.
    """
    model = EmbeddingModelInfo(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=VECTOR_DIMENSION,
        normalized=True,
        metadata={},
    )

    assert model.provider
    assert model.model_name
    assert model.dimension == VECTOR_DIMENSION


def test_vector_persistence_model_has_pgvector_embedding_column():
    """
    KnowledgeChunkEmbedding must use the PostgreSQL vector column
    introduced for native pgvector persistence.
    """
    embedding_column = (
        KnowledgeChunkEmbedding.__table__.c.embedding
    )

    assert embedding_column is not None

    column_type = embedding_column.type

    assert getattr(
        column_type,
        "dim",
        None,
    ) == VECTOR_DIMENSION


# ============================================================
# Final architectural boundary audit
# ============================================================


def test_persistence_adapter_is_not_application_service():
    """
    PostgreSQL infrastructure and application service must remain
    separate architectural components.
    """
    assert PostgresVectorStore is not VectorPersistenceService

    assert not issubclass(
        VectorPersistenceService,
        PostgresVectorStore,
    )


def test_retrieval_adapter_is_not_application_service():
    """
    PostgreSQL retrieval infrastructure and application service must
    remain separate architectural components.
    """
    assert PostgresVectorRetrieval is not VectorRetrievalService

    assert not issubclass(
        VectorRetrievalService,
        PostgresVectorRetrieval,
    )