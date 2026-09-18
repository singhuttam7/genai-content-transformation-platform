from __future__ import annotations

from uuid import uuid4

from pgvector.sqlalchemy import VECTOR

from app.models.knowledge_chunk_embedding import KnowledgeChunkEmbedding


def test_knowledge_chunk_embedding_table_name() -> None:
    assert (
        KnowledgeChunkEmbedding.__tablename__
        == "knowledge_chunk_embeddings"
    )


def test_embedding_column_uses_pgvector() -> None:
    column = KnowledgeChunkEmbedding.__table__.c.embedding

    assert isinstance(column.type, VECTOR)


def test_embedding_dimension_is_384() -> None:
    column = KnowledgeChunkEmbedding.__table__.c.embedding

    assert column.type.dim == 384


def test_chunk_id_is_not_nullable() -> None:
    column = KnowledgeChunkEmbedding.__table__.c.chunk_id

    assert column.nullable is False


def test_provider_is_not_nullable() -> None:
    column = KnowledgeChunkEmbedding.__table__.c.provider

    assert column.nullable is False


def test_model_name_is_not_nullable() -> None:
    column = KnowledgeChunkEmbedding.__table__.c.model_name

    assert column.nullable is False


def test_dimension_is_not_nullable() -> None:
    column = KnowledgeChunkEmbedding.__table__.c.dimension

    assert column.nullable is False


def test_normalized_is_not_nullable() -> None:
    column = KnowledgeChunkEmbedding.__table__.c.normalized

    assert column.nullable is False


def test_embedding_is_not_nullable() -> None:
    column = KnowledgeChunkEmbedding.__table__.c.embedding

    assert column.nullable is False


def test_metadata_is_not_nullable() -> None:
    column = KnowledgeChunkEmbedding.__table__.c.metadata

    assert column.nullable is False


def test_chunk_foreign_key_cascades() -> None:
    column = KnowledgeChunkEmbedding.__table__.c.chunk_id

    foreign_key = next(iter(column.foreign_keys))

    assert foreign_key.target_fullname == "knowledge_chunks.id"
    assert foreign_key.ondelete == "CASCADE"


def test_model_has_expected_unique_constraint() -> None:
    constraints = KnowledgeChunkEmbedding.__table__.constraints

    unique_constraints = [
        constraint
        for constraint in constraints
        if constraint.__class__.__name__ == "UniqueConstraint"
    ]

    assert len(unique_constraints) == 1

    constraint = unique_constraints[0]

    assert [
        column.name
        for column in constraint.columns
    ] == [
        "chunk_id",
        "provider",
        "model_name",
    ]


def test_primary_key_exists() -> None:
    primary_key = KnowledgeChunkEmbedding.__table__.primary_key

    assert len(primary_key.columns) == 1
    assert primary_key.columns[0].name == "id"


def test_timestamp_columns_exist() -> None:
    table = KnowledgeChunkEmbedding.__table__

    assert "created_at" in table.columns
    assert "updated_at" in table.columns