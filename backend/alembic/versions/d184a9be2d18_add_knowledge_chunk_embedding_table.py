"""add knowledge chunk embedding table

Revision ID: d184a9be2d18
Revises: f69a4264fc58
Create Date: 2026-09-17 00:36:14.174287

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "d184a9be2d18"
down_revision: Union[str, Sequence[str], None] = "f69a4264fc58"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    op.create_table(
        "knowledge_chunk_embeddings",
        sa.Column(
            "chunk_id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "provider",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "model_name",
            sa.String(length=200),
            nullable=False,
        ),
        sa.Column(
            "dimension",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "normalized",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "embedding",
            VECTOR(dim=384),
            nullable=False,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "id",
            sa.Uuid(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["chunk_id"],
            ["knowledge_chunks.id"],
            name=op.f(
                "fk_knowledge_chunk_embeddings_chunk_id_knowledge_chunks"
            ),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_knowledge_chunk_embeddings"),
        ),
        sa.UniqueConstraint(
            "chunk_id",
            "provider",
            "model_name",
            name="uq_knowledge_chunk_embeddings_chunk_provider_model",
        ),
    )

    op.create_index(
        op.f("ix_knowledge_chunk_embeddings_chunk_id"),
        "knowledge_chunk_embeddings",
        ["chunk_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_index(
        op.f("ix_knowledge_chunk_embeddings_chunk_id"),
        table_name="knowledge_chunk_embeddings",
    )

    op.drop_table("knowledge_chunk_embeddings")