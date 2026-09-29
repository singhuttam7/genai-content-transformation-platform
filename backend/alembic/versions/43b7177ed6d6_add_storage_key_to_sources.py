"""add storage key to sources

Revision ID: 43b7177ed6d6
Revises: d184a9be2d18
Create Date: 2026-09-29
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "43b7177ed6d6"
down_revision: Union[str, Sequence[str], None] = "d184a9be2d18"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add persisted storage key to sources."""
    op.add_column(
        "sources",
        sa.Column(
            "storage_key",
            sa.String(length=1024),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Remove persisted storage key from sources."""
    op.drop_column(
        "sources",
        "storage_key",
    )