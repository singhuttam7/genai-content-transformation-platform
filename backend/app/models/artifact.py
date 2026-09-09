from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from app.models.execution import Execution
    from app.models.transformation import Transformation


class Artifact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Generated communication artifact produced by an execution."""

    __tablename__ = "artifacts"

    transformation_id: Mapped[UUID] = mapped_column(
        ForeignKey("transformations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    execution_id: Mapped[UUID] = mapped_column(
        ForeignKey("executions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    artifact_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    title: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
    )

    content: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    storage_uri: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    content_hash: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    artifact_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="GENERATING",
        index=True,
    )

    transformation: Mapped["Transformation"] = relationship(
        back_populates="artifacts",
    )

    execution: Mapped["Execution"] = relationship(
        back_populates="artifacts",
    )