from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from app.models.artifact import Artifact
    from app.models.transformation import Transformation
    from app.models.workflow import Workflow


class Execution(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Represents one actual execution of a transformation workflow."""

    __tablename__ = "executions"

    transformation_id: Mapped[UUID] = mapped_column(
        ForeignKey("transformations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    workflow_id: Mapped[UUID] = mapped_column(
        ForeignKey("workflows.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    workflow_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="QUEUED",
        index=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    execution_context: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    metrics: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    transformation: Mapped["Transformation"] = relationship(
        back_populates="executions",
    )

    workflow: Mapped["Workflow"] = relationship(
        back_populates="executions",
    )

    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="execution",
        cascade="all, delete-orphan",
    )