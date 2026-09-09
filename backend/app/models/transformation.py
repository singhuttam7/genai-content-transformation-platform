from typing import TYPE_CHECKING
from uuid import UUID

from app.models.artifact import Artifact
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
    from app.models.project import Project
    from app.models.source import Source


class Transformation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Represents a user's content transformation request."""

    __tablename__ = "transformations"

    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    source_id: Mapped[UUID] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    objective: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    audience: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    tone: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    language: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="English",
    )

    detail_level: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    style: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    requested_outputs: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
    )

    configuration: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="DRAFT",
        index=True,
    )

    project: Mapped["Project"] = relationship(
        back_populates="transformations",
    )

    source: Mapped["Source"] = relationship(
        back_populates="transformations",
    )

    executions: Mapped[list["Execution"]] = relationship(
        back_populates="transformation",
        cascade="all, delete-orphan",
    )

    artifacts: Mapped[list["Artifact"]] = relationship(
    back_populates="transformation",
    cascade="all, delete-orphan",
)