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
    from app.models.source import Source
    from app.models.transformation import Transformation
    from app.models.user import User
    from app.models.workflow import Workflow


class Project(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Workspace containing related content transformations."""

    __tablename__ = "projects"

    owner_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    settings: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
    )

    owner: Mapped["User"] = relationship(
        back_populates="projects",
    )

    sources: Mapped[list["Source"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )

    transformations: Mapped[list["Transformation"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )

    workflows: Mapped[list["Workflow"]] = relationship(
        back_populates="project",
        cascade="all, delete-orphan",
    )