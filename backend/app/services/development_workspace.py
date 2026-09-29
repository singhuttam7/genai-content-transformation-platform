from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project import Project
from app.models.user import User


DEVELOPMENT_USER_EMAIL = "operator@genai-platform.local"
DEVELOPMENT_USER_NAME = "Development Operator"
DEVELOPMENT_PROJECT_NAME = "Default Workspace"
DEVELOPMENT_PROJECT_DESCRIPTION = (
    "Development workspace for the GenAI Content Transformation Platform."
)


async def get_or_create_development_workspace(
    session: AsyncSession,
) -> tuple[User, Project]:
    result = await session.execute(
        select(User).where(
            User.email == DEVELOPMENT_USER_EMAIL,
        ),
    )

    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            email=DEVELOPMENT_USER_EMAIL,
            name=DEVELOPMENT_USER_NAME,
            role="operator",
            is_active=True,
        )
        session.add(user)
        await session.flush()

    result = await session.execute(
        select(Project).where(
            Project.owner_id == user.id,
            Project.name == DEVELOPMENT_PROJECT_NAME,
        ),
    )

    project = result.scalar_one_or_none()

    if project is None:
        project = Project(
            owner_id=user.id,
            name=DEVELOPMENT_PROJECT_NAME,
            description=DEVELOPMENT_PROJECT_DESCRIPTION,
            settings={},
        )
        session.add(project)
        await session.flush()

    await session.commit()

    return user, project