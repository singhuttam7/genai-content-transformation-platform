from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.workflow import Workflow


DEVELOPMENT_WORKFLOW_NAME = "Default Transformation Workflow"
DEVELOPMENT_WORKFLOW_DESCRIPTION = (
    "Development workflow for the GenAI Content Transformation Platform."
)

TRANSFORMATION_AGENT_NAMES = {
    "advisory": "AdvisoryTransformationAgent",
    "executive_summary": "ExecutiveSummaryTransformationAgent",
    "social_media": "SocialMediaTransformationAgent",
    "infographic": "InfographicTransformationAgent",
    "presentation": "PresentationTransformationAgent",
    "video": "VideoTransformationAgent",
}


async def get_or_create_development_workflow(
    session: AsyncSession,
    project_id: UUID,
    transformation_type: str = "executive_summary",
) -> Workflow:
    agent_name = TRANSFORMATION_AGENT_NAMES.get(
        transformation_type,
    )

    if agent_name is None:
        raise ValueError(
            f"Unsupported transformation type: {transformation_type}"
        )

    workflow_name = (
        f"{DEVELOPMENT_WORKFLOW_NAME} - "
        f"{transformation_type.replace('_', ' ').title()}"
    )

    result = await session.execute(
        select(Workflow).where(
            Workflow.project_id == project_id,
            Workflow.name == workflow_name,
            Workflow.is_active.is_(True),
        ),
    )

    workflow = result.scalar_one_or_none()

    if workflow is None:
        workflow = Workflow(
            project_id=project_id,
            name=workflow_name,
            description=DEVELOPMENT_WORKFLOW_DESCRIPTION,
            version=1,
            definition={
                "input": "",
                "steps": [
                    {
                        "agent_name": agent_name,
                        "task": (
                            "Transform the provided source content "
                            f"into a {transformation_type.replace('_', ' ')}."
                        ),
                        "metadata": {
                            "transformation_type": transformation_type,
                            "environment": "development",
                        },
                    },
                ],
                "metadata": {
                    "environment": "development",
                    "managed_by": "development_workspace",
                },
            },
            is_active=True,
        )

        session.add(workflow)
        await session.flush()

    await session.commit()
    await session.refresh(workflow)

    return workflow