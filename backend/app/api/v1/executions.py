from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select

from app.api.dependencies import (
    DatabaseSession,
    WorkflowExecutorDependency,
)
from app.models.execution import Execution
from app.models.transformation import Transformation
from app.models.workflow import Workflow
from app.orchestration.execution.service import (
    ExecutionOrchestrationService,
)
from app.schemas.execution import (
    ExecutionCreateRequest,
    ExecutionListResponse,
    ExecutionResponse,
)

router = APIRouter(
    prefix="/executions",
    tags=["Executions"],
)


@router.post(
    "",
    response_model=ExecutionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_execution(
    request: ExecutionCreateRequest,
    db: DatabaseSession,
    executor: WorkflowExecutorDependency,
) -> Execution:
    """
    Create and execute a workflow job.

    The workflow version is snapshotted at job creation time so a later
    workflow edit cannot silently change an already-created execution.
    """

    transformation_result = await db.execute(
        select(Transformation).where(
            Transformation.id
            == request.transformation_id,
        )
    )

    transformation = (
        transformation_result.scalar_one_or_none()
    )

    if transformation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transformation not found.",
        )

    workflow_result = await db.execute(
        select(Workflow).where(
            Workflow.id == request.workflow_id,
        )
    )

    workflow = workflow_result.scalar_one_or_none()

    if workflow is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found.",
        )

    if (
        workflow.project_id
        != transformation.project_id
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Workflow and transformation must "
                "belong to the same project."
            ),
        )

    if not workflow.is_active:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Workflow is inactive.",
        )

    execution = Execution(
        transformation_id=transformation.id,
        workflow_id=workflow.id,
        workflow_version=workflow.version,
        status="QUEUED",
        execution_context=dict(
            request.execution_context,
        ),
        metrics={},
    )

    db.add(execution)
    await db.commit()
    await db.refresh(execution)

    orchestration_service = ExecutionOrchestrationService(
        session=db,
        executor=executor,
    )

    await orchestration_service.execute(
        execution.id,
    )

    await db.refresh(execution)

    return execution


@router.get(
    "",
    response_model=ExecutionListResponse,
)
async def list_executions(
    db: DatabaseSession,
) -> ExecutionListResponse:
    """List execution jobs ordered from newest to oldest."""

    result = await db.execute(
        select(Execution).order_by(
            Execution.created_at.desc(),
        )
    )

    items = list(result.scalars().all())

    total_result = await db.execute(
        select(func.count()).select_from(
            Execution,
        )
    )

    total = total_result.scalar_one()

    return ExecutionListResponse(
        items=items,
        total=total,
    )


@router.get(
    "/{execution_id}",
    response_model=ExecutionResponse,
)
async def get_execution(
    execution_id: UUID,
    db: DatabaseSession,
) -> Execution:
    """Return one execution job by ID."""

    result = await db.execute(
        select(Execution).where(
            Execution.id == execution_id,
        )
    )

    execution = result.scalar_one_or_none()

    if execution is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found.",
        )

    return execution