from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentStatus
from app.models.execution import Execution
from app.models.workflow import Workflow
from app.orchestration.contracts import (
    WorkflowRequest,
    WorkflowResult,
    WorkflowStep,
)
from app.orchestration.execution.executor import WorkflowExecutor


class ExecutionOrchestrationResult(BaseModel):
    """
    Application-level result returned by ExecutionOrchestrationService.

    This represents the persisted execution job and is intentionally
    separate from the internal WorkflowResult contract.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    id: UUID

    status: str

    output: Any = None

    workflow_result: WorkflowResult | None = None

    error: str | None = None

    started_at: datetime | None = None

    completed_at: datetime | None = None

    metrics: dict[str, Any] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class ExecutionOrchestrationService:
    """
    Connect persisted executions to the existing WorkflowExecutor.

    This service:
    - loads the persisted execution;
    - loads the persisted workflow;
    - verifies the workflow version;
    - validates the workflow definition;
    - builds a WorkflowRequest;
    - invokes the existing WorkflowExecutor;
    - updates the execution lifecycle;
    - returns an execution-level result.

    It does not implement another workflow engine.
    """

    def __init__(
        self,
        *,
        session: AsyncSession,
        executor: WorkflowExecutor,
    ) -> None:
        self.session = session
        self.executor = executor

    async def execute(
        self,
        execution_id: UUID,
    ) -> ExecutionOrchestrationResult:
        execution = await self._get_execution(
            execution_id,
        )

        if execution is None:
            return self._failure_result(
                execution_id=execution_id,
                error="Execution not found.",
            )

        workflow = await self._get_workflow(
            execution.workflow_id,
        )

        if workflow is None:
            return await self._persisted_failure_result(
                execution=execution,
                error="Workflow not found.",
            )

        if workflow.version != execution.workflow_version:
            return await self._persisted_failure_result(
                execution=execution,
                error="Workflow version mismatch.",
            )

        try:
            workflow_request = self._build_workflow_request(
                execution=execution,
                workflow=workflow,
            )
        except ValueError as exc:
            return await self._persisted_failure_result(
                execution=execution,
                error=str(exc),
            )

        started_at = datetime.now(
            timezone.utc,
        )

        execution.status = "RUNNING"
        execution.started_at = started_at
        execution.completed_at = None
        execution.error = None

        await self.session.commit()

        try:
            workflow_result = await self.executor.execute(
                workflow_request,
            )

        except asyncio.CancelledError:
            completed_at = datetime.now(
                timezone.utc,
            )

            execution.status = "CANCELLED"
            execution.completed_at = completed_at
            execution.error = "Execution cancelled."

            await self.session.commit()

            return ExecutionOrchestrationResult(
                id=execution.id,
                status="CANCELLED",
                output=None,
                workflow_result=None,
                error="Execution cancelled.",
                started_at=started_at,
                completed_at=completed_at,
                metrics=dict(
                    execution.metrics or {},
                ),
                metadata={
                    "execution_id": str(
                        execution.id,
                    ),
                },
            )

        except Exception as exc:
            return await self._persisted_failure_result(
                execution=execution,
                error=str(exc),
                started_at=started_at,
            )

        await self._apply_workflow_result(
            execution=execution,
            result=workflow_result,
        )

        return self._build_execution_result(
            execution=execution,
            workflow_result=workflow_result,
        )

    async def _get_execution(
        self,
        execution_id: UUID,
    ) -> Execution | None:
        result = await self.session.execute(
            select(Execution).where(
                Execution.id == execution_id,
            )
        )

        return result.scalar_one_or_none()

    async def _get_workflow(
        self,
        workflow_id: UUID,
    ) -> Workflow | None:
        result = await self.session.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
            )
        )

        return result.scalar_one_or_none()

    @staticmethod
    def _build_workflow_request(
        *,
        execution: Execution,
        workflow: Workflow,
    ) -> WorkflowRequest:
        definition = workflow.definition

        if not isinstance(
            definition,
            dict,
        ):
            raise ValueError(
                "Workflow definition must be a non-empty object."
            )

        if not definition:
            raise ValueError(
                "Workflow definition must be a non-empty object."
            )

        raw_steps = definition.get(
            "steps",
        )

        if not isinstance(
            raw_steps,
            list,
        ):
            raise ValueError(
                "Workflow definition must contain a non-empty steps list."
            )

        if not raw_steps:
            raise ValueError(
                "Workflow definition must contain a non-empty steps list."
            )

        steps: list[WorkflowStep] = []

        for index, raw_step in enumerate(
            raw_steps,
        ):
            if not isinstance(
                raw_step,
                dict,
            ):
                raise ValueError(
                    f"Workflow step at index {index} must be an object."
                )

            agent_name = raw_step.get(
                "agent_name",
            )

            task = raw_step.get(
                "task",
            )

            if not isinstance(
                agent_name,
                str,
            ) or not agent_name.strip():
                raise ValueError(
                    f"Workflow step at index {index} "
                    "must contain a non-empty agent_name."
                )

            if not isinstance(
                task,
                str,
            ) or not task.strip():
                raise ValueError(
                    f"Workflow step at index {index} "
                    "must contain a non-empty task."
                )

            step_input = raw_step.get(
                "input",
            )

            metadata = raw_step.get(
                "metadata",
                {},
            )

            if not isinstance(
                metadata,
                dict,
            ):
                raise ValueError(
                    f"Workflow step at index {index} "
                    "metadata must be an object."
                )

            steps.append(
                WorkflowStep(
                    agent_name=agent_name,
                    task=task,
                    input=step_input,
                    metadata=dict(metadata),
                )
            )

        execution_context = dict(
            execution.execution_context or {},
        )

        workflow_input = execution_context.get(
            "input",
        )

        if workflow_input is None:
            workflow_input = definition.get(
                "input",
            )

        if workflow_input is None:
            workflow_input = ""

        definition_metadata = definition.get(
            "metadata",
            {},
        )

        if not isinstance(
            definition_metadata,
            dict,
        ):
            definition_metadata = {}

        metadata = {
            **definition_metadata,
            "execution_id": str(
                execution.id,
            ),
            "workflow_id": str(
                workflow.id,
            ),
            "workflow_version": workflow.version,
        }

        return WorkflowRequest(
            input=workflow_input,
            steps=steps,
            metadata=metadata,
        )

    async def _apply_workflow_result(
        self,
        *,
        execution: Execution,
        result: WorkflowResult,
    ) -> None:
        if result.status == AgentStatus.COMPLETED:
            execution.status = "COMPLETED"

        elif result.status == AgentStatus.CANCELLED:
            execution.status = "CANCELLED"

        else:
            execution.status = "FAILED"

        execution.completed_at = datetime.now(
            timezone.utc,
        )

        execution.error = result.error

        execution.metrics = {
            **dict(
                execution.metrics or {},
            ),
            **dict(
                result.metadata or {},
            ),
        }

        await self.session.commit()

        try:
            await self.session.refresh(
                execution,
            )
        except AttributeError:
            pass

    async def _persisted_failure_result(
        self,
        *,
        execution: Execution,
        error: str,
        started_at: datetime | None = None,
    ) -> ExecutionOrchestrationResult:
        completed_at = datetime.now(
            timezone.utc,
        )

        execution.status = "FAILED"

        if (
            execution.started_at is None
            and started_at is not None
        ):
            execution.started_at = started_at

        execution.completed_at = completed_at
        execution.error = error

        await self.session.commit()

        return ExecutionOrchestrationResult(
            id=execution.id,
            status="FAILED",
            output=None,
            workflow_result=None,
            error=error,
            started_at=execution.started_at,
            completed_at=completed_at,
            metrics=dict(
                execution.metrics or {},
            ),
            metadata={
                "execution_id": str(
                    execution.id,
                ),
            },
        )

    @staticmethod
    def _failure_result(
        *,
        execution_id: UUID,
        error: str,
    ) -> ExecutionOrchestrationResult:
        return ExecutionOrchestrationResult(
            id=execution_id,
            status="FAILED",
            output=None,
            workflow_result=None,
            error=error,
            started_at=None,
            completed_at=None,
            metrics={},
            metadata={
                "execution_id": str(
                    execution_id,
                ),
            },
        )

    @staticmethod
    def _build_execution_result(
        *,
        execution: Execution,
        workflow_result: WorkflowResult,
    ) -> ExecutionOrchestrationResult:
        if workflow_result.status == AgentStatus.COMPLETED:
            status = "COMPLETED"

        elif workflow_result.status == AgentStatus.CANCELLED:
            status = "CANCELLED"

        else:
            status = "FAILED"

        return ExecutionOrchestrationResult(
            id=execution.id,
            status=status,
            output=workflow_result.output,
            workflow_result=workflow_result,
            error=workflow_result.error,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            metrics=dict(
                execution.metrics or {},
            ),
            metadata={
                **dict(
                    workflow_result.metadata or {},
                ),
                "execution_id": str(
                    execution.id,
                ),
            },
        )