from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from app.orchestration.contracts import WorkflowResult
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.agents.base import (
    AgentRegistry,
    AgentResult,
     AgentState,
    AgentStatus,
)
from app.agents.base.contracts import AgentRequest
from app.agents.base.port import AgentPort
from app.orchestration.execution.executor import WorkflowExecutor
from app.orchestration.execution.service import (
    ExecutionOrchestrationService,
)


class FakeAgent(AgentPort):
    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResult:
        return AgentResult(
            status=AgentStatus.COMPLETED,
            output="generated",
        )


class FakeScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class FakeDB:
    def __init__(
        self,
        *,
        execution,
        workflow,
    ):
        self.execution = execution
        self.workflow = workflow
        self.commit_count = 0

    async def execute(self, statement):
        text = str(statement).lower()

        if "executions" in text:
            return FakeScalarResult(
                self.execution,
            )

        if "workflows" in text:
            return FakeScalarResult(
                self.workflow,
            )

        raise AssertionError(
            f"Unexpected query: {statement}"
        )

    async def commit(self):
        self.commit_count += 1

    async def refresh(self, value):
        return None


def make_executor():
    registry = AgentRegistry()

    registry.register(
        "agent",
        FakeAgent(),
    )

    return WorkflowExecutor(
        agent_registry=registry,
    )


def make_execution(
    *,
    execution_id,
    workflow_id,
    transformation_id,
    version=1,
):
    return SimpleNamespace(
        id=execution_id,
        workflow_id=workflow_id,
        transformation_id=transformation_id,
        workflow_version=version,
        execution_context={
            "input": "source",
        },
        status="QUEUED",
        started_at=None,
        completed_at=None,
        error=None,
        metrics={},
    )


def make_workflow(
    *,
    workflow_id,
    project_id,
    version=1,
    definition=None,
):
    if definition is None:
        definition = {
            "steps": [
                {
                    "agent_name": "agent",
                    "task": "Process source",
                }
            ],
        }

    return SimpleNamespace(
        id=workflow_id,
        project_id=project_id,
        version=version,
        definition=definition,
    )


@pytest.mark.asyncio
async def test_execution_service_runs_workflow():
    execution_id = uuid4()
    workflow_id = uuid4()
    transformation_id = uuid4()

    execution = make_execution(
        execution_id=execution_id,
        workflow_id=workflow_id,
        transformation_id=transformation_id,
    )

    workflow = make_workflow(
        workflow_id=workflow_id,
        project_id=uuid4(),
    )

    db = FakeDB(
        execution=execution,
        workflow=workflow,
    )

    service = ExecutionOrchestrationService(
        session=db,
        executor=make_executor(),
    )

    result = await service.execute(
        execution_id,
    )

    assert result.status == "COMPLETED"
    assert result.error is None
    assert result.completed_at is not None
    assert result.metrics["agent_count"] == 1


@pytest.mark.asyncio
async def test_execution_service_preserves_execution_id():
    execution_id = uuid4()
    workflow_id = uuid4()
    transformation_id = uuid4()

    execution = make_execution(
        execution_id=execution_id,
        workflow_id=workflow_id,
        transformation_id=transformation_id,
    )

    workflow = make_workflow(
        workflow_id=workflow_id,
        project_id=uuid4(),
    )

    db = FakeDB(
        execution=execution,
        workflow=workflow,
    )

    service = ExecutionOrchestrationService(
        session=db,
        executor=make_executor(),
    )

    result = await service.execute(
        execution_id,
    )

    assert result.id == execution_id


@pytest.mark.asyncio
async def test_execution_service_rejects_missing_workflow():
    execution_id = uuid4()

    execution = make_execution(
        execution_id=execution_id,
        workflow_id=uuid4(),
        transformation_id=uuid4(),
    )

    db = FakeDB(
        execution=execution,
        workflow=None,
    )

    service = ExecutionOrchestrationService(
        session=db,
        executor=make_executor(),
    )

    result = await service.execute(
        execution_id,
    )

    assert result.status == "FAILED"
    assert result.error == "Workflow not found."


@pytest.mark.asyncio
async def test_execution_service_rejects_version_mismatch():
    execution_id = uuid4()
    workflow_id = uuid4()

    execution = make_execution(
        execution_id=execution_id,
        workflow_id=workflow_id,
        transformation_id=uuid4(),
        version=1,
    )

    workflow = make_workflow(
        workflow_id=workflow_id,
        project_id=uuid4(),
        version=2,
    )

    db = FakeDB(
        execution=execution,
        workflow=workflow,
    )

    service = ExecutionOrchestrationService(
        session=db,
        executor=make_executor(),
    )

    result = await service.execute(
        execution_id,
    )

    assert result.status == "FAILED"
    assert "version mismatch" in result.error.lower()


@pytest.mark.asyncio
async def test_execution_service_rejects_empty_definition():
    execution_id = uuid4()
    workflow_id = uuid4()

    execution = make_execution(
        execution_id=execution_id,
        workflow_id=workflow_id,
        transformation_id=uuid4(),
    )

    workflow = make_workflow(
        workflow_id=workflow_id,
        project_id=uuid4(),
        definition={},
    )

    db = FakeDB(
        execution=execution,
        workflow=workflow,
    )

    service = ExecutionOrchestrationService(
        session=db,
        executor=make_executor(),
    )

    result = await service.execute(
        execution_id,
    )

    assert result.status == "FAILED"
    assert result.error == (
        "Workflow definition must be a non-empty object."
    )

@pytest.mark.asyncio
async def test_execution_service_propagates_workflow_failure():
    execution_id = uuid4()
    workflow_id = uuid4()
    transformation_id = uuid4()

    execution = make_execution(
        execution_id=execution_id,
        workflow_id=workflow_id,
        transformation_id=transformation_id,
    )

    workflow = make_workflow(
        workflow_id=workflow_id,
        project_id=uuid4(),
    )

    executor = make_executor()

    executor.execute = AsyncMock(
        return_value=WorkflowResult(
            status=AgentStatus.FAILED,
            output=None,
              state=AgentState(
        status=AgentStatus.FAILED,
    ),
            metadata={
                "agent_count": 1,
            },
            error="Agent execution failed.",
        )
    )

    db = FakeDB(
        execution=execution,
        workflow=workflow,
    )

    service = ExecutionOrchestrationService(
        session=db,
        executor=executor,
    )

    result = await service.execute(
        execution_id,
    )

    assert result.status == "FAILED"
    assert result.error == "Agent execution failed."

    assert execution.status == "FAILED"
    assert execution.error == "Agent execution failed."
    assert execution.completed_at is not None

    assert db.commit_count == 2


@pytest.mark.asyncio
async def test_execution_service_propagates_workflow_cancellation():
    execution_id = uuid4()
    workflow_id = uuid4()
    transformation_id = uuid4()

    execution = make_execution(
        execution_id=execution_id,
        workflow_id=workflow_id,
        transformation_id=transformation_id,
    )

    workflow = make_workflow(
        workflow_id=workflow_id,
        project_id=uuid4(),
    )

    executor = make_executor()

    executor.execute = AsyncMock(
        side_effect=asyncio.CancelledError(),
    )

    db = FakeDB(
        execution=execution,
        workflow=workflow,
    )

    service = ExecutionOrchestrationService(
        session=db,
        executor=executor,
    )

    result = await service.execute(
        execution_id,
    )

    assert result.status == "CANCELLED"
    assert result.error == "Execution cancelled."

    assert execution.status == "CANCELLED"
    assert execution.error == "Execution cancelled."
    assert execution.completed_at is not None

    assert db.commit_count == 2