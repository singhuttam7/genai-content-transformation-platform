from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.agents.base import (
    AgentRegistry,
    AgentResult,
    AgentStatus,
)
from app.agents.base.contracts import AgentRequest
from app.agents.base.port import AgentPort
from app.api.dependencies import (
    get_db_session,
    get_workflow_executor,
)
from app.main import app
from app.orchestration.execution.executor import WorkflowExecutor


class FakeScalarResult:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalar_one(self):
        return self._value


class FakeScalars:
    def __init__(self, values):
        self._values = values

    def all(self):
        return self._values


class FakeListResult:
    def __init__(self, values):
        self._values = values

    def scalars(self):
        return FakeScalars(self._values)

    def scalar_one_or_none(self):
        if len(self._values) == 1:
            return self._values[0]

        return None


class FakeDB:
    def __init__(
        self,
        *,
        transformation=None,
        workflow=None,
        executions=None,
    ):
        self.transformation = transformation
        self.workflow = workflow
        self.executions = executions or []
        self.added = None
        self.commit_count = 0

    async def execute(self, statement):
        text = str(statement).lower()

        if "count(" in text:
            return FakeScalarResult(
                len(self.executions),
            )

        if "transformations" in text:
            return FakeScalarResult(
                self.transformation,
            )

        if "workflows" in text:
            return FakeScalarResult(
                self.workflow,
            )

        if "executions" in text:
            if "order by" in text:
                return FakeListResult(
                    self.executions,
                )

            if len(self.executions) == 1:
                return FakeScalarResult(
                    self.executions[0],
                )

            return FakeScalarResult(None)

        raise AssertionError(
            f"Unexpected query: {statement}"
        )

    def add(self, value):
        self.added = value

    async def commit(self):
        self.commit_count += 1

        if self.added is not None:
            self.added.id = uuid4()

            if self.added not in self.executions:
                self.executions.append(
                    self.added,
                )

    async def refresh(self, value):
        now = datetime.now(timezone.utc)

        if getattr(value, "created_at", None) is None:
            value.created_at = now

        value.updated_at = now


class FakeAgent(AgentPort):
    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResult:
        return AgentResult(
            status=AgentStatus.COMPLETED,
            output="generated-output",
        )


def make_transformation(
    *,
    project_id,
    transformation_id,
):
    return SimpleNamespace(
        id=transformation_id,
        project_id=project_id,
    )


def make_workflow(
    *,
    project_id,
    workflow_id,
    version=3,
    is_active=True,
    definition=None,
):
    if definition is None:
        definition = {
            "steps": [
                {
                    "agent_name": "agent",
                    "task": "Process execution",
                }
            ],
        }

    return SimpleNamespace(
        id=workflow_id,
        project_id=project_id,
        version=version,
        is_active=is_active,
        definition=definition,
    )


def make_executor() -> WorkflowExecutor:
    registry = AgentRegistry()

    registry.register(
        "agent",
        FakeAgent(),
    )

    return WorkflowExecutor(
        agent_registry=registry,
    )


@pytest.mark.asyncio
async def test_create_execution_runs_workflow_and_returns_completed_job():
    project_id = uuid4()
    transformation_id = uuid4()
    workflow_id = uuid4()

    transformation = make_transformation(
        project_id=project_id,
        transformation_id=transformation_id,
    )

    workflow = make_workflow(
        project_id=project_id,
        workflow_id=workflow_id,
    )

    db = FakeDB(
        transformation=transformation,
        workflow=workflow,
    )

    app.dependency_overrides[
        get_db_session
    ] = lambda: db

    app.dependency_overrides[
        get_workflow_executor
    ] = make_executor

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/executions",
                json={
                    "transformation_id": str(
                        transformation_id,
                    ),
                    "workflow_id": str(
                        workflow_id,
                    ),
                    "execution_context": {
                        "input": "source-content",
                    },
                },
            )

        assert response.status_code == 201

        body = response.json()

        assert body["status"] == "COMPLETED"
        assert body["workflow_version"] == 3

        assert (
            body["transformation_id"]
            == str(transformation_id)
        )

        assert (
            body["workflow_id"]
            == str(workflow_id)
        )

        assert body["started_at"] is not None
        assert body["completed_at"] is not None
        assert body["error"] is None

        assert db.added is not None

        assert (
            db.added.execution_context
            == {
                "input": "source-content",
            }
        )

        assert db.added.status == "COMPLETED"
        assert db.commit_count == 3

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )

        app.dependency_overrides.pop(
            get_workflow_executor,
            None,
        )


@pytest.mark.asyncio
async def test_create_execution_rejects_missing_transformation():
    db = FakeDB(
        transformation=None,
        workflow=make_workflow(
            project_id=uuid4(),
            workflow_id=uuid4(),
        ),
    )

    app.dependency_overrides[
        get_db_session
    ] = lambda: db

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/executions",
                json={
                    "transformation_id": str(
                        uuid4(),
                    ),
                    "workflow_id": str(
                        uuid4(),
                    ),
                },
            )

        assert response.status_code == 404

        body = response.json()

        assert body["error"]["code"] == "HTTP_404"
        assert body["error"]["message"] == (
            "Transformation not found."
        )
        assert body["error"]["details"] is None

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_create_execution_rejects_missing_workflow():
    db = FakeDB(
        transformation=make_transformation(
            project_id=uuid4(),
            transformation_id=uuid4(),
        ),
        workflow=None,
    )

    app.dependency_overrides[
        get_db_session
    ] = lambda: db

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/executions",
                json={
                    "transformation_id": str(
                        db.transformation.id,
                    ),
                    "workflow_id": str(
                        uuid4(),
                    ),
                },
            )

        assert response.status_code == 404

        body = response.json()

        assert body["error"]["code"] == "HTTP_404"
        assert body["error"]["message"] == (
            "Workflow not found."
        )
        assert body["error"]["details"] is None

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_create_execution_rejects_project_mismatch():
    transformation_project = uuid4()
    workflow_project = uuid4()

    db = FakeDB(
        transformation=make_transformation(
            project_id=transformation_project,
            transformation_id=uuid4(),
        ),
        workflow=make_workflow(
            project_id=workflow_project,
            workflow_id=uuid4(),
        ),
    )

    app.dependency_overrides[
        get_db_session
    ] = lambda: db

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/executions",
                json={
                    "transformation_id": str(
                        db.transformation.id,
                    ),
                    "workflow_id": str(
                        db.workflow.id,
                    ),
                },
            )

        assert response.status_code == 422

        body = response.json()

        assert body["error"]["code"] == "HTTP_422"
        assert body["error"]["message"] == (
            "Workflow and transformation must "
            "belong to the same project."
        )
        assert body["error"]["details"] is None

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_create_execution_rejects_inactive_workflow():
    project_id = uuid4()

    db = FakeDB(
        transformation=make_transformation(
            project_id=project_id,
            transformation_id=uuid4(),
        ),
        workflow=make_workflow(
            project_id=project_id,
            workflow_id=uuid4(),
            is_active=False,
        ),
    )

    app.dependency_overrides[
        get_db_session
    ] = lambda: db

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/api/v1/executions",
                json={
                    "transformation_id": str(
                        db.transformation.id,
                    ),
                    "workflow_id": str(
                        db.workflow.id,
                    ),
                },
            )

        assert response.status_code == 422

        body = response.json()

        assert body["error"]["code"] == "HTTP_422"
        assert body["error"]["message"] == (
            "Workflow is inactive."
        )
        assert body["error"]["details"] is None

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_execution_list_returns_items():
    execution = SimpleNamespace(
        id=uuid4(),
        transformation_id=uuid4(),
        workflow_id=uuid4(),
        workflow_version=2,
        status="QUEUED",
        started_at=None,
        completed_at=None,
        error=None,
        execution_context={},
        metrics={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    db = FakeDB(
        executions=[execution],
    )

    app.dependency_overrides[
        get_db_session
    ] = lambda: db

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                "/api/v1/executions",
            )

        assert response.status_code == 200

        body = response.json()

        assert body["total"] == 1
        assert len(body["items"]) == 1

        assert (
            body["items"][0]["status"]
            == "QUEUED"
        )

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_execution_get_unknown_returns_404():
    db = FakeDB(
        executions=[],
    )

    app.dependency_overrides[
        get_db_session
    ] = lambda: db

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"/api/v1/executions/{uuid4()}",
            )

        assert response.status_code == 404

        body = response.json()

        assert body["error"]["code"] == "HTTP_404"
        assert body["error"]["message"] == (
            "Execution not found."
        )
        assert body["error"]["details"] is None

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )