from __future__ import annotations

import hashlib
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
    get_artifact_persistence_service,
    get_db_session,
    get_rag_retrieval_service,
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

    def all(self):
        if self._value is None:
            return []

        return [self._value]


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


class IntegrationDB:
    def __init__(
        self,
        *,
        transformation=None,
        workflow=None,
        execution=None,
        artifacts=None,
    ):
        self.transformation = transformation
        self.workflow = workflow
        self.execution = execution
        self.artifacts = artifacts or []
        self.added = []

    async def execute(self, statement):
        text = str(statement).lower()

        if "count(" in text:
            return FakeScalarResult(
                len(self.artifacts),
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
            return FakeScalarResult(
                self.execution,
            )

        if "artifacts" in text:
            if "order by" in text:
                return FakeListResult(
                    self.artifacts,
                )

            if len(self.artifacts) == 1:
                return FakeScalarResult(
                    self.artifacts[0],
                )

            return FakeScalarResult(None)

        raise AssertionError(
            f"Unexpected query: {statement}"
        )

    def add(self, value):
        self.added.append(value)

        if hasattr(value, "id") and value.id is None:
            value.id = uuid4()

    async def commit(self):
        for value in self.added:
            if (
                value.__class__.__name__
                == "Execution"
            ):
                self.execution = value

            elif (
                value.__class__.__name__
                == "Artifact"
            ):
                if value not in self.artifacts:
                    self.artifacts.append(value)

    async def refresh(self, value):
        now = datetime.now(timezone.utc)

        if getattr(value, "created_at", None) is None:
            value.created_at = now

        value.updated_at = now


class FakeArtifactPersistenceService:
    """
    Lightweight artifact persistence service for integration tests.

    The production ArtifactPersistenceService requires a real
    AsyncSession. These integration tests intentionally use
    IntegrationDB, so the real service must be overridden here.
    """

    def __init__(self, db: IntegrationDB) -> None:
        self.db = db

    async def persist(
        self,
        *,
        envelope,
        transformation_id,
        execution_id,
        status="GENERATED",
        storage_uri=None,
    ):
        content = envelope.content

        if isinstance(content, str):
            normalized_content = content
        else:
            import json

            normalized_content = json.dumps(
                content,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )

        content_hash = hashlib.sha256(
            normalized_content.encode("utf-8")
        ).hexdigest()

        artifact = SimpleNamespace(
            id=uuid4(),
            transformation_id=transformation_id,
            execution_id=execution_id,
            artifact_type=envelope.artifact_type,
            title=envelope.title,
            content=normalized_content,
            storage_uri=storage_uri,
            content_hash=content_hash,
            artifact_metadata=envelope.metadata or {},
            status=status,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        self.db.add(artifact)

        return artifact


class IntegrationAgent(AgentPort):
    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResult:
        return AgentResult(
            status=AgentStatus.COMPLETED,
            output={
                "task": request.task,
                "input": request.input,
            },
        )


def make_transformation(
    *,
    transformation_id,
    project_id,
):
    return SimpleNamespace(
        id=transformation_id,
        project_id=project_id,
    )


def make_workflow(
    *,
    workflow_id,
    project_id,
    version=1,
):
    return SimpleNamespace(
        id=workflow_id,
        project_id=project_id,
        version=version,
        is_active=True,
        definition={
            "steps": [
                {
                    "agent_name": "integration-agent",
                    "task": "Transform source content",
                }
            ],
        },
    )


def make_executor():
    registry = AgentRegistry()

    registry.register(
        "integration-agent",
        IntegrationAgent(),
    )

    return WorkflowExecutor(
        agent_registry=registry,
    )


@pytest.mark.asyncio
async def test_transformation_to_execution_to_completion():
    project_id = uuid4()
    transformation_id = uuid4()
    workflow_id = uuid4()

    transformation = make_transformation(
        transformation_id=transformation_id,
        project_id=project_id,
    )

    workflow = make_workflow(
        workflow_id=workflow_id,
        project_id=project_id,
    )

    db = IntegrationDB(
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
            execution_response = await client.post(
                "/api/v1/executions",
                json={
                    "transformation_id": str(
                        transformation_id,
                    ),
                    "workflow_id": str(
                        workflow_id,
                    ),
                    "execution_context": {
                        "input": "integration-test-source-content",
                        "source": "integration-test",
                    },
                },
            )

        assert execution_response.status_code == 201

        execution_body = (
            execution_response.json()
        )

        assert (
            execution_body["status"]
            == "COMPLETED"
        )

        assert (
            execution_body["transformation_id"]
            == str(transformation_id)
        )

        assert (
            execution_body["workflow_id"]
            == str(workflow_id)
        )

        assert (
            execution_body["workflow_version"]
            == 1
        )

        assert execution_body["error"] is None

        assert db.execution is not None

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
async def test_execution_and_artifact_api_lifecycle():
    project_id = uuid4()
    transformation_id = uuid4()
    workflow_id = uuid4()
    execution_id = uuid4()

    transformation = make_transformation(
        transformation_id=transformation_id,
        project_id=project_id,
    )

    workflow = make_workflow(
        workflow_id=workflow_id,
        project_id=project_id,
    )

    execution = SimpleNamespace(
        id=execution_id,
        transformation_id=transformation_id,
        workflow_id=workflow_id,
        workflow_version=1,
        status="COMPLETED",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        error=None,
        execution_context={},
        metrics={},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    db = IntegrationDB(
        transformation=transformation,
        workflow=workflow,
        execution=execution,
    )

    app.dependency_overrides[
        get_db_session
    ] = lambda: db

    app.dependency_overrides[
        get_artifact_persistence_service
    ] = lambda: FakeArtifactPersistenceService(db)

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            artifact_response = await client.post(
                "/api/v1/artifacts",
                json={
                    "transformation_id": str(
                        transformation_id,
                    ),
                    "execution_id": str(
                        execution_id,
                    ),
                    "artifact_type": "advisory",
                    "title": "Integration Artifact",
                    "content": (
                        "Generated integration "
                        "artifact content."
                    ),
                    "metadata": {
                        "source": "integration-test",
                    },
                },
            )

        assert artifact_response.status_code == 201

        body = artifact_response.json()

        assert (
            body["transformation_id"]
            == str(transformation_id)
        )

        assert (
            body["execution_id"]
            == str(execution_id)
        )

        assert (
            body["artifact_type"]
            == "advisory"
        )

        assert (
            body["title"]
            == "Integration Artifact"
        )

        assert body["status"] == "GENERATED"

        assert (
            body["metadata"]["source"]
            == "integration-test"
        )

    finally:
        app.dependency_overrides.pop(
            get_artifact_persistence_service,
            None,
        )

        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_execution_lifecycle_can_be_retrieved_after_creation():
    project_id = uuid4()
    transformation_id = uuid4()
    workflow_id = uuid4()

    transformation = make_transformation(
        transformation_id=transformation_id,
        project_id=project_id,
    )

    workflow = make_workflow(
        workflow_id=workflow_id,
        project_id=project_id,
    )

    db = IntegrationDB(
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
            create_response = await client.post(
                "/api/v1/executions",
                json={
                    "transformation_id": str(
                        transformation_id,
                    ),
                    "workflow_id": str(
                        workflow_id,
                    ),
                    "execution_context": {
                        "input": "integration-test-source-content",
                    },
                },
            )

            assert (
                create_response.status_code
                == 201
            )

            execution_id = create_response.json()[
                "id"
            ]

            get_response = await client.get(
                f"/api/v1/executions/{execution_id}",
            )

        assert get_response.status_code == 200

        body = get_response.json()

        assert body["id"] == execution_id
        assert body["status"] == "COMPLETED"

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )

        app.dependency_overrides.pop(
            get_workflow_executor,
            None,
        )