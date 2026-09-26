from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_db_session
from app.main import app


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


class FakeDB:
    def __init__(
        self,
        *,
        transformation=None,
        execution=None,
        artifacts=None,
    ):
        self.transformation = transformation
        self.execution = execution
        self.artifacts = artifacts or []
        self.added = None

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

            return FakeScalarResult(
                None,
            )

        raise AssertionError(
            f"Unexpected query: {statement}"
        )

    def add(self, value):
        self.added = value

    async def commit(self):
        if self.added is not None:
            self.added.id = uuid4()

    async def refresh(self, value):
        now = datetime.now(
            timezone.utc,
        )

        value.created_at = now
        value.updated_at = now


def make_transformation(
    *,
    transformation_id,
):
    return SimpleNamespace(
        id=transformation_id,
    )


def make_execution(
    *,
    execution_id,
    transformation_id,
):
    return SimpleNamespace(
        id=execution_id,
        transformation_id=transformation_id,
    )


def make_artifact():
    now = datetime.now(
        timezone.utc,
    )

    return SimpleNamespace(
        id=uuid4(),
        transformation_id=uuid4(),
        execution_id=uuid4(),
        artifact_type="advisory",
        title="Advisory",
        content="Generated advisory content.",
        storage_uri=None,
        content_hash="abc123",
        artifact_metadata={
            "language": "English",
        },
        status="GENERATED",
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_create_artifact_returns_generated_artifact():
    transformation_id = uuid4()
    execution_id = uuid4()

    content = "Generated advisory content."

    db = FakeDB(
        transformation=make_transformation(
            transformation_id=transformation_id,
        ),
        execution=make_execution(
            execution_id=execution_id,
            transformation_id=transformation_id,
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
                "/api/v1/artifacts",
                json={
                    "transformation_id": str(
                        transformation_id,
                    ),
                    "execution_id": str(
                        execution_id,
                    ),
                    "artifact_type": "advisory",
                    "title": "Generated Advisory",
                    "content": content,
                    "metadata": {
                        "language": "English",
                    },
                },
            )

        assert response.status_code == 201

        body = response.json()

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
            == "Generated Advisory"
        )

        assert body["content"] == content

        assert body["status"] == "GENERATED"

        expected_hash = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        assert (
            body["content_hash"]
            == expected_hash
        )

        assert (
            body["metadata"]["language"]
            == "English"
        )

        assert db.added is not None

        assert (
            db.added.content_hash
            == expected_hash
        )

        assert (
            db.added.artifact_metadata
            == {"language": "English"}
        )

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_create_artifact_rejects_missing_transformation():
    db = FakeDB(
        transformation=None,
        execution=None,
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
                "/api/v1/artifacts",
                json={
                    "transformation_id": str(
                        uuid4(),
                    ),
                    "execution_id": str(
                        uuid4(),
                    ),
                    "artifact_type": "advisory",
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
async def test_create_artifact_rejects_missing_execution():
    transformation_id = uuid4()

    db = FakeDB(
        transformation=make_transformation(
            transformation_id=transformation_id,
        ),
        execution=None,
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
                "/api/v1/artifacts",
                json={
                    "transformation_id": str(
                        transformation_id,
                    ),
                    "execution_id": str(
                        uuid4(),
                    ),
                    "artifact_type": "advisory",
                },
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


@pytest.mark.asyncio
async def test_create_artifact_rejects_execution_from_different_transformation():
    transformation_id = uuid4()
    other_transformation_id = uuid4()
    execution_id = uuid4()

    db = FakeDB(
        transformation=make_transformation(
            transformation_id=transformation_id,
        ),
        execution=make_execution(
            execution_id=execution_id,
            transformation_id=other_transformation_id,
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
                "/api/v1/artifacts",
                json={
                    "transformation_id": str(
                        transformation_id,
                    ),
                    "execution_id": str(
                        execution_id,
                    ),
                    "artifact_type": "advisory",
                },
            )

        assert response.status_code == 422

        body = response.json()

        assert body["error"]["code"] == "HTTP_422"
        assert body["error"]["message"] == (
            "Execution and transformation must "
            "refer to the same transformation."
        )
        assert body["error"]["details"] is None

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_artifact_list_returns_items():
    artifact = make_artifact()

    db = FakeDB(
        artifacts=[artifact],
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
                "/api/v1/artifacts",
            )

        assert response.status_code == 200

        body = response.json()

        assert body["total"] == 1

        assert len(
            body["items"]
        ) == 1

        assert (
            body["items"][0]["artifact_type"]
            == "advisory"
        )

        assert (
            body["items"][0]["metadata"]
            == {"language": "English"}
        )

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_artifact_get_returns_artifact():
    artifact = make_artifact()

    db = FakeDB(
        artifacts=[artifact],
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
                f"/api/v1/artifacts/{artifact.id}",
            )

        assert response.status_code == 200

        body = response.json()

        assert (
            body["id"]
            == str(artifact.id)
        )

        assert (
            body["artifact_type"]
            == "advisory"
        )

        assert (
            body["content"]
            == "Generated advisory content."
        )

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_artifact_get_unknown_returns_404():
    db = FakeDB(
        artifacts=[],
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
                f"/api/v1/artifacts/{uuid4()}",
            )

        assert response.status_code == 404

        body = response.json()

        assert body["error"]["code"] == "HTTP_404"
        assert body["error"]["message"] == (
            "Artifact not found."
        )
        assert body["error"]["details"] is None

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_create_artifact_normalizes_structured_content():
    transformation_id = uuid4()
    execution_id = uuid4()

    content = {
        "summary": "Important information",
        "points": [
            "one",
            "two",
        ],
    }

    db = FakeDB(
        transformation=make_transformation(
            transformation_id=transformation_id,
        ),
        execution=make_execution(
            execution_id=execution_id,
            transformation_id=transformation_id,
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
                "/api/v1/artifacts",
                json={
                    "transformation_id": str(
                        transformation_id,
                    ),
                    "execution_id": str(
                        execution_id,
                    ),
                    "artifact_type": "executive_summary",
                    "content": content,
                },
            )

        assert response.status_code == 201

        body = response.json()

        expected_content = (
            '{"points":["one","two"],'
            '"summary":"Important information"}'
        )

        assert (
            body["content"]
            == expected_content
        )

        expected_hash = hashlib.sha256(
            expected_content.encode("utf-8")
        ).hexdigest()

        assert (
            body["content_hash"]
            == expected_hash
        )

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_create_artifact_rejects_extra_request_fields():
    transformation_id = uuid4()
    execution_id = uuid4()

    db = FakeDB(
        transformation=make_transformation(
            transformation_id=transformation_id,
        ),
        execution=make_execution(
            execution_id=execution_id,
            transformation_id=transformation_id,
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
                "/api/v1/artifacts",
                json={
                    "transformation_id": str(
                        transformation_id,
                    ),
                    "execution_id": str(
                        execution_id,
                    ),
                    "artifact_type": "advisory",
                    "unexpected": True,
                },
            )

        assert response.status_code == 422

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )