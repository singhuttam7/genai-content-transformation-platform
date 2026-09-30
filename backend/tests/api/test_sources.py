from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.dependencies import get_db_session
from app.ingestion.results import IngestionResult
from app.ingestion.schemas import (
    CanonicalContent,
    ContentBlock,
    ContentBlockType,
    InputType,
    ProcessingStatus,
    SourceReference,
)
from app.main import app


API_PREFIX = "/api/v1"


# ============================================================
# Existing ingestion fixtures
# ============================================================


def make_ingestion_result():
    source_id = uuid4()

    canonical = CanonicalContent(
        source=SourceReference(
            source_id=source_id,
            source_type=InputType.TEXT,
            title="Test Source",
            filename=None,
            mime_type="text/plain",
            content_hash="abc123",
            storage_uri="test://source",
        ),
        title="Test Source",
        language="en",
        text="This is test source content.",
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="This is test source content.",
                order=0,
            ),
        ],
        entities=[],
        topics=[],
        claims=[],
        keywords=[],
        context={},
        provenance={
            "ingestion": "test",
        },
        metadata={
            "fixture": "api-test",
        },
    )

    return IngestionResult(
        source_id=source_id,
        canonical_content=canonical,
        storage_key="sources/test.txt",
        storage_uri="storage://sources/test.txt",
        content_hash="abc123",
        status=ProcessingStatus.COMPLETED,
        metadata={
            "api_test": True,
        },
    )


# ============================================================
# Source-list test fixtures
# ============================================================


class FakeScalarResult:
    def __init__(self, value):
        self._value = value

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


class SourceListFakeDB:
    def __init__(self, sources):
        self.sources = sources

    async def execute(self, statement):
        text = str(statement).lower()

        if "count(" in text:
            return FakeScalarResult(
                len(self.sources),
            )

        if "sources" in text:
            return FakeListResult(
                self.sources,
            )

        raise AssertionError(
            f"Unexpected query: {statement}"
        )


def make_source(
    *,
    project_id,
    source_id=None,
    source_type="text",
    title="Test Source",
    status="completed",
    created_at=None,
):
    now = (
        created_at
        or datetime.now(timezone.utc)
    )

    return SimpleNamespace(
        id=source_id or uuid4(),
        project_id=project_id,
        source_type=source_type,
        title=title,
        original_filename=None,
        mime_type="text/plain",
        storage_uri="storage://test/source",
        content_hash="abc123",
        source_metadata={
            "fixture": "source-list-test",
        },
        status=status,
        created_at=now,
        updated_at=now,
    )


# ============================================================
# Source creation API tests
# ============================================================


@pytest.mark.asyncio
async def test_create_source_rejects_missing_input_type() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"{API_PREFIX}/sources",
            json={
                "content": "Test content",
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_source_rejects_invalid_input_type() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"{API_PREFIX}/sources",
            json={
                "input_type": "invalid",
                "content": "Test content",
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_source_rejects_empty_text_content() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"{API_PREFIX}/sources",
            json={
                "input_type": "text",
                "content": "",
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_source_delegates_to_ingestion_service() -> None:
    result = make_ingestion_result()

    mock_ingest = AsyncMock(
        return_value=result,
    )

    with patch(
        "app.api.v1.sources.IngestionApplicationService.ingest",
        mock_ingest,
    ):
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"{API_PREFIX}/sources",
                json={
                    "input_type": "text",
                    "title": "Test Source",
                    "content": "This is test source content.",
                    "metadata": {
                        "fixture": "api-test",
                    },
                },
            )

    assert response.status_code == 201

    body = response.json()

    assert body["source_id"] == str(result.source_id)
    assert body["status"] == "completed"
    assert body["storage_key"] == "sources/test.txt"
    assert body["storage_uri"] == "storage://sources/test.txt"
    assert body["content_hash"] == "abc123"
    assert body["title"] == "Test Source"
    assert body["source_type"] == "text"
    assert body["canonical_text"] == (
        "This is test source content."
    )
    assert len(body["segments"]) == 1

    assert (
        body["segments"][0]["content"]
        == "This is test source content."
    )

    assert body["metadata"]["fixture"] == "api-test"
    assert body["metadata"]["api_test"] is True
    assert body["provenance"]["ingestion"] == "test"

    mock_ingest.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_source_maps_value_error_to_422() -> None:
    mock_ingest = AsyncMock(
        side_effect=ValueError(
            "Invalid ingestion request.",
        ),
    )

    with patch(
        "app.api.v1.sources.IngestionApplicationService.ingest",
        mock_ingest,
    ):
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"{API_PREFIX}/sources",
                json={
                    "input_type": "text",
                    "content": "Test content",
                },
            )

    assert response.status_code == 422

    body = response.json()

    assert body["error"]["code"] == "HTTP_422"
    assert body["error"]["message"] == (
        "Invalid ingestion request."
    )
    assert body["error"]["details"] is None


@pytest.mark.asyncio
async def test_create_source_preserves_source_id() -> None:
    source_id = uuid4()

    result = make_ingestion_result()

    result = result.model_copy(
        update={
            "source_id": source_id,
            "canonical_content": (
                result.canonical_content.model_copy(
                    update={
                        "source": (
                            result.canonical_content.source.model_copy(
                                update={
                                    "source_id": source_id,
                                },
                            )
                        ),
                    },
                )
            ),
        },
    )

    mock_ingest = AsyncMock(
        return_value=result,
    )

    with patch(
        "app.api.v1.sources.IngestionApplicationService.ingest",
        mock_ingest,
    ):
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"{API_PREFIX}/sources",
                json={
                    "source_id": str(source_id),
                    "input_type": "text",
                    "content": "Test content",
                },
            )

    assert response.status_code == 201
    assert response.json()["source_id"] == str(
        source_id
    )


# ============================================================
# URL Source API Tests
# ============================================================


@pytest.mark.asyncio
async def test_create_url_source_delegates_to_ingestion_service() -> None:
    result = make_ingestion_result()

    mock_ingest = AsyncMock(
        return_value=result,
    )

    with patch(
        "app.api.v1.sources.IngestionApplicationService.ingest",
        mock_ingest,
    ):
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                f"{API_PREFIX}/sources",
                json={
                    "input_type": "url",
                    "title": "AI Article",
                    "url": "https://example.com/article",
                    "metadata": {
                        "fixture": "url-api-test",
                    },
                },
            )

    assert response.status_code == 201

    body = response.json()

    assert body["source_id"] == str(result.source_id)
    assert body["status"] == "completed"
    assert body["storage_key"] == "sources/test.txt"
    assert body["storage_uri"] == "storage://sources/test.txt"
    assert body["content_hash"] == "abc123"

    mock_ingest.assert_awaited_once()

    ingestion_request = (
        mock_ingest.await_args.kwargs["request"]
    )

    assert ingestion_request.input_type == InputType.URL
    assert ingestion_request.url == (
        "https://example.com/article"
    )
    assert ingestion_request.title == "AI Article"
    assert ingestion_request.metadata == {
        "fixture": "url-api-test",
    }


@pytest.mark.asyncio
async def test_create_url_source_rejects_missing_url() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"{API_PREFIX}/sources",
            json={
                "input_type": "url",
                "title": "Missing URL",
            },
        )

    assert response.status_code == 422


# ============================================================
# Source List API Tests
# ============================================================


@pytest.mark.asyncio
async def test_list_sources_returns_items() -> None:
    project_id = uuid4()

    source = make_source(
        project_id=project_id,
        title="Knowledge Source",
    )

    db = SourceListFakeDB(
        sources=[source],
    )

    app.dependency_overrides[
        get_db_session
    ] = lambda: db

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"{API_PREFIX}/sources",
                params={
                    "project_id": str(project_id),
                },
            )

        assert response.status_code == 200

        body = response.json()

        assert body["total"] == 1
        assert len(body["items"]) == 1

        item = body["items"][0]

        assert item["id"] == str(source.id)
        assert item["project_id"] == str(
            project_id
        )
        assert item["source_type"] == "text"
        assert item["title"] == "Knowledge Source"
        assert item["status"] == "completed"

        assert (
            item["metadata"]["fixture"]
            == "source-list-test"
        )

        assert "canonical_text" not in item
        assert "segments" not in item

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_list_sources_requires_project_id() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(
            f"{API_PREFIX}/sources",
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_list_sources_preserves_project_scope() -> None:
    project_id = uuid4()

    source = make_source(
        project_id=project_id,
    )

    db = SourceListFakeDB(
        sources=[source],
    )

    app.dependency_overrides[
        get_db_session
    ] = lambda: db

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"{API_PREFIX}/sources",
                params={
                    "project_id": str(project_id),
                },
            )

        assert response.status_code == 200

        body = response.json()

        assert body["total"] == 1

        assert (
            body["items"][0]["project_id"]
            == str(project_id)
        )

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_list_sources_supports_source_type_filter() -> None:
    project_id = uuid4()

    source = make_source(
        project_id=project_id,
        source_type="pdf",
        title="Research Paper",
    )

    db = SourceListFakeDB(
        sources=[source],
    )

    app.dependency_overrides[
        get_db_session
    ] = lambda: db

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"{API_PREFIX}/sources",
                params={
                    "project_id": str(project_id),
                    "source_type": "pdf",
                },
            )

        assert response.status_code == 200

        body = response.json()

        assert body["total"] == 1
        assert (
            body["items"][0]["source_type"]
            == "pdf"
        )

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )


@pytest.mark.asyncio
async def test_list_sources_supports_status_filter() -> None:
    project_id = uuid4()

    source = make_source(
        project_id=project_id,
        status="processing",
    )

    db = SourceListFakeDB(
        sources=[source],
    )

    app.dependency_overrides[
        get_db_session
    ] = lambda: db

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.get(
                f"{API_PREFIX}/sources",
                params={
                    "project_id": str(project_id),
                    "source_status": "processing",
                },
            )

        assert response.status_code == 200

        body = response.json()

        assert body["total"] == 1
        assert (
            body["items"][0]["status"]
            == "processing"
        )

    finally:
        app.dependency_overrides.pop(
            get_db_session,
            None,
        )