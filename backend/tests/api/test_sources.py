from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

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
    assert body["canonical_text"] == "This is test source content."

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
            "canonical_content": result.canonical_content.model_copy(
                update={
                    "source": result.canonical_content.source.model_copy(
                        update={
                            "source_id": source_id,
                        },
                    ),
                },
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
    assert response.json()["source_id"] == str(source_id)