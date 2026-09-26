from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


API_PREFIX = "/api/v1"


@pytest.mark.asyncio
async def test_create_transformation_rejects_missing_required_fields() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"{API_PREFIX}/transformations",
            json={
                "transformation_type": "advisory",
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_create_transformation_rejects_invalid_type() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.post(
            f"{API_PREFIX}/transformations",
            json={
                "project_id": str(uuid4()),
                "source_id": str(uuid4()),
                "transformation_type": "invalid",
            },
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_unknown_transformation_returns_404() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(
            f"{API_PREFIX}/transformations/{uuid4()}",
        )

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "HTTP_404"
    assert body["error"]["message"] == (
    "Transformation not found."
)
    assert body["error"]["details"] is None


@pytest.mark.asyncio
async def test_list_transformations_returns_response() -> None:
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get(
            f"{API_PREFIX}/transformations",
        )

    assert response.status_code == 200

    body = response.json()

    assert "items" in body
    assert "total" in body
    assert isinstance(body["items"], list)
    assert isinstance(body["total"], int)