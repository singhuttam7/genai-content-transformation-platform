from __future__ import annotations

import asyncio
from dataclasses import dataclass
from ipaddress import ip_address
from uuid import uuid4

import pytest

from app.database.session import SessionFactory
from app.ingestion.application_service import IngestionApplicationService
from app.ingestion.fetchers.connection import ConnectionTarget
from app.ingestion.fetchers.http import HTTPFetcher
from app.ingestion.schemas import InputType, ProcessingStatus
from app.models.project import Project
from app.models.source import Source
from app.models.user import User
from app.storage.dependencies import get_storage_service


# ============================================================
# Deterministic Local HTTP Server
# ============================================================


class LocalHTTPServer:
    """
    Small deterministic HTTP server used to test the complete
    URL ingestion application flow without external Internet
    access or DNS resolution.
    """

    def __init__(
        self,
        *,
        body: bytes,
        content_type: str = "text/html; charset=utf-8",
        status_code: int = 200,
    ) -> None:
        self.body = body
        self.content_type = content_type
        self.status_code = status_code

        self.server: asyncio.AbstractServer | None = None
        self.host = "127.0.0.1"
        self.port: int | None = None

        self.requests: list[bytes] = []

    async def start(self) -> None:
        self.server = await asyncio.start_server(
            self._handle_client,
            host=self.host,
            port=0,
        )

        sockets = self.server.sockets

        if not sockets:
            raise RuntimeError(
                "Test HTTP server did not expose a socket."
            )

        address = sockets[0].getsockname()

        self.port = address[1]

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            request = await reader.readuntil(
                b"\r\n\r\n",
            )

            self.requests.append(request)

            response = (
                f"HTTP/1.1 {self.status_code} OK\r\n"
                f"Content-Type: {self.content_type}\r\n"
                f"Content-Length: {len(self.body)}\r\n"
                "Connection: close\r\n"
                "\r\n"
            ).encode() + self.body

            writer.write(response)

            await writer.drain()

        except (
            asyncio.IncompleteReadError,
            ConnectionError,
        ):
            pass

        finally:
            writer.close()

            try:
                await writer.wait_closed()
            except ConnectionError:
                pass


# ============================================================
# Fixed Connection Strategy
# ============================================================


@dataclass
class FixedConnectionStrategy:
    """
    Deterministic connection strategy.

    The requested hostname is intentionally not resolved through
    normal DNS. The strategy returns the local test server address.
    """

    target: ConnectionTarget

    calls: list[tuple[str, int]]

    async def select_target(
        self,
        hostname: str,
        port: int,
    ) -> ConnectionTarget:
        self.calls.append(
            (
                hostname,
                port,
            )
        )

        return self.target


# ============================================================
# Helpers
# ============================================================


def create_local_target(
    *,
    hostname: str,
    port: int,
) -> ConnectionTarget:
    return ConnectionTarget(
        hostname=hostname,
        port=port,
        address=ip_address(
            "127.0.0.1",
        ),
    )


async def create_test_user_and_project():
    user_id = uuid4()
    project_id = uuid4()

    async with SessionFactory() as session:
        user = User(
            id=user_id,
            email=f"url-integration-{user_id}@example.com",
            name="URL Integration Test User",
        )

        project = Project(
            id=project_id,
            name=f"URL Integration Test Project {project_id}",
            owner_id=user_id,
        )

        session.add(user)
        session.add(project)

        await session.commit()

    return user_id, project_id


async def cleanup_test_data(
    *,
    user_id,
    project_id,
    source_id=None,
) -> None:
    async with SessionFactory() as session:
        if source_id is not None:
            source = await session.get(
                Source,
                source_id,
            )

            if source is not None:
                await session.delete(source)

        project = await session.get(
            Project,
            project_id,
        )

        if project is not None:
            await session.delete(project)

        user = await session.get(
            User,
            user_id,
        )

        if user is not None:
            await session.delete(user)

        await session.commit()


# ============================================================
# Complete URL Ingestion Integration Test
# ============================================================


@pytest.mark.asyncio
async def test_url_ingestion_application_e2e_with_real_http_fetcher() -> None:
    """
    Verify the complete URL ingestion flow using the real
    HTTPFetcher and the real IngestionApplicationService.

    Flow:

        Local HTTP Server
                ↓
        Real HTTPFetcher
                ↓
        IngestionApplicationService
                ↓
        HTMLDocumentProcessor
                ↓
        Storage
                ↓
        Database
                ↓
        IngestionResult
    """

    html_content = (
        b"<html>"
        b"<head>"
        b"<title>AI Communication Article</title>"
        b"</head>"
        b"<body>"
        b"<article>"
        b"<h1>Artificial Intelligence</h1>"
        b"<p>"
        b"Artificial intelligence is transforming communication."
        b"</p>"
        b"<p>"
        b"Organizations are using AI to create content faster."
        b"</p>"
        b"</article>"
        b"</body>"
        b"</html>"
    )

    server = LocalHTTPServer(
        body=html_content,
    )

    await server.start()

    assert server.port is not None

    hostname = "url-ingestion.example.test"

    target = create_local_target(
        hostname=hostname,
        port=server.port,
    )

    strategy = FixedConnectionStrategy(
        target=target,
        calls=[],
    )

    fetcher = HTTPFetcher(
        connection_strategy=strategy,
    )

    user_id, project_id = await create_test_user_and_project()

    storage = get_storage_service()

    source_id = uuid4()
    storage_key = None
    result = None

    try:
        url = (
            f"http://{hostname}:{server.port}"
            "/article/ai-communication"
        )

        async with SessionFactory() as session:
            service = IngestionApplicationService(
                session=session,
                storage=storage,
                url_fetcher=fetcher,
            )

            request = service_request = None

            from app.ingestion.schemas import IngestionRequest

            service_request = IngestionRequest(
                project_id=project_id,
                source_id=source_id,
                input_type=InputType.URL,
                title="AI Communication Article",
                url=url,
                metadata={
                    "test_case": "real_http_url_ingestion",
                },
            )

            result = await service.ingest(
                request=service_request,
            )

        # ----------------------------------------------------
        # Ingestion result
        # ----------------------------------------------------

        assert result.source_id == source_id

        assert result.status == ProcessingStatus.COMPLETED

        assert result.storage_key is not None
        assert result.storage_uri is not None
        assert result.content_hash is not None

        storage_key = result.storage_key

        # ----------------------------------------------------
        # Canonical content
        # ----------------------------------------------------

        assert result.canonical_content is not None

        canonical = result.canonical_content

        assert canonical.title == "AI Communication Article"

        assert (
            "Artificial intelligence is transforming communication."
            in canonical.text
        )

        assert (
            "Organizations are using AI to create content faster."
            in canonical.text
        )

        # ----------------------------------------------------
        # Fetch metadata
        # ----------------------------------------------------

        assert canonical.metadata is not None

        assert canonical.metadata["source_url"] == url

        assert canonical.metadata["final_url"] == url

        assert canonical.metadata["http_status_code"] == 200

        assert (
            canonical.metadata["fetched_content_type"]
            == "text/html"
        )

        assert (
            canonical.metadata["test_case"]
            == "real_http_url_ingestion"
        )

        # ----------------------------------------------------
        # HTTPFetcher connection strategy
        # ----------------------------------------------------

        assert strategy.calls == [
            (
                hostname,
                server.port,
            )
        ]

        # ----------------------------------------------------
        # Local HTTP server received the request
        # ----------------------------------------------------

        assert len(server.requests) == 1

        received_request = server.requests[0]

        assert (
            b"GET /article/ai-communication HTTP/1.1"
            in received_request
        )

        assert (
            b"Host: url-ingestion.example.test:"
            in received_request
        )

        # ----------------------------------------------------
        # Database persistence
        # ----------------------------------------------------

        async with SessionFactory() as verify_session:
            source = await verify_session.get(
                Source,
                source_id,
            )

            project = await verify_session.get(
                Project,
                project_id,
            )

            user = await verify_session.get(
                User,
                user_id,
            )

            assert source is not None
            assert project is not None
            assert user is not None

            assert source.project_id == project_id

            assert source.storage_key == storage_key

        # ----------------------------------------------------
        # Stored content
        # ----------------------------------------------------

        stored_content = await storage.download(
            storage_key,
        )

        assert stored_content == html_content

    finally:
        # ----------------------------------------------------
        # Storage cleanup
        # ----------------------------------------------------

        if storage_key is not None:
            await storage.delete(
                storage_key,
            )

        # ----------------------------------------------------
        # Database cleanup
        # ----------------------------------------------------

        await cleanup_test_data(
            user_id=user_id,
            project_id=project_id,
            source_id=source_id,
        )

        # ----------------------------------------------------
        # HTTP server cleanup
        # ----------------------------------------------------

        await server.stop()