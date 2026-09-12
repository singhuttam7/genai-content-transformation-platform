from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass
from ipaddress import ip_address

import pytest

from app.ingestion.fetchers.connection import (
    ConnectionStrategy,
    ConnectionTarget,
)
from app.ingestion.fetchers.http import (
    HTTPFetcher,
)


# ============================================================
# Test Connection Strategy
# ============================================================


@dataclass
class FixedConnectionStrategy:
    """
    Deterministic connection strategy for integration tests.

    It deliberately returns a pre-selected IP address instead
    of performing DNS resolution.

    This allows the test to prove that HTTPFetcher actually
    uses the validated ConnectionTarget.
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
# Local HTTP Server
# ============================================================


class LocalHTTPServer:
    """
    Small deterministic HTTP server used only for tests.

    The server listens on localhost.

    The client uses a fake hostname such as example.test,
    while the pinned transport connects directly to the
    server's localhost IP.

    Therefore, a successful request proves that the transport
    is using the validated IP rather than performing DNS.
    """

    def __init__(
        self,
        *,
        status_code: int = 200,
        content_type: str = "text/html; charset=utf-8",
        body: bytes = b"<html><body>Hello</body></html>",
        redirect_location: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.content_type = content_type
        self.body = body
        self.redirect_location = (
            redirect_location
        )

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
                b"\r\n\r\n"
            )

            self.requests.append(request)

            if self.redirect_location is not None:
                response = (
                    "HTTP/1.1 302 Found\r\n"
                    f"Location: {self.redirect_location}\r\n"
                    "Content-Length: 0\r\n"
                    "Connection: close\r\n"
                    "\r\n"
                ).encode()

            else:
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
# Helpers
# ============================================================


def create_local_target(
    *,
    hostname: str,
    port: int,
) -> ConnectionTarget:
    """
    Create a ConnectionTarget pointing to the local
    deterministic test server.
    """

    return ConnectionTarget(
        hostname=hostname,
        port=port,
        address=ip_address(
            "127.0.0.1"
        ),
    )


# ============================================================
# Basic End-to-End Test
# ============================================================


@pytest.mark.asyncio
async def test_http_fetcher_uses_validated_ip_end_to_end() -> None:
    """
    Prove that HTTPFetcher can successfully communicate with
    a server through the IP returned by ConnectionStrategy.

    The requested hostname is example.test.

    example.test is NOT resolved by the operating system.

    The injected ConnectionStrategy returns:

        hostname = example.test
        address  = 127.0.0.1

    A successful response therefore proves that the actual
    network connection uses the validated IP.
    """

    server = LocalHTTPServer(
        body=b"<html><body>Integration Test</body></html>",
    )

    await server.start()

    assert server.port is not None

    hostname = "example.test"

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

    url = (
        f"http://{hostname}:{server.port}/"
    )

    try:
        result = await fetcher.fetch(url)

        assert result.url == url

        assert result.final_url == url

        assert result.status_code == 200

        assert (
            result.content_type
            == "text/html"
        )

        assert (
            result.content
            == b"<html><body>Integration Test</body></html>"
        )

        assert strategy.calls == [
            (
                hostname,
                server.port,
            )
        ]

        assert len(server.requests) == 1

        request = server.requests[0]

        assert (
            b"GET / HTTP/1.1"
            in request
        )

        assert (
            b"Host: example.test:"
            in request
        )

    finally:
        await server.stop()


# ============================================================
# Path Preservation
# ============================================================


@pytest.mark.asyncio
async def test_http_fetcher_preserves_request_path() -> None:
    """
    Verify that the original URL path reaches the local server
    while the TCP connection still uses the validated IP.
    """

    server = LocalHTTPServer(
        body=b"path-test",
    )

    await server.start()

    assert server.port is not None

    hostname = "example.test"

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

    url = (
        f"http://{hostname}:{server.port}"
        "/article/security/advisory"
        "?page=1"
    )

    try:
        result = await fetcher.fetch(url)

        assert result.content == b"path-test"

        assert len(server.requests) == 1

        request = server.requests[0]

        assert (
            b"GET /article/security/advisory?page=1 HTTP/1.1"
            in request
        )

    finally:
        await server.stop()


# ============================================================
# Connection Strategy Boundary
# ============================================================


@pytest.mark.asyncio
async def test_http_fetcher_passes_exact_hostname_and_port_to_strategy() -> None:
    """
    Verify that HTTPFetcher extracts the hostname and effective
    port correctly before constructing the pinned transport.
    """

    server = LocalHTTPServer(
        body=b"boundary-test",
    )

    await server.start()

    assert server.port is not None

    hostname = "example.test"

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

    url = (
        f"http://{hostname}:{server.port}/"
    )

    try:
        await fetcher.fetch(url)

        assert strategy.calls == [
            (
                hostname,
                server.port,
            )
        ]

    finally:
        await server.stop()


# ============================================================
# Content-Type Validation
# ============================================================


@pytest.mark.asyncio
async def test_http_fetcher_rejects_unsupported_content_type() -> None:
    """
    Verify that the response Content-Type security policy
    remains active after pinned transport integration.
    """

    server = LocalHTTPServer(
        content_type="application/octet-stream",
        body=b"binary-content",
    )

    await server.start()

    assert server.port is not None

    hostname = "example.test"

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

    url = (
        f"http://{hostname}:{server.port}/"
    )

    try:
        with pytest.raises(
            ValueError,
            match="Unsupported response Content-Type",
        ):
            await fetcher.fetch(url)

    finally:
        await server.stop()


# ============================================================
# HTTP Status Validation
# ============================================================


@pytest.mark.asyncio
async def test_http_fetcher_rejects_error_status() -> None:
    """
    Verify that HTTP error status validation remains active.
    """

    server = LocalHTTPServer(
        status_code=404,
        body=b"not-found",
    )

    await server.start()

    assert server.port is not None

    hostname = "example.test"

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

    url = (
        f"http://{hostname}:{server.port}/"
    )

    try:
        with pytest.raises(
            ValueError,
            match="HTTP request failed with status 404",
        ):
            await fetcher.fetch(url)

    finally:
        await server.stop()


# ============================================================
# Redirect Re-validation
# ============================================================


@pytest.mark.asyncio
async def test_redirect_destination_is_validated_again() -> None:
    """
    Verify that a redirect causes HTTPFetcher to request a
    fresh ConnectionTarget.

    Two different hostnames are used:

        example.test
        second.example.test

    Both are pinned to the same deterministic local server.
    """

    first_server = LocalHTTPServer(
        redirect_location=(
            "http://second.example.test:"
            "{PORT}/final"
        ),
    )

    await first_server.start()

    assert first_server.port is not None

    second_server = LocalHTTPServer(
        body=b"redirect-final",
    )

    await second_server.start()

    assert second_server.port is not None

    first_redirect = (
        "http://second.example.test:"
        f"{second_server.port}/final"
    )

    # Reconfigure the first server's redirect target.
    first_server.redirect_location = (
        first_redirect
    )

    first_hostname = "example.test"
    second_hostname = "second.example.test"

    first_target = create_local_target(
        hostname=first_hostname,
        port=first_server.port,
    )

    second_target = create_local_target(
        hostname=second_hostname,
        port=second_server.port,
    )

    class RedirectAwareStrategy:
        def __init__(self) -> None:
            self.calls: list[
                tuple[str, int]
            ] = []

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

            if hostname == first_hostname:
                return first_target

            if hostname == second_hostname:
                return second_target

            raise ValueError(
                f"Unexpected hostname: {hostname}"
            )

    strategy = RedirectAwareStrategy()

    fetcher = HTTPFetcher(
        connection_strategy=strategy,
    )

    url = (
        f"http://{first_hostname}:"
        f"{first_server.port}/"
    )

    try:
        result = await fetcher.fetch(url)

        assert (
            result.content
            == b"redirect-final"
        )

        assert result.final_url == (
            f"http://{second_hostname}:"
            f"{second_server.port}/final"
        )

        assert strategy.calls == [
            (
                first_hostname,
                first_server.port,
            ),
            (
                second_hostname,
                second_server.port,
            ),
        ]

        assert len(
            first_server.requests
        ) == 1

        assert len(
            second_server.requests
        ) == 1

        assert (
            b"GET / HTTP/1.1"
            in first_server.requests[0]
        )

        assert (
            b"GET /final HTTP/1.1"
            in second_server.requests[0]
        )

    finally:
        await first_server.stop()
        await second_server.stop()


# ============================================================
# Redirect Security
# ============================================================


@pytest.mark.asyncio
async def test_redirect_to_unexpected_destination_is_rejected() -> None:
    """
    Verify that a redirect destination must be accepted by the
    connection strategy before the second request is performed.
    """

    server = LocalHTTPServer(
        redirect_location=(
            "http://attacker.example/"
        ),
    )

    await server.start()

    assert server.port is not None

    hostname = "example.test"

    target = create_local_target(
        hostname=hostname,
        port=server.port,
    )

    class RestrictiveStrategy:
        def __init__(self) -> None:
            self.calls: list[
                tuple[str, int]
            ] = []

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

            if hostname != "example.test":
                raise ValueError(
                    "Destination rejected by test security policy."
                )

            return target

    strategy = RestrictiveStrategy()

    fetcher = HTTPFetcher(
        connection_strategy=strategy,
    )

    url = (
        f"http://{hostname}:{server.port}/"
    )

    try:
        with pytest.raises(
            ValueError,
            match="Destination rejected",
        ):
            await fetcher.fetch(url)

        assert strategy.calls == [
            (
                "example.test",
                server.port,
            ),
            (
                "attacker.example",
                80,
            ),
        ]

        assert len(
            server.requests
        ) == 1

    finally:
        await server.stop()


# ============================================================
# No DNS Requirement
# ============================================================


@pytest.mark.asyncio
async def test_fetch_succeeds_without_dns_resolution() -> None:
    """
    Strong integration proof:

    The hostname is intentionally non-resolvable for this test.
    The connection strategy supplies 127.0.0.1.

    If HTTPX/HTTPCore attempted its own DNS lookup, this test
    would fail before reaching the local server.
    """

    server = LocalHTTPServer(
        body=b"no-dns-required",
    )

    await server.start()

    assert server.port is not None

    hostname = (
        "this-host-should-not-be-dns-resolved.example"
    )

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

    url = (
        f"http://{hostname}:{server.port}/"
    )

    try:
        result = await fetcher.fetch(url)

        assert (
            result.content
            == b"no-dns-required"
        )

        assert strategy.calls == [
            (
                hostname,
                server.port,
            )
        ]

        assert len(server.requests) == 1

    finally:
        await server.stop()