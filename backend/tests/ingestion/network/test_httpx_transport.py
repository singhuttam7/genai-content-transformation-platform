from __future__ import annotations

import asyncio
import socket
import ssl

import pytest

from app.ingestion.fetchers.connection import (
    ConnectionTarget,
)
from app.ingestion.fetchers.transport import (
    PinnedNetworkBackend,
    PinnedHTTPTransport,
    _SocketNetworkStream,
)


def create_target() -> ConnectionTarget:
    """
    Create a deterministic validated connection target
    for transport-layer tests.
    """

    return ConnectionTarget(
        hostname="example.com",
        port=80,
        address=__import__(
            "ipaddress"
        ).ip_address(
            "93.184.216.34"
        ),
    )


class FakeSocket:
    """
    Minimal socket implementation used to test
    PinnedNetworkBackend without creating a real
    outbound connection.
    """

    def __init__(self) -> None:
        self.connected_address = None
        self.closed = False
        self.blocking = None
        self.timeout = None
        self.options = []

    def settimeout(
        self,
        value: float | None,
    ) -> None:
        self.timeout = value

    def setsockopt(
        self,
        level,
        option,
        value,
    ) -> None:
        self.options.append(
            (
                level,
                option,
                value,
            )
        )

    def bind(self, address) -> None:
        self.bound_address = address

    def connect(self, address) -> None:
        self.connected_address = address

    def setblocking(
        self,
        value: bool,
    ) -> None:
        self.blocking = value

    def close(self) -> None:
        self.closed = True

    def getpeername(self):
        return self.connected_address

    def getsockname(self):
        return (
            "127.0.0.1",
            12345,
        )


@pytest.mark.asyncio
async def test_pinned_backend_uses_validated_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = create_target()

    fake_socket = FakeSocket()

    def fake_socket_factory(
        family: int,
        socket_type: int,
    ) -> FakeSocket:
        assert family == socket.AF_INET
        assert (
            socket_type
            == socket.SOCK_STREAM
        )

        return fake_socket

    monkeypatch.setattr(
        "app.ingestion.fetchers.transport.socket.socket",
        fake_socket_factory,
    )

    backend = PinnedNetworkBackend(target)

    stream = await backend.connect_tcp(
        target.hostname,
        target.port,
    )

    assert isinstance(
        stream,
        _SocketNetworkStream,
    )

    assert (
        fake_socket.connected_address
        == (
            str(target.address),
            target.port,
        )
    )

    assert fake_socket.blocking is False

    await stream.aclose()

    assert fake_socket.closed is True


@pytest.mark.asyncio
async def test_pinned_backend_rejects_wrong_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = create_target()

    fake_socket = FakeSocket()

    def fake_socket_factory(
        family: int,
        socket_type: int,
    ) -> FakeSocket:
        return fake_socket

    monkeypatch.setattr(
        "app.ingestion.fetchers.transport.socket.socket",
        fake_socket_factory,
    )

    backend = PinnedNetworkBackend(target)

    with pytest.raises(
        ValueError,
        match="hostname does not match",
    ):
        await backend.connect_tcp(
            "attacker-controlled-host.example",
            target.port,
        )

    assert (
        fake_socket.connected_address
        is None
    )


@pytest.mark.asyncio
async def test_pinned_backend_rejects_wrong_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = create_target()

    fake_socket = FakeSocket()

    def fake_socket_factory(
        family: int,
        socket_type: int,
    ) -> FakeSocket:
        return fake_socket

    monkeypatch.setattr(
        "app.ingestion.fetchers.transport.socket.socket",
        fake_socket_factory,
    )

    backend = PinnedNetworkBackend(target)

    with pytest.raises(
        ValueError,
        match="port does not match",
    ):
        await backend.connect_tcp(
            target.hostname,
            443,
        )

    assert (
        fake_socket.connected_address
        is None
    )


@pytest.mark.asyncio
async def test_pinned_backend_accepts_case_insensitive_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = create_target()

    fake_socket = FakeSocket()

    def fake_socket_factory(
        family: int,
        socket_type: int,
    ) -> FakeSocket:
        return fake_socket

    monkeypatch.setattr(
        "app.ingestion.fetchers.transport.socket.socket",
        fake_socket_factory,
    )

    backend = PinnedNetworkBackend(target)

    stream = await backend.connect_tcp(
        "EXAMPLE.COM",
        target.port,
    )

    assert (
        fake_socket.connected_address
        == (
            str(target.address),
            target.port,
        )
    )

    await stream.aclose()


@pytest.mark.asyncio
async def test_pinned_backend_passes_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = create_target()

    fake_socket = FakeSocket()

    def fake_socket_factory(
        family: int,
        socket_type: int,
    ) -> FakeSocket:
        return fake_socket

    monkeypatch.setattr(
        "app.ingestion.fetchers.transport.socket.socket",
        fake_socket_factory,
    )

    backend = PinnedNetworkBackend(target)

    stream = await backend.connect_tcp(
        target.hostname,
        target.port,
        timeout=5.0,
    )

    assert fake_socket.timeout == 5.0

    await stream.aclose()


@pytest.mark.asyncio
async def test_pinned_backend_passes_socket_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = create_target()

    fake_socket = FakeSocket()

    def fake_socket_factory(
        family: int,
        socket_type: int,
    ) -> FakeSocket:
        return fake_socket

    monkeypatch.setattr(
        "app.ingestion.fetchers.transport.socket.socket",
        fake_socket_factory,
    )

    backend = PinnedNetworkBackend(target)

    socket_options = [
        (
            socket.SOL_SOCKET,
            socket.SO_KEEPALIVE,
            1,
        )
    ]

    stream = await backend.connect_tcp(
        target.hostname,
        target.port,
        socket_options=socket_options,
    )

    assert (
        fake_socket.options
        == socket_options
    )

    await stream.aclose()


@pytest.mark.asyncio
async def test_pinned_backend_supports_ipv6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = ConnectionTarget(
        hostname="example.com",
        port=80,
        address=__import__(
            "ipaddress"
        ).ip_address(
            "2606:4700:4700::1111"
        ),
    )

    fake_socket = FakeSocket()

    def fake_socket_factory(
        family: int,
        socket_type: int,
    ) -> FakeSocket:
        assert family == socket.AF_INET6
        assert (
            socket_type
            == socket.SOCK_STREAM
        )

        return fake_socket

    monkeypatch.setattr(
        "app.ingestion.fetchers.transport.socket.socket",
        fake_socket_factory,
    )

    backend = PinnedNetworkBackend(target)

    stream = await backend.connect_tcp(
        target.hostname,
        target.port,
    )

    assert (
        fake_socket.connected_address
        == (
            str(target.address),
            target.port,
            0,
            0,
        )
    )

    await stream.aclose()


@pytest.mark.asyncio
async def test_pinned_backend_closes_socket_on_connect_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = create_target()

    fake_socket = FakeSocket()

    def fake_socket_factory(
        family: int,
        socket_type: int,
    ) -> FakeSocket:
        return fake_socket

    def failing_connect(address) -> None:
        raise OSError(
            "simulated connection failure"
        )

    fake_socket.connect = failing_connect

    monkeypatch.setattr(
        "app.ingestion.fetchers.transport.socket.socket",
        fake_socket_factory,
    )

    backend = PinnedNetworkBackend(target)

    with pytest.raises(
        OSError,
        match="simulated connection failure",
    ):
        await backend.connect_tcp(
            target.hostname,
            target.port,
        )

    assert fake_socket.closed is True


@pytest.mark.asyncio
async def test_tls_passes_original_hostname_to_ssl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_socket, server_socket = (
        socket.socketpair()
    )

    stream = _SocketNetworkStream(
        client_socket
    )

    context = ssl.create_default_context()

    captured: dict[str, object] = {}

    class FakeSSLSocket:
        def setblocking(
            self,
            value: bool,
        ) -> None:
            captured["blocking"] = value

        def settimeout(
            self,
            value: float | None,
        ) -> None:
            captured["timeout"] = value

        def close(self) -> None:
            captured["closed"] = True

    def fake_wrap_socket(
        raw_socket: socket.socket,
        *,
        server_hostname: str,
    ) -> FakeSSLSocket:
        captured["raw_socket"] = raw_socket
        captured["server_hostname"] = (
            server_hostname
        )

        return FakeSSLSocket()

    monkeypatch.setattr(
        context,
        "wrap_socket",
        fake_wrap_socket,
    )

    try:
        result = await stream.start_tls(
            context,
            server_hostname="example.com",
        )

        assert result is stream

        assert (
            captured["raw_socket"]
            is client_socket
        )

        assert (
            captured["server_hostname"]
            == "example.com"
        )

        assert captured["blocking"] is False
        assert captured["timeout"] is None

    finally:
        await stream.aclose()
        server_socket.close()


@pytest.mark.asyncio
async def test_tls_failure_closes_raw_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_socket, server_socket = (
        socket.socketpair()
    )

    stream = _SocketNetworkStream(
        client_socket
    )

    context = ssl.create_default_context()

    captured: dict[str, object] = {}

    def fake_wrap_socket(
        raw_socket: socket.socket,
        *,
        server_hostname: str,
    ):
        captured["raw_socket"] = raw_socket

        raise ssl.SSLError(
            "simulated TLS failure"
        )

    monkeypatch.setattr(
        context,
        "wrap_socket",
        fake_wrap_socket,
    )

    try:
        with pytest.raises(ssl.SSLError):
            await stream.start_tls(
                context,
                server_hostname="example.com",
            )

        assert (
            captured["raw_socket"]
            is client_socket
        )

        with pytest.raises(OSError):
            client_socket.getpeername()

    finally:
        server_socket.close()


@pytest.mark.asyncio
async def test_pinned_http_transport_uses_target() -> None:
    target = create_target()

    transport = PinnedHTTPTransport(
        target
    )

    try:
        assert transport.target is target

        assert isinstance(
            transport._network_backend,
            PinnedNetworkBackend,
        )

        assert (
            transport._network_backend.target
            is target
        )

    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_pinned_http_transport_has_connection_pool() -> None:
    target = create_target()

    transport = PinnedHTTPTransport(
        target
    )

    try:
        assert transport._pool is not None

        assert (
            transport._network_backend
            is not None
        )

    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_tls_stream_exposes_ssl_socket_compatibly() -> None:
    """
    Verify that the TLS stream exposes its socket through
    get_extra_info("socket") without losing the underlying
    socket object.

    This regression test protects the HTTPX/HTTPCore transport
    integration from the production error:

        Socket cannot be of type SSLSocket
    """

    client_socket, server_socket = socket.socketpair()

    stream = _SocketNetworkStream(
        client_socket,
    )

    try:
        captured: dict[str, object] = {}

        class FakeSSLSocket:
            def setblocking(
                self,
                value: bool,
            ) -> None:
                captured["blocking"] = value

            def settimeout(
                self,
                value: float | None,
            ) -> None:
                captured["timeout"] = value

            def close(self) -> None:
                captured["closed"] = True

        context = ssl.create_default_context()

        def fake_wrap_socket(
            raw_socket: socket.socket,
            *,
            server_hostname: str,
        ) -> FakeSSLSocket:
            captured["raw_socket"] = raw_socket
            captured["server_hostname"] = (
                server_hostname
            )

            return FakeSSLSocket()

        context.wrap_socket = fake_wrap_socket  # type: ignore[method-assign]

        result = await stream.start_tls(
            context,
            server_hostname="example.com",
        )

        assert result is stream

        assert (
            captured["raw_socket"]
            is client_socket
        )

        assert (
            captured["server_hostname"]
            == "example.com"
        )

        assert captured["blocking"] is False
        assert captured["timeout"] is None

        exposed_socket = stream.get_extra_info(
            "socket",
        )

        assert exposed_socket is stream._socket

        assert exposed_socket is not None

    finally:
        await stream.aclose()
        server_socket.close()