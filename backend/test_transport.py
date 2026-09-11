from __future__ import annotations

import asyncio
import socket
import ssl

import pytest

from app.ingestion.fetchers.transport import (
    _SocketNetworkStream,
)


def create_socket_pair() -> tuple[socket.socket, socket.socket]:
    """
    Create a connected socket pair.

    Windows does not provide socket.socketpair() consistently across
    all environments, so use a local TCP connection instead.
    """

    server = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    server.bind(("127.0.0.1", 0))
    server.listen(1)

    host, port = server.getsockname()

    client = socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    )

    client.connect((host, port))

    server_socket, _ = server.accept()

    server.close()

    return client, server_socket


@pytest.mark.asyncio
async def test_read_reads_data_from_socket() -> None:
    client_socket, server_socket = create_socket_pair()

    stream = _SocketNetworkStream(client_socket)

    try:
        server_socket.sendall(b"hello")

        result = await stream.read(1024)

        assert result == b"hello"

    finally:
        await stream.aclose()
        server_socket.close()


@pytest.mark.asyncio
async def test_write_writes_data_to_socket() -> None:
    client_socket, server_socket = create_socket_pair()

    stream = _SocketNetworkStream(client_socket)

    try:
        await stream.write(b"hello")

        result = await asyncio.to_thread(
            server_socket.recv,
            1024,
        )

        assert result == b"hello"

    finally:
        await stream.aclose()
        server_socket.close()


@pytest.mark.asyncio
async def test_read_supports_timeout() -> None:
    client_socket, server_socket = create_socket_pair()

    stream = _SocketNetworkStream(client_socket)

    try:
        with pytest.raises(asyncio.TimeoutError):
            await stream.read(
                1024,
                timeout=0.01,
            )

    finally:
        await stream.aclose()
        server_socket.close()


@pytest.mark.asyncio
async def test_write_supports_timeout() -> None:
    client_socket, server_socket = create_socket_pair()

    stream = _SocketNetworkStream(client_socket)

    try:
        await stream.write(
            b"hello",
            timeout=1.0,
        )

        result = await asyncio.to_thread(
            server_socket.recv,
            1024,
        )

        assert result == b"hello"

    finally:
        await stream.aclose()
        server_socket.close()


@pytest.mark.asyncio
async def test_get_extra_info() -> None:
    client_socket, server_socket = create_socket_pair()

    stream = _SocketNetworkStream(client_socket)

    try:
        assert stream.get_extra_info("socket") is client_socket

        server_address = stream.get_extra_info(
            "server_address"
        )

        client_address = stream.get_extra_info(
            "client_address"
        )

        assert server_address is not None
        assert client_address is not None

        assert (
            stream.get_extra_info("unknown")
            is None
        )

    finally:
        await stream.aclose()
        server_socket.close()


@pytest.mark.asyncio
async def test_aclose_closes_socket() -> None:
    client_socket, server_socket = create_socket_pair()

    stream = _SocketNetworkStream(client_socket)

    await stream.aclose()

    try:
        with pytest.raises(OSError):
            client_socket.getpeername()
    finally:
        server_socket.close()


@pytest.mark.asyncio
async def test_tls_requires_hostname() -> None:
    client_socket, server_socket = create_socket_pair()

    stream = _SocketNetworkStream(client_socket)

    context = ssl.create_default_context()

    try:
        with pytest.raises(
            ValueError,
            match="TLS server hostname is required",
        ):
            await stream.start_tls(
                context,
                server_hostname=None,
            )

    finally:
        await stream.aclose()
        server_socket.close()


@pytest.mark.asyncio
async def test_tls_passes_original_hostname_to_ssl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_socket, server_socket = create_socket_pair()

    stream = _SocketNetworkStream(client_socket)

    context = ssl.create_default_context()

    captured: dict[str, object] = {}

    class FakeSSLSocket:
        def setblocking(self, value: bool) -> None:
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
        captured["server_hostname"] = server_hostname

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
    client_socket, server_socket = create_socket_pair()

    stream = _SocketNetworkStream(client_socket)

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