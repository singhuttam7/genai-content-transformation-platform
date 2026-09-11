from __future__ import annotations

import asyncio
import socket
import ssl
from typing import Any

import httpcore
import httpx

from app.ingestion.fetchers.connection import ConnectionTarget


# ============================================================
# Pinned Network Backend
# ============================================================


class PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """
    httpcore network backend that connects only to a
    pre-validated IP address.

    The original hostname is preserved separately and is used for:

    - Hostname validation
    - TLS/SNI
    - Certificate verification

    The actual TCP connection is always made directly to
    the validated IP address.
    """

    def __init__(
        self,
        target: ConnectionTarget,
    ) -> None:
        self.target = target

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        """
        Establish a TCP connection using the pre-validated IP.

        HTTPCore provides the hostname and port it wants to connect
        to. Both values must match the security-approved target.

        The hostname is NOT resolved here.
        """

        # ----------------------------------------------------
        # Hostname enforcement
        # ----------------------------------------------------

        if (
            host.casefold()
            != self.target.hostname.casefold()
        ):
            raise ValueError(
                "HTTP transport hostname does not match "
                "the validated connection target."
            )

        # ----------------------------------------------------
        # Port enforcement
        # ----------------------------------------------------

        if port != self.target.port:
            raise ValueError(
                "HTTP transport port does not match "
                "the validated connection target."
            )

        # ----------------------------------------------------
        # Use ONLY the validated IP
        # ----------------------------------------------------

        validated_ip = str(
            self.target.address
        )

        sock = await asyncio.to_thread(
            self._connect_socket,
            validated_ip,
            port,
            timeout,
            local_address,
            socket_options,
        )

        return _SocketNetworkStream(sock)

    # ========================================================
    # Socket Connection
    # ========================================================

    @staticmethod
    def _connect_socket(
        address: str,
        port: int,
        timeout: float | None,
        local_address: str | None,
        socket_options: Any,
    ) -> socket.socket:
        """
        Create and connect a socket directly to the validated IP.

        No hostname resolution occurs here.
        """

        family = (
            socket.AF_INET6
            if ":" in address
            else socket.AF_INET
        )

        sock = socket.socket(
            family,
            socket.SOCK_STREAM,
        )

        try:
            # ------------------------------------------------
            # Socket timeout
            # ------------------------------------------------

            if timeout is not None:
                sock.settimeout(timeout)

            # ------------------------------------------------
            # Socket options
            # ------------------------------------------------

            if socket_options:
                for level, option, value in socket_options:
                    sock.setsockopt(
                        level,
                        option,
                        value,
                    )

            # ------------------------------------------------
            # Local bind
            # ------------------------------------------------

            if local_address is not None:
                if family == socket.AF_INET6:
                    sock.bind(
                        (
                            local_address,
                            0,
                            0,
                            0,
                        )
                    )
                else:
                    sock.bind(
                        (
                            local_address,
                            0,
                        )
                    )

            # ------------------------------------------------
            # Direct IP connection
            # ------------------------------------------------

            if family == socket.AF_INET6:
                sock.connect(
                    (
                        address,
                        port,
                        0,
                        0,
                    )
                )
            else:
                sock.connect(
                    (
                        address,
                        port,
                    )
                )

            # HTTPCore expects an asynchronous stream.
            sock.setblocking(False)

            return sock

        except Exception:
            sock.close()
            raise


# ============================================================
# Socket Network Stream
# ============================================================


class _SocketNetworkStream(
    httpcore.AsyncNetworkStream
):
    """
    httpcore stream backed by a standard Python socket.
    """

    def __init__(
        self,
        sock: socket.socket,
    ) -> None:
        self._socket = sock

    # ========================================================
    # Read
    # ========================================================

    async def read(
        self,
        max_bytes: int,
        timeout: float | None = None,
    ) -> bytes:
        """
        Read data asynchronously from the socket.
        """

        loop = asyncio.get_running_loop()

        operation = loop.sock_recv(
            self._socket,
            max_bytes,
        )

        if timeout is None:
            return await operation

        return await asyncio.wait_for(
            operation,
            timeout,
        )

    # ========================================================
    # Write
    # ========================================================

    async def write(
        self,
        buffer: bytes,
        timeout: float | None = None,
    ) -> None:
        """
        Write data asynchronously to the socket.
        """

        loop = asyncio.get_running_loop()

        operation = loop.sock_sendall(
            self._socket,
            buffer,
        )

        if timeout is None:
            await operation
            return

        await asyncio.wait_for(
            operation,
            timeout,
        )

    # ========================================================
    # Close
    # ========================================================

    async def aclose(self) -> None:
        """
        Close the underlying socket.
        """

        self._socket.close()

    # ========================================================
    # Connection Information
    # ========================================================

    def get_extra_info(
        self,
        info: str,
    ) -> Any:
        """
        Expose connection information expected by httpcore.
        """

        if info == "socket":
            return self._socket

        if info == "server_address":
            return self._socket.getpeername()

        if info == "client_address":
            return self._socket.getsockname()

        return None

    # ========================================================
    # TLS
    # ========================================================

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        """
        Upgrade the existing TCP connection to TLS.

        The TCP connection remains pinned to the validated IP.

        The original hostname is supplied as server_hostname so that:

        - TLS SNI uses the original hostname.
        - Certificate verification uses the original hostname.
        - The validated IP remains the TCP destination.
        """

        if server_hostname is None:
            raise ValueError(
                "TLS server hostname is required."
            )

        raw_socket = self._socket

        def wrap_socket() -> ssl.SSLSocket:
            """
            Perform the synchronous TLS handshake inside
            a worker thread so the asyncio event loop is
            not blocked.
            """

            raw_socket.setblocking(True)

            if timeout is not None:
                raw_socket.settimeout(timeout)

            tls_socket = ssl_context.wrap_socket(
                raw_socket,
                server_hostname=server_hostname,
            )

            # Return the socket to async/non-blocking mode.
            tls_socket.settimeout(None)
            tls_socket.setblocking(False)

            return tls_socket

        try:
            if timeout is None:
                tls_socket = await asyncio.to_thread(
                    wrap_socket
                )
            else:
                tls_socket = await asyncio.wait_for(
                    asyncio.to_thread(
                        wrap_socket
                    ),
                    timeout,
                )

            self._socket = tls_socket

            return self

        except Exception:
            raw_socket.close()
            raise


# ============================================================
# Pinned HTTPX Transport
# ============================================================


class PinnedHTTPTransport(
    httpx.AsyncBaseTransport
):
    """
    HTTPX transport that uses a pre-validated
    ConnectionTarget.

    Security flow:

        URL
         ↓
        URLSecurityValidator
         ↓
        SecureDNSResolver
         ↓
        SecureConnectionStrategy
         ↓
        ConnectionTarget
         ↓
        PinnedHTTPTransport
         ↓
        PinnedNetworkBackend
         ↓
        validated IP
         ↓
        TCP connection

    HTTPX/HTTPCore therefore does not perform an independent
    DNS lookup for the destination.
    """

    def __init__(
        self,
        target: ConnectionTarget,
    ) -> None:
        self.target = target

        self._network_backend = (
            PinnedNetworkBackend(
                target
            )
        )

        self._pool = (
            httpcore.AsyncConnectionPool(
                network_backend=(
                    self._network_backend
                ),
            )
        )

    # ========================================================
    # HTTP Request
    # ========================================================

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        """
        Convert an HTTPX request into an HTTPCore request,
        execute it through the pinned network backend,
        and convert the response back to HTTPX.
        """

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # httpcore 1.0.9 accepts:
        #
        #   URL
        #   bytes
        #   str
        #
        # It does NOT accept the older tuple representation.
        #
        # Therefore we pass the complete HTTPX URL as a string.
        # ----------------------------------------------------

        httpcore_request = httpcore.Request(
            method=request.method,
            url=str(request.url),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )

        # ----------------------------------------------------
        # Execute through HTTPCore.
        #
        # HTTPCore will eventually call:
        #
        # PinnedNetworkBackend.connect_tcp()
        #
        # which connects directly to target.address.
        # ----------------------------------------------------

        response = (
            await self._pool.handle_async_request(
                httpcore_request
            )
        )

        # ----------------------------------------------------
        # Convert HTTPCore response into HTTPX response.
        # ----------------------------------------------------

        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_HTTPcoreAsyncByteStream(
                response.stream
            ),
            extensions=response.extensions,
            request=request,
        )

    # ========================================================
    # Close
    # ========================================================

    async def aclose(self) -> None:
        """
        Close the underlying HTTPCore connection pool.
        """

        await self._pool.aclose()


# ============================================================
# HTTPCore → HTTPX Stream Adapter
# ============================================================


class _HTTPcoreAsyncByteStream(
    httpx.AsyncByteStream
):
    """
    Adapter that exposes an HTTPCore async response stream
    through the HTTPX AsyncByteStream interface.
    """

    def __init__(
        self,
        stream: httpcore.AsyncIterableByteStream,
    ) -> None:
        self._stream = stream

    async def __aiter__(self):
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()