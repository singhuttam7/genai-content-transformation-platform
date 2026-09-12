from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.ingestion.fetchers.connection import ConnectionTarget
from app.ingestion.fetchers.http import HTTPFetcher


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def fetcher() -> HTTPFetcher:
    """Create a default HTTP fetcher."""
    return HTTPFetcher()


@pytest.fixture
def mock_connection_strategy() -> AsyncMock:
    """Create a deterministic connection strategy mock."""
    strategy = AsyncMock()

    strategy.select_target.return_value = ConnectionTarget(
        hostname="example.com",
        port=443,
        address=__import__("ipaddress").ip_address(
            "93.184.216.34"
        ),
    )

    return strategy


# ============================================================
# Test Helpers
# ============================================================


def make_response(
    *,
    status_code: int = 200,
    content: bytes = b"<html><body>Hello</body></html>",
    content_type: str = "text/html",
    headers: dict[str, str] | None = None,
    url: str = "https://example.com/",
) -> httpx.Response:
    """Create a deterministic HTTP response."""

    response_headers = {
        "content-type": content_type,
    }

    if headers:
        response_headers.update(headers)

    return httpx.Response(
        status_code=status_code,
        headers=response_headers,
        content=content,
        request=httpx.Request(
            "GET",
            url,
        ),
    )


def create_mock_client(
    responses: list[httpx.Response],
) -> tuple[MagicMock, MagicMock]:
    """
    Create a mocked httpx AsyncClient and its
    asynchronous context manager.
    """

    mock_client = MagicMock()

    mock_client.get = AsyncMock(
        side_effect=responses,
    )

    mock_client_cm = MagicMock()

    mock_client_cm.__aenter__ = AsyncMock(
        return_value=mock_client,
    )

    mock_client_cm.__aexit__ = AsyncMock(
        return_value=None,
    )

    return mock_client, mock_client_cm


# ============================================================
# Successful Requests
# ============================================================


@pytest.mark.asyncio
async def test_successful_html_fetch(
    fetcher: HTTPFetcher,
) -> None:
    """A valid HTML response should be returned."""

    response = make_response()

    mock_client, mock_client_cm = create_mock_client(
        [response],
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ),
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        result = await fetcher.fetch(
            "https://example.com",
        )

    assert result.status_code == 200

    assert result.content_type == "text/html"

    assert result.content == (
        b"<html><body>Hello</body></html>"
    )

    assert result.final_url == (
        "https://example.com/"
    )

    assert result.url == "https://example.com"

    mock_client.get.assert_awaited_once_with(
        "https://example.com/",
    )


# ============================================================
# HTTP Status Validation
# ============================================================


@pytest.mark.asyncio
async def test_http_404_is_rejected(
    fetcher: HTTPFetcher,
) -> None:
    """HTTP 404 responses should be rejected."""

    response = make_response(
        status_code=404,
    )

    _, mock_client_cm = create_mock_client(
        [response],
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ),
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="HTTP request failed with status 404",
        ):
            await fetcher.fetch(
                "https://example.com",
            )


@pytest.mark.asyncio
async def test_http_500_is_rejected(
    fetcher: HTTPFetcher,
) -> None:
    """HTTP 500 responses should be rejected."""

    response = make_response(
        status_code=500,
    )

    _, mock_client_cm = create_mock_client(
        [response],
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ),
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="HTTP request failed with status 500",
        ):
            await fetcher.fetch(
                "https://example.com",
            )


# ============================================================
# Content-Type Validation
# ============================================================


@pytest.mark.asyncio
async def test_unsupported_content_type_is_rejected(
    fetcher: HTTPFetcher,
) -> None:
    """Unsupported response media types should be rejected."""

    response = make_response(
        content_type="application/json",
    )

    _, mock_client_cm = create_mock_client(
        [response],
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ),
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Unsupported response Content-Type",
        ):
            await fetcher.fetch(
                "https://example.com",
            )


@pytest.mark.asyncio
async def test_missing_content_type_is_rejected(
    fetcher: HTTPFetcher,
) -> None:
    """Responses without Content-Type should be rejected."""

    response = httpx.Response(
        status_code=200,
        content=b"hello",
        request=httpx.Request(
            "GET",
            "https://example.com",
        ),
    )

    _, mock_client_cm = create_mock_client(
        [response],
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ),
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Response is missing Content-Type",
        ):
            await fetcher.fetch(
                "https://example.com",
            )


@pytest.mark.asyncio
async def test_content_type_parameters_are_normalized(
    fetcher: HTTPFetcher,
) -> None:
    """
    Content-Type parameters such as charset should be
    stripped before validation.
    """

    response = make_response(
        content_type="text/html; charset=utf-8",
    )

    _, mock_client_cm = create_mock_client(
        [response],
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ),
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        result = await fetcher.fetch(
            "https://example.com",
        )

    assert result.content_type == "text/html"


# ============================================================
# Redirect Handling
# ============================================================


@pytest.mark.asyncio
async def test_redirect_is_followed(
    fetcher: HTTPFetcher,
) -> None:
    """Safe absolute redirects should be followed."""

    redirect_response = make_response(
        status_code=302,
        headers={
            "location": "https://example.org/final",
        },
        url="https://example.com/",
    )

    final_response = make_response(
        content=b"<html>Final</html>",
        url="https://example.org/final",
    )

    mock_client, mock_client_cm = create_mock_client(
        [
            redirect_response,
            final_response,
        ],
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ),
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        result = await fetcher.fetch(
            "https://example.com",
        )

    assert result.final_url == (
        "https://example.org/final"
    )

    assert result.content == (
        b"<html>Final</html>"
    )

    assert mock_client.get.await_count == 2


@pytest.mark.asyncio
async def test_relative_redirect_is_followed(
    fetcher: HTTPFetcher,
) -> None:
    """Relative redirect targets should be resolved correctly."""

    redirect_response = make_response(
        status_code=302,
        headers={
            "location": "/article",
        },
        url="https://example.com/",
    )

    final_response = make_response(
        content=b"<html>Article</html>",
        url="https://example.com/article",
    )

    mock_client, mock_client_cm = create_mock_client(
        [
            redirect_response,
            final_response,
        ],
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ),
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        result = await fetcher.fetch(
            "https://example.com",
        )

    assert result.final_url == (
        "https://example.com/article"
    )

    assert result.content == (
        b"<html>Article</html>"
    )


@pytest.mark.asyncio
async def test_redirect_without_location_is_rejected(
    fetcher: HTTPFetcher,
) -> None:
    """Redirects without Location must be rejected."""

    response = make_response(
        status_code=302,
    )

    _, mock_client_cm = create_mock_client(
        [response],
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ),
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="missing the Location header",
        ):
            await fetcher.fetch(
                "https://example.com",
            )


@pytest.mark.asyncio
async def test_unsafe_redirect_is_rejected(
    fetcher: HTTPFetcher,
) -> None:
    """
    Redirect targets must pass URL security validation.
    """

    redirect_response = make_response(
        status_code=302,
        headers={
            "location": "http://127.0.0.1/admin",
        },
    )

    _, mock_client_cm = create_mock_client(
        [redirect_response],
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ),
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Loopback IP addresses are not allowed",
        ):
            await fetcher.fetch(
                "https://example.com",
            )


@pytest.mark.asyncio
async def test_redirect_limit_is_enforced(
    fetcher: HTTPFetcher,
) -> None:
    """
    Redirect chains must not exceed the configured limit.

    url_max_redirects is currently 5.

    The test supplies six redirect responses so the
    sixth redirect triggers the protection.
    """

    responses = [
        make_response(
            status_code=302,
            headers={
                "location": "https://example.com/next",
            },
        )
        for _ in range(6)
    ]

    mock_client, mock_client_cm = create_mock_client(
        responses,
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ),
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Maximum redirect limit exceeded",
        ):
            await fetcher.fetch(
                "https://example.com",
            )

    assert mock_client.get.await_count == 6


# ============================================================
# Connection Strategy Boundary
# ============================================================


@pytest.mark.asyncio
async def test_destination_resolution_uses_connection_strategy(
    mock_connection_strategy: AsyncMock,
) -> None:
    """
    HTTPFetcher must delegate hostname resolution and
    destination selection to ConnectionStrategy.
    """

    fetcher = HTTPFetcher(
        connection_strategy=mock_connection_strategy,
    )

    target = await fetcher._validate_destination(
        "https://example.com/article",
    )

    assert target.hostname == "example.com"
    assert target.port == 443

    mock_connection_strategy.select_target.assert_awaited_once_with(
        "example.com",
        443,
    )


@pytest.mark.asyncio
async def test_http_destination_uses_port_80(
    mock_connection_strategy: AsyncMock,
) -> None:
    """HTTP URLs should use port 80 when no port is specified."""

    fetcher = HTTPFetcher(
        connection_strategy=mock_connection_strategy,
    )

    await fetcher._validate_destination(
        "http://example.com/article",
    )

    mock_connection_strategy.select_target.assert_awaited_once_with(
        "example.com",
        80,
    )


@pytest.mark.asyncio
async def test_explicit_port_is_preserved(
    mock_connection_strategy: AsyncMock,
) -> None:
    """Explicit URL ports must be passed unchanged."""

    fetcher = HTTPFetcher(
        connection_strategy=mock_connection_strategy,
    )

    await fetcher._validate_destination(
        "https://example.com:8443/article",
    )

    mock_connection_strategy.select_target.assert_awaited_once_with(
        "example.com",
        8443,
    )


@pytest.mark.asyncio
async def test_destination_resolution_propagates_strategy_error(
    mock_connection_strategy: AsyncMock,
) -> None:
    """
    Security errors from the connection strategy must
    propagate instead of being swallowed.
    """

    mock_connection_strategy.select_target.side_effect = (
        ValueError(
            "Resolved destination is a private address."
        )
    )

    fetcher = HTTPFetcher(
        connection_strategy=mock_connection_strategy,
    )

    with pytest.raises(
        ValueError,
        match="Resolved destination is a private address",
    ):
        await fetcher._validate_destination(
            "https://example.com",
        )


@pytest.mark.asyncio
async def test_missing_hostname_is_rejected_before_strategy(
    mock_connection_strategy: AsyncMock,
) -> None:
    """
    HTTPFetcher should reject malformed URLs before invoking
    the connection strategy.
    """

    fetcher = HTTPFetcher(
        connection_strategy=mock_connection_strategy,
    )

    with pytest.raises(
        ValueError,
        match="URL must contain a hostname",
    ):
        await fetcher._validate_destination(
            "https:///missing-host",
        )

    mock_connection_strategy.select_target.assert_not_awaited()


@pytest.mark.asyncio
async def test_redirect_destination_is_resolved_again(
    mock_connection_strategy: AsyncMock,
) -> None:
    """
    Every redirect must trigger a fresh destination
    resolution rather than reusing the previous target.
    """

    mock_connection_strategy.select_target.side_effect = [
        ConnectionTarget(
            hostname="example.com",
            port=443,
            address=__import__("ipaddress").ip_address(
                "93.184.216.34"
            ),
        ),
        ConnectionTarget(
            hostname="example.org",
            port=443,
            address=__import__("ipaddress").ip_address(
                "93.184.216.34"
            ),
        ),
    ]

    fetcher = HTTPFetcher(
        connection_strategy=mock_connection_strategy,
    )

    redirect_response = make_response(
        status_code=302,
        headers={
            "location": "https://example.org/final",
        },
        url="https://example.com/",
    )

    final_response = make_response(
        content=b"<html>Final</html>",
        url="https://example.org/final",
    )

    mock_client, mock_client_cm = create_mock_client(
        [
            redirect_response,
            final_response,
        ],
    )

    with patch(
        "app.ingestion.fetchers.http.httpx.AsyncClient",
        return_value=mock_client_cm,
    ):
        result = await fetcher.fetch(
            "https://example.com",
        )

    assert result.final_url == (
        "https://example.org/final"
    )

    assert (
        mock_connection_strategy.select_target.await_count
        == 2
    )

    calls = (
        mock_connection_strategy
        .select_target
        .await_args_list
    )

    assert calls[0].args == (
        "example.com",
        443,
    )

    assert calls[1].args == (
        "example.org",
        443,
    )


# ============================================================
# Response Size Protection
# ============================================================


@pytest.mark.asyncio
async def test_content_length_limit_is_enforced(
    fetcher: HTTPFetcher,
) -> None:
    """
    A response whose declared Content-Length exceeds the
    configured maximum should be rejected.
    """

    response = make_response(
        headers={
            "content-length": str(
                20 * 1024 * 1024
            ),
        },
    )

    _, mock_client_cm = create_mock_client(
        [response],
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ),
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Response exceeds the maximum allowed size",
        ):
            await fetcher.fetch(
                "https://example.com",
            )


@pytest.mark.asyncio
async def test_invalid_content_length_is_rejected(
    fetcher: HTTPFetcher,
) -> None:
    """Malformed Content-Length values should be rejected."""

    response = make_response(
        headers={
            "content-length": "not-a-number",
        },
    )

    _, mock_client_cm = create_mock_client(
        [response],
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ),
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        with pytest.raises(
            ValueError,
            match="Invalid Content-Length header",
        ):
            await fetcher.fetch(
                "https://example.com",
            )


# ============================================================
# HTTP Client Configuration
# ============================================================


@pytest.mark.asyncio
async def test_http_client_disables_automatic_redirects(
    fetcher: HTTPFetcher,
) -> None:
    """Automatic redirects must remain disabled."""

    response = make_response()

    _, mock_client_cm = create_mock_client(
        [response],
    )

    with (
        patch(
            "app.ingestion.fetchers.http.httpx.AsyncClient",
            return_value=mock_client_cm,
        ) as async_client,
        patch.object(
            fetcher,
            "_validate_destination",
            new=AsyncMock(),
        ),
    ):
        await fetcher.fetch(
            "https://example.com",
        )

    async_client.assert_called_once()

    call_kwargs = (
        async_client.call_args.kwargs
    )

    assert call_kwargs[
        "follow_redirects"
    ] is False