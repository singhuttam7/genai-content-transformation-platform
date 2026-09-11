from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from app.core.config import settings
from app.ingestion.fetchers.connection import (
    ConnectionStrategy,
    SecureConnectionStrategy,
)
from app.ingestion.fetchers.dns import (
    SecureDNSResolver,
)
from app.ingestion.fetchers.transport import (
    PinnedHTTPTransport,
)
from app.security.sanitization.url import (
    URLSecurityValidator,
)


# ============================================================
# Fetched Response
# ============================================================


@dataclass(frozen=True, slots=True)
class FetchedResponse:
    """
    Normalized HTTP response returned by the ingestion fetcher.
    """

    url: str
    final_url: str
    status_code: int
    content_type: str
    content: bytes
    headers: dict[str, str]


# ============================================================
# HTTP Fetcher
# ============================================================


class HTTPFetcher:
    """
    Secure asynchronous HTTP fetcher for URL ingestion.

    Responsibilities:
    - Validate URLs using the security layer.
    - Obtain a security-approved connection target.
    - Use the validated connection target for the actual
      TCP connection.
    - Disable automatic redirects.
    - Validate every redirect target.
    - Enforce redirect limits.
    - Enforce request timeout.
    - Validate HTTP status.
    - Validate response Content-Type.
    - Stream response bodies with a strict size limit.

    DNS resolution and destination IP validation are delegated
    to the connection strategy.

    The actual network connection is delegated to the pinned
    HTTP transport.

    The fetcher does not parse HTML.
    Parsing belongs to the ingestion parser layer.
    """

    REDIRECT_STATUS_CODES = frozenset(
        {
            301,
            302,
            303,
            307,
            308,
        }
    )

    def __init__(
        self,
        *,
        validator: URLSecurityValidator | None = None,
        connection_strategy: ConnectionStrategy | None = None,
    ) -> None:
        """
        Initialize the HTTP fetcher.

        Args:
            validator:
                URL-level security validator.

            connection_strategy:
                Strategy responsible for securely resolving the
                destination hostname and selecting a validated
                connection target.

        Dependency injection is supported so that the fetcher
        can be tested without performing real DNS resolution.
        """

        self.validator = (
            validator or URLSecurityValidator()
        )

        self.connection_strategy = (
            connection_strategy
            or SecureConnectionStrategy(
                resolver=SecureDNSResolver()
            )
        )

    # ========================================================
    # Public API
    # ========================================================

    async def fetch(
        self,
        url: str,
    ) -> FetchedResponse:
        """
        Fetch a URL securely.

        Every URL, including redirect destinations, is:

            1. URL validated.
            2. DNS resolved through the security layer.
            3. Converted into a validated ConnectionTarget.
            4. Passed to PinnedHTTPTransport.
            5. Connected directly to the validated IP.

        Raises:
            ValueError:
                If the URL or response violates security policy.

            httpx.HTTPError:
                If the HTTP request fails at the transport layer.
        """

        # ----------------------------------------------------
        # Initial URL validation
        # ----------------------------------------------------

        current_url = self.validator.validate(url)

        max_redirects = (
            settings.url_max_redirects
        )

        timeout = httpx.Timeout(
            settings.url_fetch_timeout_seconds
        )

        # ----------------------------------------------------
        # Redirect-aware request loop
        # ----------------------------------------------------

        for redirect_count in range(
            max_redirects + 1
        ):

            # ------------------------------------------------
            # Secure destination resolution
            # ------------------------------------------------

            connection_target = (
                await self._validate_destination(
                    current_url
                )
            )

            # ------------------------------------------------
            # Create a transport pinned to the validated
            # connection target.
            # ------------------------------------------------

            transport = PinnedHTTPTransport(
                connection_target
            )

            # ------------------------------------------------
            # HTTP client
            #
            # A new transport/client is intentionally created
            # for every redirect destination.
            #
            # This prevents a transport associated with one
            # validated destination from being reused for a
            # different redirect destination.
            # ------------------------------------------------

            async with httpx.AsyncClient(
                transport=transport,
                timeout=timeout,
                follow_redirects=False,
            ) as client:

                # --------------------------------------------
                # Perform request
                # --------------------------------------------

                response = await client.get(
                    current_url
                )

                # --------------------------------------------
                # Redirect handling
                # --------------------------------------------

                if (
                    response.status_code
                    in self.REDIRECT_STATUS_CODES
                ):

                    if (
                        redirect_count
                        >= max_redirects
                    ):
                        raise ValueError(
                            "Maximum redirect limit exceeded."
                        )

                    location = (
                        response.headers.get(
                            "location"
                        )
                    )

                    if not location:
                        raise ValueError(
                            "Redirect response is missing "
                            "the Location header."
                        )

                    # ----------------------------------------
                    # Resolve relative redirect against the
                    # current URL.
                    # ----------------------------------------

                    next_url = str(
                        httpx.URL(
                            current_url
                        ).join(
                            location
                        )
                    )

                    # ----------------------------------------
                    # Every redirect must pass through the
                    # URL security validator before another
                    # request is made.
                    # ----------------------------------------

                    current_url = (
                        self.validator.validate(
                            next_url
                        )
                    )

                    continue

                # --------------------------------------------
                # Final response
                # --------------------------------------------

                self._validate_status(
                    response.status_code
                )

                content_type = (
                    self._get_content_type(
                        response
                    )
                )

                self._validate_content_type(
                    content_type
                )

                content = (
                    await self._read_response(
                        response
                    )
                )

                return FetchedResponse(
                    url=url,
                    final_url=current_url,
                    status_code=response.status_code,
                    content_type=content_type,
                    content=content,
                    headers=dict(
                        response.headers
                    ),
                )

        raise RuntimeError(
            "HTTP fetcher exited without a response."
        )

    # ========================================================
    # Secure Destination Resolution
    # ========================================================

    async def _validate_destination(
        self,
        url: str,
    ):
        """
        Validate the URL destination and return the
        security-approved connection target.

        DNS resolution and destination IP validation are
        delegated to the connection strategy.

        The returned target represents the exact destination
        that will be used by PinnedHTTPTransport.

        This method intentionally does not perform an
        independent DNS lookup.
        """

        parsed = urlsplit(url)

        hostname = parsed.hostname

        if not hostname:
            raise ValueError(
                "URL must contain a hostname."
            )

        port = parsed.port or (
            443
            if parsed.scheme == "https"
            else 80
        )

        return await self.connection_strategy.select_target(
            hostname,
            port,
        )

    # ========================================================
    # HTTP Status
    # ========================================================

    @staticmethod
    def _validate_status(
        status_code: int,
    ) -> None:
        """
        Reject unsuccessful HTTP responses.
        """

        if status_code >= 400:
            raise ValueError(
                f"HTTP request failed with status "
                f"{status_code}."
            )

    # ========================================================
    # Content-Type
    # ========================================================

    @staticmethod
    def _get_content_type(
        response: httpx.Response,
    ) -> str:
        """
        Extract normalized media type from Content-Type.

        Example:

            text/html; charset=utf-8

        becomes:

            text/html
        """

        content_type = (
            response.headers.get(
                "content-type",
                "",
            )
        )

        content_type = (
            content_type
            .split(";", 1)[0]
            .strip()
            .lower()
        )

        if not content_type:
            raise ValueError(
                "Response is missing Content-Type."
            )

        return content_type

    @staticmethod
    def _validate_content_type(
        content_type: str,
    ) -> None:
        """
        Allow only configured response content types.
        """

        allowed = {
            value.strip().lower()
            for value
            in settings.url_allowed_content_types
        }

        if content_type not in allowed:
            raise ValueError(
                f"Unsupported response Content-Type: "
                f"{content_type}."
            )

    # ========================================================
    # Response Size Protection
    # ========================================================

    @staticmethod
    async def _read_response(
        response: httpx.Response,
    ) -> bytes:
        """
        Read response incrementally while enforcing the
        configured maximum response size.
        """

        max_size = (
            settings.url_max_response_size_mb
            * 1024
            * 1024
        )

        # ----------------------------------------------------
        # Check declared Content-Length first
        # ----------------------------------------------------

        content_length = (
            response.headers.get(
                "content-length"
            )
        )

        if content_length is not None:

            try:
                declared_size = int(
                    content_length
                )

            except ValueError as exc:
                raise ValueError(
                    "Invalid Content-Length header."
                ) from exc

            if declared_size > max_size:
                raise ValueError(
                    "Response exceeds the maximum "
                    "allowed size."
                )

        # ----------------------------------------------------
        # Incremental body reading
        # ----------------------------------------------------

        chunks: list[bytes] = []

        total_size = 0

        async for chunk in response.aiter_bytes():

            total_size += len(chunk)

            if total_size > max_size:
                raise ValueError(
                    "Response exceeds the maximum "
                    "allowed size."
                )

            chunks.append(chunk)

        return b"".join(chunks)