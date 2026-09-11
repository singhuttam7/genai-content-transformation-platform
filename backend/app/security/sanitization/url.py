from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit, urlunsplit


class URLSecurityValidator:
    """
    Validate and normalize externally supplied URLs.

    This validator performs URL-level security checks before
    any network request is made.

    Network-level checks such as DNS resolution, redirect
    validation, and connection policy belong to the HTTP
    fetcher.
    """

    ALLOWED_SCHEMES = frozenset({"http", "https"})

    def validate(self, url: str) -> str:
        """
        Validate a URL and return its normalized representation.

        Raises:
            ValueError: If the URL is invalid or unsafe.
        """

        if not isinstance(url, str):
            raise ValueError("URL must be a string.")

        url = url.strip()

        if not url:
            raise ValueError("URL cannot be empty.")

        if any(character.isspace() for character in url):
            raise ValueError(
                "URL cannot contain whitespace."
            )

        try:
            parsed = urlsplit(url)
        except ValueError as exc:
            raise ValueError("Invalid URL.") from exc

        self._validate_scheme(parsed.scheme)
        self._validate_credentials(parsed)
        self._validate_hostname(parsed.hostname)

        return self._normalize(parsed)

    @classmethod
    def _validate_scheme(cls, scheme: str) -> None:
        """Allow only HTTP and HTTPS schemes."""

        scheme = scheme.lower()

        if scheme not in cls.ALLOWED_SCHEMES:
            raise ValueError(
                "Only HTTP and HTTPS URLs are allowed."
            )

    @staticmethod
    def _validate_credentials(parsed) -> None:
        """Reject URLs containing embedded credentials."""

        if (
            parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "URLs containing credentials are not allowed."
            )

    @staticmethod
    def _validate_hostname(
        hostname: str | None,
    ) -> None:
        """Validate hostname and reject unsafe literal IPs."""

        if not hostname:
            raise ValueError(
                "URL must contain a hostname."
            )

        hostname = hostname.strip().lower()

        # --------------------------------------------------------
        # Localhost aliases
        # --------------------------------------------------------

        if hostname == "localhost":
            raise ValueError(
                "Localhost URLs are not allowed."
            )

        if hostname.endswith(".localhost"):
            raise ValueError(
                "Localhost URLs are not allowed."
            )

        if hostname in {
            "localhost.localdomain",
            "ip6-localhost",
            "ip6-loopback",
        }:
            raise ValueError(
                "Localhost URLs are not allowed."
            )

        # --------------------------------------------------------
        # Literal IP validation
        # --------------------------------------------------------

        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            # Normal DNS hostname.
            return

        # Check specific/special classifications BEFORE
        # is_private because Python's ipaddress module may
        # classify some special addresses as private as well.

        if address.is_loopback:
            raise ValueError(
                "Loopback IP addresses are not allowed."
            )

        if address.is_unspecified:
            raise ValueError(
                "Unspecified IP addresses are not allowed."
            )

        if address.is_link_local:
            raise ValueError(
                "Link-local IP addresses are not allowed."
            )

        if address.is_multicast:
            raise ValueError(
                "Multicast IP addresses are not allowed."
            )

        if address.is_private:
            raise ValueError(
                "Private IP addresses are not allowed."
            )

    @staticmethod
    def _normalize(parsed) -> str:
        """
        Normalize the URL without changing its meaning.

        Preserves:
        - scheme
        - hostname
        - port
        - path
        - query
        - fragment
        """

        scheme = parsed.scheme.lower()

        hostname = parsed.hostname

        if hostname is None:
            raise ValueError(
                "URL must contain a hostname."
            )

        hostname = hostname.lower()

        # IPv6 hostnames need brackets when reconstructed
        # into a URL netloc.
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"

        port = parsed.port

        if port is not None:
            netloc = f"{hostname}:{port}"
        else:
            netloc = hostname

        return urlunsplit(
            (
                scheme,
                netloc,
                parsed.path or "/",
                parsed.query,
                parsed.fragment,
            )
        )