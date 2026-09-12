from __future__ import annotations

import pytest

from app.security.sanitization.url import URLSecurityValidator


@pytest.fixture
def validator() -> URLSecurityValidator:
    return URLSecurityValidator()


# ============================================================
# Valid URLs
# ============================================================


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://example.com",
        "https://example.com/article",
        "https://example.com/news/security",
        "https://example.com:8443/article",
        "https://example.com/article?id=123",
        "https://example.com/article?id=123&lang=en",
        "https://subdomain.example.com/article",
        "HTTPS://EXAMPLE.COM/article",
    ],
)
def test_valid_urls(
    validator: URLSecurityValidator,
    url: str,
) -> None:
    result = validator.validate(url)

    assert result
    assert result.startswith(("http://", "https://"))


# ============================================================
# Empty / malformed URLs
# ============================================================


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not-a-url",
        "://example.com",
        "https://",
        "https:///article",
    ],
)
def test_invalid_urls(
    validator: URLSecurityValidator,
    url: str,
) -> None:
    with pytest.raises(ValueError):
        validator.validate(url)


def test_non_string_url(
    validator: URLSecurityValidator,
) -> None:
    with pytest.raises(
        ValueError,
        match="URL must be a string",
    ):
        validator.validate(None)  # type: ignore[arg-type]


def test_url_with_whitespace(
    validator: URLSecurityValidator,
) -> None:
    with pytest.raises(
        ValueError,
        match="whitespace",
    ):
        validator.validate(
            "https://example.com/some article"
        )


def test_surrounding_whitespace_is_normalized(
    validator: URLSecurityValidator,
) -> None:
    result = validator.validate(
        "  https://example.com/article  "
    )

    assert result == "https://example.com/article"


# ============================================================
# Scheme validation
# ============================================================


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "data:text/html,test",
        "ssh://example.com",
        "mailto:test@example.com",
    ],
)
def test_unsupported_schemes(
    validator: URLSecurityValidator,
    url: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="HTTP and HTTPS",
    ):
        validator.validate(url)


# ============================================================
# Credential protection
# ============================================================


@pytest.mark.parametrize(
    "url",
    [
        "https://user@example.com",
        "https://user:password@example.com",
        "http://admin:secret@example.com/article",
    ],
)
def test_urls_with_credentials_are_rejected(
    validator: URLSecurityValidator,
    url: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="credentials",
    ):
        validator.validate(url)


# ============================================================
# Localhost protection
# ============================================================


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost",
        "http://localhost:8000",
        "https://localhost/admin",
        "http://localhost.localdomain",
        "http://ip6-localhost",
        "http://ip6-loopback",
    ],
)
def test_localhost_urls_are_rejected(
    validator: URLSecurityValidator,
    url: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Localhost",
    ):
        validator.validate(url)


# ============================================================
# Loopback protection
# ============================================================


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1",
        "http://127.0.0.2",
        "http://127.255.255.254",
        "http://[::1]",
    ],
)
def test_loopback_addresses_are_rejected(
    validator: URLSecurityValidator,
    url: str,
) -> None:
    with pytest.raises(ValueError):
        validator.validate(url)


# ============================================================
# Private IPv4 protection
# ============================================================


@pytest.mark.parametrize(
    "url",
    [
        "http://10.0.0.1",
        "http://10.255.255.254",
        "http://172.16.0.1",
        "http://172.31.255.254",
        "http://192.168.0.1",
        "http://192.168.100.50",
    ],
)
def test_private_ipv4_addresses_are_rejected(
    validator: URLSecurityValidator,
    url: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Private IP",
    ):
        validator.validate(url)


# ============================================================
# Link-local protection
# ============================================================


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.0.1",
        "http://169.254.169.254",
    ],
)
def test_link_local_addresses_are_rejected(
    validator: URLSecurityValidator,
    url: str,
) -> None:
    with pytest.raises(ValueError):
        validator.validate(url)


# ============================================================
# IPv6 protection
# ============================================================


@pytest.mark.parametrize(
    "url",
    [
        "http://[::1]",
        "http://[fc00::1]",
        "http://[fd00::1]",
        "http://[fe80::1]",
        "http://[::]",
    ],
)
def test_unsafe_ipv6_addresses_are_rejected(
    validator: URLSecurityValidator,
    url: str,
) -> None:
    with pytest.raises(ValueError):
        validator.validate(url)


# ============================================================
# Multicast protection
# ============================================================


@pytest.mark.parametrize(
    "url",
    [
        "http://224.0.0.1",
        "http://239.255.255.255",
        "http://[ff02::1]",
    ],
)
def test_multicast_addresses_are_rejected(
    validator: URLSecurityValidator,
    url: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Multicast",
    ):
        validator.validate(url)


# ============================================================
# Unspecified address protection
# ============================================================


@pytest.mark.parametrize(
    "url",
    [
        "http://0.0.0.0",
        "http://[::]",
    ],
)
def test_unspecified_addresses_are_rejected(
    validator: URLSecurityValidator,
    url: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="Unspecified",
    ):
        validator.validate(url)


# ============================================================
# URL normalization
# ============================================================


def test_scheme_is_normalized(
    validator: URLSecurityValidator,
) -> None:
    result = validator.validate(
        "HTTPS://Example.COM/article"
    )

    assert result == "https://example.com/article"


def test_default_path_is_added(
    validator: URLSecurityValidator,
) -> None:
    result = validator.validate(
        "https://example.com"
    )

    assert result == "https://example.com/"


def test_query_is_preserved(
    validator: URLSecurityValidator,
) -> None:
    result = validator.validate(
        "https://example.com/search?q=security&page=2"
    )

    assert result == (
        "https://example.com/search?q=security&page=2"
    )


def test_fragment_is_preserved(
    validator: URLSecurityValidator,
) -> None:
    result = validator.validate(
        "https://example.com/article#section"
    )

    assert result == (
        "https://example.com/article#section"
    )


def test_port_is_preserved(
    validator: URLSecurityValidator,
) -> None:
    result = validator.validate(
        "https://example.com:8443/article"
    )

    assert result == (
        "https://example.com:8443/article"
    )


# ============================================================
# Special hostname cases
# ============================================================


def test_hostname_is_required(
    validator: URLSecurityValidator,
) -> None:
    with pytest.raises(
        ValueError,
        match="hostname",
    ):
        validator.validate(
            "https:///article"
        )


def test_multicast_hostname_is_rejected(
    validator: URLSecurityValidator,
) -> None:
    with pytest.raises(
        ValueError,
        match="Multicast",
    ):
        validator.validate(
            "http://224.0.0.1"
        )