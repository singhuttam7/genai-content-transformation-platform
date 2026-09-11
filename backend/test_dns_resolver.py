from __future__ import annotations

import socket
from ipaddress import IPv4Address, IPv6Address

import pytest

from app.ingestion.fetchers.dns import SecureDNSResolver


def make_addrinfo(ip: str) -> tuple:
    return (
        socket.AF_INET6 if ":" in ip else socket.AF_INET,
        socket.SOCK_STREAM,
        6,
        "",
        (ip, 443),
    )


class FakeSocket:
    def __init__(
        self,
        results: list[tuple] | None = None,
        error: OSError | None = None,
    ) -> None:
        self.results = results or []
        self.error = error

    def getaddrinfo(
        self,
        hostname: str,
        port: int,
        *,
        type: int,
    ) -> list[tuple]:
        if self.error is not None:
            raise self.error
        return self.results


@pytest.mark.asyncio
async def test_resolve_ipv4(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSocket(
        results=[make_addrinfo("93.184.216.34")]
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fake.getaddrinfo,
    )

    resolver = SecureDNSResolver()

    result = await resolver.resolve("example.com", 443)

    assert result.hostname == "example.com"
    assert result.port == 443
    assert result.addresses == (
        IPv4Address("93.184.216.34"),
    )


@pytest.mark.asyncio
async def test_resolve_ipv6(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSocket(
        results=[make_addrinfo("2606:4700:4700::1111")]
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fake.getaddrinfo,
    )

    resolver = SecureDNSResolver()

    result = await resolver.resolve("example.com", 443)

    assert result.addresses == (
        IPv6Address("2606:4700:4700::1111"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.1.1",
        "224.0.0.1",
        "0.0.0.0",
    ],
)
async def test_rejects_unsafe_ipv4(
    monkeypatch: pytest.MonkeyPatch,
    ip: str,
) -> None:
    fake = FakeSocket(
        results=[make_addrinfo(ip)]
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fake.getaddrinfo,
    )

    resolver = SecureDNSResolver()

    with pytest.raises(ValueError):
        await resolver.resolve("example.com", 443)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ip",
    [
        "::1",
        "::",
        "fe80::1",
        "ff02::1",
        "fc00::1",
    ],
)
async def test_rejects_unsafe_ipv6(
    monkeypatch: pytest.MonkeyPatch,
    ip: str,
) -> None:
    fake = FakeSocket(
        results=[make_addrinfo(ip)]
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fake.getaddrinfo,
    )

    resolver = SecureDNSResolver()

    with pytest.raises(ValueError):
        await resolver.resolve("example.com", 443)


@pytest.mark.asyncio
async def test_dns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSocket(
        error=OSError("DNS failure")
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fake.getaddrinfo,
    )

    resolver = SecureDNSResolver()

    with pytest.raises(
        ValueError,
        match="Unable to resolve hostname",
    ):
        await resolver.resolve("example.com", 443)


@pytest.mark.asyncio
async def test_empty_hostname() -> None:
    resolver = SecureDNSResolver()

    with pytest.raises(
        ValueError,
        match="Hostname cannot be empty",
    ):
        await resolver.resolve("", 443)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "port",
    [
        0,
        -1,
        65536,
        70000,
    ],
)
async def test_invalid_port(port: int) -> None:
    resolver = SecureDNSResolver()

    with pytest.raises(
        ValueError,
        match="Port must be between 1 and 65535",
    ):
        await resolver.resolve("example.com", port)


@pytest.mark.asyncio
async def test_non_integer_port() -> None:
    resolver = SecureDNSResolver()

    with pytest.raises(
        ValueError,
        match="Port must be an integer",
    ):
        await resolver.resolve("example.com", "443")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_empty_dns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSocket(results=[])

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fake.getaddrinfo,
    )

    resolver = SecureDNSResolver()

    with pytest.raises(
        ValueError,
        match="Hostname did not resolve",
    ):
        await resolver.resolve("example.com", 443)


@pytest.mark.asyncio
async def test_duplicate_addresses_are_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ip = "93.184.216.34"

    fake = FakeSocket(
        results=[
            make_addrinfo(ip),
            make_addrinfo(ip),
            make_addrinfo(ip),
        ]
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fake.getaddrinfo,
    )

    resolver = SecureDNSResolver()

    result = await resolver.resolve("example.com", 443)

    assert result.addresses == (
        IPv4Address(ip),
    )


@pytest.mark.asyncio
async def test_multiple_valid_addresses_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSocket(
        results=[
            make_addrinfo("93.184.216.34"),
            make_addrinfo("93.184.216.35"),
            make_addrinfo("2606:4700:4700::1111"),
        ]
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fake.getaddrinfo,
    )

    resolver = SecureDNSResolver()

    result = await resolver.resolve("example.com", 443)

    assert result.addresses == (
        IPv4Address("93.184.216.34"),
        IPv4Address("93.184.216.35"),
        IPv6Address("2606:4700:4700::1111"),
    )


@pytest.mark.asyncio
async def test_mixed_safe_and_unsafe_addresses_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeSocket(
        results=[
            make_addrinfo("93.184.216.34"),
            make_addrinfo("127.0.0.1"),
        ]
    )

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fake.getaddrinfo,
    )

    resolver = SecureDNSResolver()

    with pytest.raises(
        ValueError,
        match="loopback",
    ):
        await resolver.resolve("example.com", 443)