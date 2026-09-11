from __future__ import annotations

from ipaddress import IPv4Address, IPv6Address

import pytest

from app.ingestion.fetchers.connection import (
    ConnectionTarget,
    SecureConnectionStrategy,
)
from app.ingestion.fetchers.dns import ResolvedDestination


class FakeDNSResolver:
    def __init__(
        self,
        destination: ResolvedDestination | None = None,
        error: Exception | None = None,
    ) -> None:
        self.destination = destination
        self.error = error
        self.calls: list[tuple[str, int]] = []

    async def resolve(
        self,
        hostname: str,
        port: int,
    ) -> ResolvedDestination:
        self.calls.append((hostname, port))

        if self.error is not None:
            raise self.error

        if self.destination is None:
            raise AssertionError("No DNS destination configured.")

        return self.destination


@pytest.mark.asyncio
async def test_select_target_with_ipv4_address() -> None:
    destination = ResolvedDestination(
        hostname="example.com",
        port=443,
        addresses=(
            IPv4Address("93.184.216.34"),
        ),
    )

    resolver = FakeDNSResolver(destination=destination)
    strategy = SecureConnectionStrategy(resolver=resolver)

    target = await strategy.select_target("example.com", 443)

    assert target == ConnectionTarget(
        hostname="example.com",
        port=443,
        address=IPv4Address("93.184.216.34"),
    )

    assert resolver.calls == [
        ("example.com", 443),
    ]


@pytest.mark.asyncio
async def test_select_target_with_ipv6_address() -> None:
    destination = ResolvedDestination(
        hostname="example.com",
        port=443,
        addresses=(
            IPv6Address("2001:db8::1"),
        ),
    )

    resolver = FakeDNSResolver(destination=destination)
    strategy = SecureConnectionStrategy(resolver=resolver)

    target = await strategy.select_target("example.com", 443)

    assert target.hostname == "example.com"
    assert target.port == 443
    assert target.address == IPv6Address("2001:db8::1")


@pytest.mark.asyncio
async def test_first_valid_address_is_selected() -> None:
    destination = ResolvedDestination(
        hostname="example.com",
        port=443,
        addresses=(
            IPv4Address("93.184.216.34"),
            IPv4Address("93.184.216.35"),
        ),
    )

    resolver = FakeDNSResolver(destination=destination)
    strategy = SecureConnectionStrategy(resolver=resolver)

    target = await strategy.select_target("example.com", 443)

    assert target.address == IPv4Address("93.184.216.34")


@pytest.mark.asyncio
async def test_hostname_is_preserved_for_tls_sni() -> None:
    destination = ResolvedDestination(
        hostname="secure.example.com",
        port=443,
        addresses=(
            IPv4Address("93.184.216.34"),
        ),
    )

    resolver = FakeDNSResolver(destination=destination)
    strategy = SecureConnectionStrategy(resolver=resolver)

    target = await strategy.select_target(
        "secure.example.com",
        443,
    )

    assert target.hostname == "secure.example.com"
    assert target.address == IPv4Address("93.184.216.34")


@pytest.mark.asyncio
async def test_resolver_error_is_propagated() -> None:
    resolver = FakeDNSResolver(
        error=ValueError("Unable to resolve hostname."),
    )

    strategy = SecureConnectionStrategy(resolver=resolver)

    with pytest.raises(ValueError, match="Unable to resolve hostname"):
        await strategy.select_target("example.com", 443)


@pytest.mark.asyncio
async def test_empty_address_list_is_rejected() -> None:
    destination = ResolvedDestination(
        hostname="example.com",
        port=443,
        addresses=(),
    )

    resolver = FakeDNSResolver(destination=destination)
    strategy = SecureConnectionStrategy(resolver=resolver)

    with pytest.raises(
        ValueError,
        match="DNS resolution returned no addresses",
    ):
        await strategy.select_target("example.com", 443)


@pytest.mark.asyncio
async def test_resolver_receives_original_hostname_and_port() -> None:
    destination = ResolvedDestination(
        hostname="api.example.com",
        port=8443,
        addresses=(
            IPv4Address("93.184.216.34"),
        ),
    )

    resolver = FakeDNSResolver(destination=destination)
    strategy = SecureConnectionStrategy(resolver=resolver)

    await strategy.select_target(
        "api.example.com",
        8443,
    )

    assert resolver.calls == [
        ("api.example.com", 8443),
    ]


@pytest.mark.asyncio
async def test_strategy_does_not_perform_independent_dns_lookup() -> None:
    destination = ResolvedDestination(
        hostname="example.com",
        port=443,
        addresses=(
            IPv4Address("93.184.216.34"),
        ),
    )

    resolver = FakeDNSResolver(destination=destination)
    strategy = SecureConnectionStrategy(resolver=resolver)

    await strategy.select_target("example.com", 443)

    # Exactly one resolution must happen through the injected resolver.
    assert len(resolver.calls) == 1