from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address
from typing import Protocol

from app.ingestion.fetchers.dns import DNSResolver, ResolvedDestination


IPAddress = IPv4Address | IPv6Address


@dataclass(frozen=True, slots=True)
class ConnectionTarget:
    """
    Validated network destination selected for an outbound connection.

    hostname:
        Original hostname used for TLS/SNI verification.

    port:
        Destination TCP port.

    address:
        Validated IP address to which the connection should be established.
    """

    hostname: str
    port: int
    address: IPAddress


class ConnectionStrategy(Protocol):
    """
    Contract for selecting a validated outbound connection target.

    The strategy is intentionally independent of HTTP. This allows the
    networking layer to evolve without coupling DNS/security decisions
    directly to the HTTP fetcher.
    """

    async def select_target(
        self,
        hostname: str,
        port: int,
    ) -> ConnectionTarget:
        ...


class SecureConnectionStrategy:
    """
    Selects a safe outbound connection target using a DNSResolver.

    Security responsibilities:
    - Obtain addresses only through the injected DNS resolver.
    - Trust only addresses already validated by that resolver.
    - Preserve the original hostname for TLS/SNI.
    - Never perform a second DNS lookup here.

    Actual TCP/TLS/HTTP connection establishment is intentionally handled
    by the transport integration layer.
    """

    def __init__(self, *, resolver: DNSResolver) -> None:
        self.resolver = resolver

    async def select_target(
        self,
        hostname: str,
        port: int,
    ) -> ConnectionTarget:
        destination = await self.resolver.resolve(hostname, port)

        address = self._select_address(destination)

        return ConnectionTarget(
            hostname=destination.hostname,
            port=destination.port,
            address=address,
        )

    @staticmethod
    def _select_address(
        destination: ResolvedDestination,
    ) -> IPAddress:
        if not destination.addresses:
            raise ValueError("DNS resolution returned no addresses.")

        # The DNS resolver has already validated every address.
        # Preserve resolver ordering so connection policy remains
        # deterministic from the strategy's perspective.
        return destination.addresses[0]