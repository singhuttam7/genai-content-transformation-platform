from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address
from typing import Protocol


IPAddress = IPv4Address | IPv6Address


@dataclass(frozen=True, slots=True)
class ResolvedDestination:
    """
    Result of resolving a network destination.

    The addresses represent the IP destinations resolved
    for the requested hostname and port.
    """

    hostname: str
    port: int
    addresses: tuple[IPAddress, ...]


class DNSResolver(Protocol):
    """
    Contract for asynchronous hostname resolution.
    """

    async def resolve(
        self,
        hostname: str,
        port: int,
    ) -> ResolvedDestination:
        """
        Resolve a hostname into validated network addresses.
        """
        ...


class SecureDNSResolver:
    """
    Secure asynchronous DNS resolver.

    Responsibilities:
    - Resolve hostnames without blocking the event loop.
    - Support IPv4 and IPv6.
    - Reject unsafe destination addresses.
    - Remove duplicate addresses.
    - Return a deterministic resolution result.

    HTTP behavior and URL parsing are intentionally outside
    this component.
    """

    async def resolve(
        self,
        hostname: str,
        port: int,
    ) -> ResolvedDestination:
        """
        Resolve and validate a hostname.

        Raises:
            ValueError:
                If the hostname cannot be resolved or resolves
                to an unsafe destination.
        """

        if not hostname:
            raise ValueError(
                "Hostname cannot be empty."
            )

        if not isinstance(port, int):
            raise ValueError(
                "Port must be an integer."
            )

        if not 1 <= port <= 65535:
            raise ValueError(
                "Port must be between 1 and 65535."
            )

        try:
            results = await asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )

        except OSError as exc:
            raise ValueError(
                "Unable to resolve hostname."
            ) from exc

        if not results:
            raise ValueError(
                "Hostname did not resolve."
            )

        addresses: list[IPAddress] = []

        seen: set[IPAddress] = set()

        for result in results:
            sockaddr = result[4]

            if not sockaddr:
                raise ValueError(
                    "DNS resolution returned an invalid address."
                )

            ip_string = sockaddr[0]

            try:
                address = ipaddress.ip_address(
                    ip_string
                )

            except ValueError as exc:
                raise ValueError(
                    "DNS resolution returned an invalid "
                    "IP address."
                ) from exc

            self._validate_address(
                address
            )

            if address not in seen:
                seen.add(address)
                addresses.append(address)

        if not addresses:
            raise ValueError(
                "Hostname did not resolve to a valid address."
            )

        return ResolvedDestination(
            hostname=hostname,
            port=port,
            addresses=tuple(addresses),
        )

    @staticmethod
    def _validate_address(
        address: IPAddress,
    ) -> None:
        """
        Reject unsafe IP destinations.

        Specific classifications are checked before private
        because Python's ipaddress module can classify some
        special-purpose addresses as private.
        """

        # ----------------------------------------------------
        # Loopback
        # ----------------------------------------------------

        if address.is_loopback:
            raise ValueError(
                "Resolved destination is a "
                "loopback address."
            )

        # ----------------------------------------------------
        # Unspecified
        # ----------------------------------------------------

        if address.is_unspecified:
            raise ValueError(
                "Resolved destination is "
                "unspecified."
            )

        # ----------------------------------------------------
        # Link-local
        # ----------------------------------------------------

        if address.is_link_local:
            raise ValueError(
                "Resolved destination is a "
                "link-local address."
            )

        # ----------------------------------------------------
        # Multicast
        # ----------------------------------------------------

        if address.is_multicast:
            raise ValueError(
                "Resolved destination is a "
                "multicast address."
            )

        # ----------------------------------------------------
        # Private
        # ----------------------------------------------------

        if address.is_private:
            raise ValueError(
                "Resolved destination is a "
                "private address."
            )