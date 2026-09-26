from __future__ import annotations

from collections.abc import Iterator

from app.tools.base.port import ToolPort


class ToolRegistry:
    """
    Registry for discovering tools by their canonical names.

    The registry is responsible only for registration and lookup.
    It does not execute tools and does not contain orchestration logic.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolPort] = {}

    def register(
        self,
        name: str,
        tool: ToolPort,
    ) -> None:
        """
        Register a tool under a canonical name.

        Duplicate names are rejected to prevent accidental replacement
        of an already-registered capability.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Tool name must be a non-empty string."
            )

        if not isinstance(tool, ToolPort):
            raise TypeError(
                "tool must be a ToolPort."
            )

        canonical_name = name.strip()

        if canonical_name in self._tools:
            raise ValueError(
                f"Tool '{canonical_name}' is already registered."
            )

        self._tools[canonical_name] = tool

    def resolve(
        self,
        name: str,
    ) -> ToolPort:
        """
        Resolve a registered tool by name.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Tool name must be a non-empty string."
            )

        canonical_name = name.strip()

        try:
            return self._tools[canonical_name]
        except KeyError as exc:
            raise KeyError(
                f"Tool '{canonical_name}' is not registered."
            ) from exc

    def contains(
        self,
        name: str,
    ) -> bool:
        """
        Return whether a tool is registered under the given name.
        """
        if not isinstance(name, str) or not name.strip():
            return False

        return name.strip() in self._tools

    @property
    def names(self) -> tuple[str, ...]:
        """
        Return registered tool names in deterministic registration order.
        """
        return tuple(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)