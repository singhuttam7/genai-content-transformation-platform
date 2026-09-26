from __future__ import annotations

import pytest

from app.tools.base import (
    ToolPort,
    ToolRegistry,
    ToolRequest,
    ToolResult,
    ToolStatus,
)


class FakeTool(ToolPort):
    async def execute(
        self,
        request: ToolRequest,
    ) -> ToolResult:
        return ToolResult(
            status=ToolStatus.COMPLETED,
            output=request.input,
        )


class AnotherFakeTool(ToolPort):
    async def execute(
        self,
        request: ToolRequest,
    ) -> ToolResult:
        return ToolResult(
            status=ToolStatus.COMPLETED,
            output="another",
        )


class TestToolRegistry:
    def test_registry_starts_empty(self) -> None:
        registry = ToolRegistry()

        assert len(registry) == 0
        assert registry.names == ()

    def test_registers_tool(self) -> None:
        registry = ToolRegistry()
        tool = FakeTool()

        registry.register(
            "search",
            tool,
        )

        assert len(registry) == 1
        assert registry.contains("search")
        assert registry.names == ("search",)

    def test_resolve_returns_registered_tool(self) -> None:
        registry = ToolRegistry()
        tool = FakeTool()

        registry.register(
            "search",
            tool,
        )

        resolved = registry.resolve("search")

        assert resolved is tool
        assert isinstance(resolved, ToolPort)

    def test_registration_preserves_registration_order(self) -> None:
        registry = ToolRegistry()

        first = FakeTool()
        second = AnotherFakeTool()

        registry.register("search", first)
        registry.register("browser", second)

        assert registry.names == (
            "search",
            "browser",
        )

    def test_duplicate_registration_is_rejected(self) -> None:
        registry = ToolRegistry()

        registry.register(
            "search",
            FakeTool(),
        )

        with pytest.raises(
            ValueError,
            match="Tool 'search' is already registered.",
        ):
            registry.register(
                "search",
                AnotherFakeTool(),
            )

    def test_unknown_tool_is_rejected(self) -> None:
        registry = ToolRegistry()

        with pytest.raises(
            KeyError,
            match="Tool 'search' is not registered.",
        ):
            registry.resolve("search")

    def test_empty_tool_name_registration_is_rejected(self) -> None:
        registry = ToolRegistry()

        with pytest.raises(
            ValueError,
            match="Tool name must be a non-empty string.",
        ):
            registry.register(
                "",
                FakeTool(),
            )

    def test_whitespace_tool_name_registration_is_rejected(
        self,
    ) -> None:
        registry = ToolRegistry()

        with pytest.raises(
            ValueError,
            match="Tool name must be a non-empty string.",
        ):
            registry.register(
                "   ",
                FakeTool(),
            )

    def test_tool_name_is_canonicalized(self) -> None:
        registry = ToolRegistry()
        tool = FakeTool()

        registry.register(
            "  search  ",
            tool,
        )

        assert registry.contains("search")
        assert registry.resolve("search") is tool
        assert registry.names == ("search",)

    def test_duplicate_is_checked_after_canonicalization(
        self,
    ) -> None:
        registry = ToolRegistry()

        registry.register(
            "search",
            FakeTool(),
        )

        with pytest.raises(
            ValueError,
            match="Tool 'search' is already registered.",
        ):
            registry.register(
                "  search  ",
                AnotherFakeTool(),
            )

    def test_invalid_tool_type_is_rejected(self) -> None:
        registry = ToolRegistry()

        with pytest.raises(
            TypeError,
            match="tool must be a ToolPort.",
        ):
            registry.register(
                "search",
                object(),  # type: ignore[arg-type]
            )

    def test_contains_unknown_tool_returns_false(self) -> None:
        registry = ToolRegistry()

        assert registry.contains("search") is False

    def test_contains_empty_name_returns_false(self) -> None:
        registry = ToolRegistry()

        assert registry.contains("") is False
        assert registry.contains("   ") is False

    def test_contains_canonicalizes_name(self) -> None:
        registry = ToolRegistry()

        registry.register(
            "search",
            FakeTool(),
        )

        assert registry.contains("  search  ") is True

    def test_resolve_empty_name_is_rejected(self) -> None:
        registry = ToolRegistry()

        with pytest.raises(
            ValueError,
            match="Tool name must be a non-empty string.",
        ):
            registry.resolve("")

    def test_resolve_whitespace_name_is_rejected(self) -> None:
        registry = ToolRegistry()

        with pytest.raises(
            ValueError,
            match="Tool name must be a non-empty string.",
        ):
            registry.resolve("   ")

    def test_resolve_canonicalizes_name(self) -> None:
        registry = ToolRegistry()
        tool = FakeTool()

        registry.register(
            "search",
            tool,
        )

        assert registry.resolve("  search  ") is tool

    def test_registry_iteration_is_deterministic(self) -> None:
        registry = ToolRegistry()

        registry.register("search", FakeTool())
        registry.register("browser", AnotherFakeTool())

        assert list(registry) == [
            "search",
            "browser",
        ]

    def test_registry_does_not_execute_tools(self) -> None:
        registry = ToolRegistry()
        tool = FakeTool()

        registry.register(
            "search",
            tool,
        )

        resolved = registry.resolve("search")

        assert resolved is tool
        assert registry.names == ("search",)