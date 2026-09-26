from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from app.agents.base import (
    AgentPort,
    AgentRegistry,
    AgentResult,
    AgentStatus,
)


class FakeAgent(AgentPort):
    async def execute(self, request):
        return AgentResult(
            status=AgentStatus.COMPLETED,
            output="done",
        )


def test_register_and_resolve_agent() -> None:
    registry = AgentRegistry()
    agent = FakeAgent()

    registry.register("test-agent", agent)

    assert registry.resolve("test-agent") is agent


def test_registry_strips_agent_name() -> None:
    registry = AgentRegistry()
    agent = FakeAgent()

    registry.register("  test-agent  ", agent)

    assert registry.contains("test-agent")
    assert registry.resolve("test-agent") is agent


def test_duplicate_registration_is_rejected() -> None:
    registry = AgentRegistry()
    agent = FakeAgent()

    registry.register("test-agent", agent)

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        registry.register("test-agent", agent)


def test_invalid_name_is_rejected() -> None:
    registry = AgentRegistry()

    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        registry.register("", FakeAgent())


def test_invalid_agent_is_rejected() -> None:
    registry = AgentRegistry()

    with pytest.raises(
        TypeError,
        match="AgentPort",
    ):
        registry.register(
            "test-agent",
            object(),  # type: ignore[arg-type]
        )


def test_unknown_agent_is_rejected() -> None:
    registry = AgentRegistry()

    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        registry.resolve("missing")


def test_contains_unknown_agent() -> None:
    registry = AgentRegistry()

    assert registry.contains("missing") is False


def test_names_are_deterministic() -> None:
    registry = AgentRegistry()

    registry.register("first", FakeAgent())
    registry.register("second", FakeAgent())

    assert registry.names == (
        "first",
        "second",
    )


def test_registry_length() -> None:
    registry = AgentRegistry()

    assert len(registry) == 0

    registry.register(
        "agent",
        FakeAgent(),
    )

    assert len(registry) == 1


def test_registry_iteration() -> None:
    registry = AgentRegistry()

    registry.register(
        "first",
        FakeAgent(),
    )
    registry.register(
        "second",
        FakeAgent(),
    )

    assert list(registry) == [
        "first",
        "second",
    ]