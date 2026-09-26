from __future__ import annotations

from collections.abc import Iterator

from app.agents.base.port import AgentPort


class AgentRegistry:
    """
    Registry for discovering and resolving executable agents.

    The registry is deliberately limited to registration and lookup.
    Workflow sequencing and execution remain responsibilities of the
    workflow executor.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentPort] = {}

    def register(
        self,
        name: str,
        agent: AgentPort,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Agent name must be a non-empty string."
            )

        if not isinstance(agent, AgentPort):
            raise TypeError(
                "agent must be an AgentPort."
            )

        canonical_name = name.strip()

        if canonical_name in self._agents:
            raise ValueError(
                f"Agent '{canonical_name}' is already registered."
            )

        self._agents[canonical_name] = agent

    def resolve(
        self,
        name: str,
    ) -> AgentPort:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Agent name must be a non-empty string."
            )

        canonical_name = name.strip()

        try:
            return self._agents[canonical_name]
        except KeyError as exc:
            raise KeyError(
                f"Agent '{canonical_name}' is not registered."
            ) from exc

    def contains(
        self,
        name: str,
    ) -> bool:
        if not isinstance(name, str) or not name.strip():
            return False

        return name.strip() in self._agents

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._agents.keys())

    def __len__(self) -> int:
        return len(self._agents)

    def __iter__(self) -> Iterator[str]:
        return iter(self._agents)