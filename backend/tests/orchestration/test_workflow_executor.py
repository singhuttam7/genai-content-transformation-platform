from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.agents.base import (
    AgentPort,
    AgentRegistry,
    AgentRequest,
    AgentResult,
     AgentState,
    AgentStatus,
)
from app.orchestration.contracts import (
    WorkflowRequest,
    WorkflowStep,
    WorkflowResult,
)
from app.orchestration.failure_policy import WorkflowFailurePolicy
from app.orchestration.execution.executor import WorkflowExecutor


class FakeAgent(AgentPort):
    """
    Test implementation of AgentPort.

    The execute() method is implemented at class level so that the
    AgentPort ABC contract is genuinely satisfied. An AsyncMock is then
    installed on the instance so individual tests can inspect calls and
    inject failures.
    """

    def __init__(
        self,
        *,
        output: object,
        status: AgentStatus = AgentStatus.COMPLETED,
        error: str | None = None,
    ) -> None:
        self._result = AgentResult(
            status=status,
            output=output,
            error=error,
        )

        self.execute = AsyncMock(
            side_effect=self._execute_impl,
        )

    async def execute(
        self,
        request: AgentRequest,
    ) -> AgentResult:
        """
        Concrete implementation required by AgentPort.

        Instances replace this method with AsyncMock in __init__ so tests
        can inspect execution calls.
        """
        return await self._execute_impl(request)

    async def _execute_impl(
        self,
        request: AgentRequest,
    ) -> AgentResult:
        return self._result


def make_executor(
    *agents: tuple[str, AgentPort],
) -> WorkflowExecutor:
    registry = AgentRegistry()

    for name, agent in agents:
        registry.register(name, agent)

    return WorkflowExecutor(
        agent_registry=registry,
    )


@pytest.mark.asyncio
async def test_single_agent_workflow() -> None:
    agent = FakeAgent(
        output="result",
    )

    executor = make_executor(
        ("agent", agent),
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Process input",
            )
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.COMPLETED
    assert result.output == "result"
    assert len(result.state.history) == 1
    assert result.state.history[0].agent_name == "agent"


@pytest.mark.asyncio
async def test_agents_execute_in_declared_order() -> None:
    first = FakeAgent(
        output="first-output",
    )
    second = FakeAgent(
        output="second-output",
    )

    executor = make_executor(
        ("first", first),
        ("second", second),
    )

    request = WorkflowRequest(
        input="initial-input",
        steps=[
            WorkflowStep(
                agent_name="first",
                task="First task",
            ),
            WorkflowStep(
                agent_name="second",
                task="Second task",
            ),
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.COMPLETED
    assert result.output == "second-output"

    first_request = first.execute.await_args.args[0]
    second_request = second.execute.await_args.args[0]

    assert first_request.input == "initial-input"
    assert second_request.input == "first-output"


@pytest.mark.asyncio
async def test_explicit_step_input_overrides_previous_output() -> None:
    first = FakeAgent(
        output="first-output",
    )
    second = FakeAgent(
        output="second-output",
    )

    executor = make_executor(
        ("first", first),
        ("second", second),
    )

    request = WorkflowRequest(
        input="initial",
        steps=[
            WorkflowStep(
                agent_name="first",
                task="First",
            ),
            WorkflowStep(
                agent_name="second",
                task="Second",
                input="explicit-input",
            ),
        ],
    )

    await executor.execute(request)

    second_request = second.execute.await_args.args[0]

    assert second_request.input == "explicit-input"


@pytest.mark.asyncio
async def test_workflow_metadata_is_propagated() -> None:
    agent = FakeAgent(
        output="result",
    )

    executor = make_executor(
        ("agent", agent),
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Process",
                metadata={
                    "step_key": "step-value",
                },
            )
        ],
        metadata={
            "request_id": "request-123",
        },
    )

    await executor.execute(request)

    agent_request = agent.execute.await_args.args[0]

    assert agent_request.metadata["request_id"] == "request-123"
    assert agent_request.metadata["step_key"] == "step-value"
    assert agent_request.metadata["workflow_agent"] == "agent"


@pytest.mark.asyncio
async def test_workflow_records_each_successful_step() -> None:
    first = FakeAgent(
        output="first",
    )
    second = FakeAgent(
        output="second",
    )

    executor = make_executor(
        ("first", first),
        ("second", second),
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="first",
                task="First",
            ),
            WorkflowStep(
                agent_name="second",
                task="Second",
            ),
        ],
    )

    result = await executor.execute(request)

    assert len(result.state.history) == 2
    assert result.state.history[0].status == AgentStatus.COMPLETED
    assert result.state.history[1].status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_failed_agent_stops_workflow() -> None:
    failing = FakeAgent(
        output=None,
        status=AgentStatus.FAILED,
        error="agent failed",
    )
    next_agent = FakeAgent(
        output="should-not-run",
    )

    executor = make_executor(
        ("failing", failing),
        ("next", next_agent),
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="failing",
                task="Fail",
            ),
            WorkflowStep(
                agent_name="next",
                task="Should not execute",
            ),
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.FAILED
    assert result.error == "agent failed"
    assert len(result.state.history) == 1
    next_agent.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_exception_fails_workflow() -> None:
    agent = FakeAgent(
        output="unused",
    )

    agent.execute.side_effect = RuntimeError(
        "unexpected failure",
    )

    executor = make_executor(
        ("agent", agent),
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Process",
            )
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.FAILED
    assert result.error == "unexpected failure"
    assert len(result.state.history) == 1
    assert result.state.history[0].status == AgentStatus.FAILED


@pytest.mark.asyncio
async def test_missing_agent_fails_before_execution() -> None:
    executor = make_executor()

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="missing",
                task="Process",
            )
        ],
    )

    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        await executor.execute(request)


def test_constructor_rejects_invalid_registry() -> None:
    with pytest.raises(
        TypeError,
        match="AgentRegistry",
    ):
        WorkflowExecutor(
            agent_registry=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_completed_workflow_clears_current_agent() -> None:
    agent = FakeAgent(
        output="result",
    )

    executor = make_executor(
        ("agent", agent),
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Process",
            )
        ],
    )

    result = await executor.execute(request)

    assert result.state.current_agent is None
    assert result.state.status == AgentStatus.COMPLETED


@pytest.mark.asyncio
async def test_failed_agent_is_not_retried_by_default() -> None:
    agent = FakeAgent(
        output=None,
        status=AgentStatus.FAILED,
        error="failed",
    )

    executor = make_executor(
        ("agent", agent),
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Process",
            )
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.FAILED
    assert agent.execute.await_count == 1
    assert len(result.state.history) == 1
    assert result.state.history[0].metadata["attempt"] == 1


@pytest.mark.asyncio
async def test_failed_agent_is_retried_when_policy_allows() -> None:
    agent = FakeAgent(
        output="recovered",
    )

    agent.execute.side_effect = [
        AgentResult(
            status=AgentStatus.FAILED,
            error="temporary failure",
        ),
        AgentResult(
            status=AgentStatus.COMPLETED,
            output="recovered",
        ),
    ]

    executor = WorkflowExecutor(
        agent_registry=AgentRegistry(),
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=2,
        ),
    )

    executor.agent_registry.register(
        "agent",
        agent,
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Process",
            )
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.COMPLETED
    assert result.output == "recovered"
    assert agent.execute.await_count == 2
    assert len(result.state.history) == 2

    assert result.state.history[0].status == AgentStatus.FAILED
    assert result.state.history[0].metadata["attempt"] == 1

    assert result.state.history[1].status == AgentStatus.COMPLETED
    assert result.state.history[1].metadata["attempt"] == 2


@pytest.mark.asyncio
async def test_retry_exhaustion_fails_workflow() -> None:
    agent = FakeAgent(
        output=None,
        status=AgentStatus.FAILED,
        error="persistent failure",
    )

    executor = WorkflowExecutor(
        agent_registry=AgentRegistry(),
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=3,
        ),
    )

    executor.agent_registry.register(
        "agent",
        agent,
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Process",
            )
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.FAILED
    assert result.error == "persistent failure"
    assert agent.execute.await_count == 3
    assert len(result.state.history) == 3

    assert [
        step.metadata["attempt"]
        for step in result.state.history
    ] == [1, 2, 3]


@pytest.mark.asyncio
async def test_successful_agent_is_never_retried() -> None:
    agent = FakeAgent(
        output="success",
    )

    executor = WorkflowExecutor(
        agent_registry=AgentRegistry(),
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=5,
        ),
    )

    executor.agent_registry.register(
        "agent",
        agent,
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Process",
            )
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.COMPLETED
    assert agent.execute.await_count == 1
    assert len(result.state.history) == 1


@pytest.mark.asyncio
async def test_retry_metadata_is_passed_to_agent() -> None:
    agent = FakeAgent(
        output="success",
    )

    agent.execute.side_effect = [
        AgentResult(
            status=AgentStatus.FAILED,
            error="temporary",
        ),
        AgentResult(
            status=AgentStatus.COMPLETED,
            output="success",
        ),
    ]

    executor = WorkflowExecutor(
        agent_registry=AgentRegistry(),
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=2,
        ),
    )

    executor.agent_registry.register(
        "agent",
        agent,
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Process",
            )
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.COMPLETED

    first_request = agent.execute.await_args_list[0].args[0]
    second_request = agent.execute.await_args_list[1].args[0]

    assert first_request.metadata["workflow_attempt"] == 1
    assert second_request.metadata["workflow_attempt"] == 2

    assert first_request.metadata["workflow_max_attempts"] == 2
    assert second_request.metadata["workflow_max_attempts"] == 2


def test_executor_rejects_invalid_failure_policy() -> None:
    with pytest.raises(
        TypeError,
        match="WorkflowFailurePolicy",
    ):
        WorkflowExecutor(
            agent_registry=AgentRegistry(),
            failure_policy=object(),  # type: ignore[arg-type]
        )


def test_executor_uses_default_failure_policy() -> None:
    executor = WorkflowExecutor(
        agent_registry=AgentRegistry(),
    )

    assert executor.failure_policy == WorkflowFailurePolicy()   

@pytest.mark.asyncio
async def test_recovered_agent_output_feeds_next_agent() -> None:
    recovering_agent = FakeAgent(
        output="unused",
    )

    recovering_agent.execute.side_effect = [
        AgentResult(
            status=AgentStatus.FAILED,
            output=None,
            error="temporary failure",
        ),
        AgentResult(
            status=AgentStatus.COMPLETED,
            output="recovered-output",
        ),
    ]

    next_agent = FakeAgent(
        output="final-output",
    )

    registry = AgentRegistry()
    registry.register(
        "recovering",
        recovering_agent,
    )
    registry.register(
        "next",
        next_agent,
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=2,
        ),
    )

    request = WorkflowRequest(
        input="initial-input",
        steps=[
            WorkflowStep(
                agent_name="recovering",
                task="Recover the input",
            ),
            WorkflowStep(
                agent_name="next",
                task="Process recovered output",
            ),
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.COMPLETED
    assert result.output == "final-output"

    next_request = next_agent.execute.await_args.args[0]

    assert next_request.input == "recovered-output"

    assert len(result.state.history) == 3

    assert result.state.history[0].status == AgentStatus.FAILED
    assert result.state.history[0].metadata["attempt"] == 1

    assert result.state.history[1].status == AgentStatus.COMPLETED
    assert result.state.history[1].metadata["attempt"] == 2

    assert result.state.history[2].status == AgentStatus.COMPLETED
    assert result.state.history[2].agent_name == "next"


@pytest.mark.asyncio
async def test_recovery_preserves_workflow_metadata() -> None:
    recovering_agent = FakeAgent(
        output="unused",
    )

    recovering_agent.execute.side_effect = [
        AgentResult(
            status=AgentStatus.FAILED,
            error="temporary",
        ),
        AgentResult(
            status=AgentStatus.COMPLETED,
            output="recovered",
        ),
    ]

    registry = AgentRegistry()
    registry.register(
        "agent",
        recovering_agent,
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=2,
        ),
    )

    request = WorkflowRequest(
        input="input",
        metadata={
            "request_id": "request-123",
            "source": "test",
        },
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Recover",
                metadata={
                    "step_id": "step-123",
                },
            )
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.COMPLETED

    first_request = (
        recovering_agent.execute.await_args_list[0].args[0]
    )
    second_request = (
        recovering_agent.execute.await_args_list[1].args[0]
    )

    for agent_request in (
        first_request,
        second_request,
    ):
        assert agent_request.metadata["request_id"] == "request-123"
        assert agent_request.metadata["source"] == "test"
        assert agent_request.metadata["step_id"] == "step-123"
        assert agent_request.metadata["workflow_agent"] == "agent"

    assert first_request.metadata["workflow_attempt"] == 1
    assert second_request.metadata["workflow_attempt"] == 2


@pytest.mark.asyncio
async def test_recovery_does_not_repeat_successful_attempt() -> None:
    agent = FakeAgent(
        output="success",
    )

    agent.execute.side_effect = [
        AgentResult(
            status=AgentStatus.FAILED,
            error="temporary",
        ),
        AgentResult(
            status=AgentStatus.COMPLETED,
            output="success",
        ),
        AgentResult(
            status=AgentStatus.COMPLETED,
            output="should-not-run",
        ),
    ]

    registry = AgentRegistry()
    registry.register(
        "agent",
        agent,
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=3,
        ),
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Recover",
            )
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.COMPLETED
    assert result.output == "success"
    assert agent.execute.await_count == 2
    assert len(result.state.history) == 2


@pytest.mark.asyncio
async def test_later_agent_failure_stops_after_earlier_recovery() -> None:
    recovering_agent = FakeAgent(
        output="unused",
    )

    recovering_agent.execute.side_effect = [
        AgentResult(
            status=AgentStatus.FAILED,
            error="temporary",
        ),
        AgentResult(
            status=AgentStatus.COMPLETED,
            output="recovered",
        ),
    ]

    failing_agent = FakeAgent(
        output=None,
        status=AgentStatus.FAILED,
        error="later failure",
    )

    never_run_agent = FakeAgent(
        output="should-not-run",
    )

    registry = AgentRegistry()
    registry.register(
        "recovering",
        recovering_agent,
    )
    registry.register(
        "failing",
        failing_agent,
    )
    registry.register(
        "never",
        never_run_agent,
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=2,
        ),
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="recovering",
                task="Recover",
            ),
            WorkflowStep(
                agent_name="failing",
                task="Fail later",
            ),
            WorkflowStep(
                agent_name="never",
                task="Must not execute",
            ),
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.FAILED
    assert result.error == "later failure"

    assert recovering_agent.execute.await_count == 2
    assert failing_agent.execute.await_count == 2
    assert never_run_agent.execute.await_count == 0

    assert len(result.state.history) == 4

    assert result.state.history[0].agent_name == "recovering"
    assert result.state.history[0].status == AgentStatus.FAILED

    assert result.state.history[1].agent_name == "recovering"
    assert result.state.history[1].status == AgentStatus.COMPLETED

    assert result.state.history[2].agent_name == "failing"
    assert result.state.history[2].status == AgentStatus.FAILED

    assert result.state.history[3].agent_name == "failing"
    assert result.state.history[3].status == AgentStatus.FAILED


@pytest.mark.asyncio
async def test_recovery_attempts_use_same_step_input() -> None:
    agent = FakeAgent(
        output="recovered",
    )

    agent.execute.side_effect = [
        AgentResult(
            status=AgentStatus.FAILED,
            error="temporary",
        ),
        AgentResult(
            status=AgentStatus.COMPLETED,
            output="recovered",
        ),
    ]

    registry = AgentRegistry()
    registry.register(
        "agent",
        agent,
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=2,
        ),
    )

    request = WorkflowRequest(
        input="original-input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Recover",
            )
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.COMPLETED

    first_request = agent.execute.await_args_list[0].args[0]
    second_request = agent.execute.await_args_list[1].args[0]

    assert first_request.input == "original-input"
    assert second_request.input == "original-input"

@pytest.mark.asyncio
async def test_agent_cancellation_marks_workflow_cancelled() -> None:
    agent = FakeAgent(
        output="unused",
    )

    agent.execute.side_effect = asyncio.CancelledError()

    registry = AgentRegistry()
    registry.register(
        "agent",
        agent,
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=3,
        ),
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Process",
            )
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.CANCELLED
    assert result.error is None
    assert result.output is None

    assert agent.execute.await_count == 1

    assert len(result.state.history) == 1

    step = result.state.history[0]

    assert step.status == AgentStatus.CANCELLED
    assert step.metadata["attempt"] == 1
    assert step.metadata["max_attempts"] == 3


@pytest.mark.asyncio
async def test_cancellation_is_never_retried() -> None:
    agent = FakeAgent(
        output="unused",
    )

    agent.execute.side_effect = asyncio.CancelledError()

    registry = AgentRegistry()
    registry.register(
        "agent",
        agent,
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=5,
        ),
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="agent",
                task="Process",
            )
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.CANCELLED
    assert agent.execute.await_count == 1
    assert len(result.state.history) == 1


@pytest.mark.asyncio
async def test_cancellation_stops_following_agents() -> None:
    cancelling_agent = FakeAgent(
        output="unused",
    )

    cancelling_agent.execute.side_effect = (
        asyncio.CancelledError()
    )

    next_agent = FakeAgent(
        output="should-not-run",
    )

    registry = AgentRegistry()
    registry.register(
        "cancelling",
        cancelling_agent,
    )
    registry.register(
        "next",
        next_agent,
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=3,
        ),
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="cancelling",
                task="Cancel",
            ),
            WorkflowStep(
                agent_name="next",
                task="Must not execute",
            ),
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.CANCELLED

    assert cancelling_agent.execute.await_count == 1
    assert next_agent.execute.await_count == 0

    assert len(result.state.history) == 1
    assert result.state.history[0].status == AgentStatus.CANCELLED


@pytest.mark.asyncio
async def test_cancellation_after_previous_success_preserves_history() -> None:
    first_agent = FakeAgent(
        output="first-output",
    )

    cancelling_agent = FakeAgent(
        output="unused",
    )

    cancelling_agent.execute.side_effect = (
        asyncio.CancelledError()
    )

    registry = AgentRegistry()
    registry.register(
        "first",
        first_agent,
    )
    registry.register(
        "cancelling",
        cancelling_agent,
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=3,
        ),
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="first",
                task="First",
            ),
            WorkflowStep(
                agent_name="cancelling",
                task="Cancel",
            ),
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.CANCELLED

    assert len(result.state.history) == 2

    assert result.state.history[0].agent_name == "first"
    assert result.state.history[0].status == AgentStatus.COMPLETED

    assert result.state.history[1].agent_name == "cancelling"
    assert result.state.history[1].status == AgentStatus.CANCELLED

    assert result.state.current_agent == "cancelling"

@pytest.mark.asyncio
async def test_workflow_executes_agents_in_declared_order() -> None:
    execution_order: list[str] = []

    class OrderedAgent(AgentPort):
        def __init__(self, name: str) -> None:
            self.name = name

        async def execute(
            self,
            request: AgentRequest,
        ) -> AgentResult:
            execution_order.append(self.name)

            return AgentResult(
                status=AgentStatus.COMPLETED,
                output=f"{self.name}-output",
            )

    registry = AgentRegistry()
    registry.register(
        "first",
        OrderedAgent("first"),
    )
    registry.register(
        "second",
        OrderedAgent("second"),
    )
    registry.register(
        "third",
        OrderedAgent("third"),
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
    )

    request = WorkflowRequest(
        input="initial",
        steps=[
            WorkflowStep(
                agent_name="first",
                task="first task",
            ),
            WorkflowStep(
                agent_name="second",
                task="second task",
            ),
            WorkflowStep(
                agent_name="third",
                task="third task",
            ),
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.COMPLETED
    assert execution_order == [
        "first",
        "second",
        "third",
    ]

@pytest.mark.asyncio
async def test_retry_attempts_use_the_same_step_input() -> None:
    received_inputs: list[object] = []

    class RetryAgent(AgentPort):
        def __init__(self) -> None:
            self.calls = 0

        async def execute(
            self,
            request: AgentRequest,
        ) -> AgentResult:
            self.calls += 1
            received_inputs.append(request.input)

            if self.calls == 1:
                return AgentResult(
                    status=AgentStatus.FAILED,
                    error="temporary failure",
                )

            return AgentResult(
                status=AgentStatus.COMPLETED,
                output="recovered",
            )

    agent = RetryAgent()

    registry = AgentRegistry()
    registry.register(
        "retry",
        agent,
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=2,
        ),
    )

    request = WorkflowRequest(
        input={
            "document": "source text",
            "version": 1,
        },
        steps=[
            WorkflowStep(
                agent_name="retry",
                task="process document",
            ),
        ],
    )

    result = await executor.execute(request)

    assert result.status == AgentStatus.COMPLETED
    assert received_inputs == [
        {
            "document": "source text",
            "version": 1,
        },
        {
            "document": "source text",
            "version": 1,
        },
    ]

@pytest.mark.asyncio
async def test_attempt_metadata_is_deterministic() -> None:
    requests: list[AgentRequest] = []

    class RetryAgent(AgentPort):
        def __init__(self) -> None:
            self.calls = 0

        async def execute(
            self,
            request: AgentRequest,
        ) -> AgentResult:
            self.calls += 1
            requests.append(request)

            if self.calls == 1:
                return AgentResult(
                    status=AgentStatus.FAILED,
                    error="temporary failure",
                )

            return AgentResult(
                status=AgentStatus.COMPLETED,
                output="success",
            )

    registry = AgentRegistry()
    registry.register(
        "retry",
        RetryAgent(),
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=2,
        ),
    )

    result = await executor.execute(
        WorkflowRequest(
            input="input",
            steps=[
                WorkflowStep(
                    agent_name="retry",
                    task="retry task",
                ),
            ],
        )
    )

    assert result.status == AgentStatus.COMPLETED
    assert len(requests) == 2

    assert requests[0].metadata["workflow_attempt"] == 1
    assert requests[1].metadata["workflow_attempt"] == 2

    assert requests[0].metadata["workflow_max_attempts"] == 2
    assert requests[1].metadata["workflow_max_attempts"] == 2

    assert requests[0].metadata["workflow_agent"] == "retry"
    assert requests[1].metadata["workflow_agent"] == "retry"

    assert (
        requests[0].metadata["workflow_execution_id"]
        == requests[1].metadata["workflow_execution_id"]
    )

@pytest.mark.asyncio
async def test_failed_workflow_preserves_all_attempts_in_history() -> None:
    class AlwaysFailAgent(AgentPort):
        async def execute(
            self,
            request: AgentRequest,
        ) -> AgentResult:
            return AgentResult(
                status=AgentStatus.FAILED,
                error="permanent failure",
            )

    registry = AgentRegistry()
    registry.register(
        "failing",
        AlwaysFailAgent(),
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
        failure_policy=WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=3,
        ),
    )

    result = await executor.execute(
        WorkflowRequest(
            input="input",
            steps=[
                WorkflowStep(
                    agent_name="failing",
                    task="permanent failure",
                ),
            ],
        )
    )

    assert result.status == AgentStatus.FAILED
    assert result.error == "permanent failure"
    assert len(result.state.history) == 3

    assert [
        step.metadata["attempt"]
        for step in result.state.history
    ] == [1, 2, 3]

    assert all(
        step.status == AgentStatus.FAILED
        for step in result.state.history
    )

@pytest.mark.asyncio
async def test_terminal_failure_does_not_execute_following_agents() -> None:
    executed: list[str] = []

    class FailingAgent(AgentPort):
        async def execute(
            self,
            request: AgentRequest,
        ) -> AgentResult:
            executed.append("failing")

            return AgentResult(
                status=AgentStatus.FAILED,
                error="terminal failure",
            )

    class FollowingAgent(AgentPort):
        async def execute(
            self,
            request: AgentRequest,
        ) -> AgentResult:
            executed.append("following")

            return AgentResult(
                status=AgentStatus.COMPLETED,
                output="should never happen",
            )

    registry = AgentRegistry()
    registry.register(
        "failing",
        FailingAgent(),
    )
    registry.register(
        "following",
        FollowingAgent(),
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
    )

    result = await executor.execute(
        WorkflowRequest(
            input="input",
            steps=[
                WorkflowStep(
                    agent_name="failing",
                    task="fail",
                ),
                WorkflowStep(
                    agent_name="following",
                    task="must not execute",
                ),
            ],
        )
    )

    assert result.status == AgentStatus.FAILED
    assert executed == ["failing"]
    assert len(result.state.history) == 1
    assert result.state.history[0].agent_name == "failing"

@pytest.mark.asyncio
async def test_missing_agent_stops_workflow_deterministically() -> None:
    executed: list[str] = []

    class ExistingAgent(AgentPort):
        async def execute(
            self,
            request: AgentRequest,
        ) -> AgentResult:
            executed.append("existing")

            return AgentResult(
                status=AgentStatus.COMPLETED,
                output="existing-output",
            )

    registry = AgentRegistry()
    registry.register(
        "existing",
        ExistingAgent(),
    )

    executor = WorkflowExecutor(
        agent_registry=registry,
    )

    request = WorkflowRequest(
        input="input",
        steps=[
            WorkflowStep(
                agent_name="existing",
                task="run existing",
            ),
            WorkflowStep(
                agent_name="missing",
                task="missing agent",
            ),
        ],
    )

    with pytest.raises(
        KeyError,
        match="Agent 'missing' is not registered.",
    ):
        await executor.execute(request)

    assert executed == ["existing"]
