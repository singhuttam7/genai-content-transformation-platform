from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.agents.base import (
    AgentPort,
    AgentRegistry,
    AgentRequest,
    AgentResult,
    AgentState,
    AgentStatus,
    AgentStep,
)
from app.orchestration.contracts import (
    WorkflowRequest,
    WorkflowResult,
    WorkflowStep,
)
from app.orchestration.failure_policy import WorkflowFailurePolicy


class WorkflowExecutor:
    """
    Execute a workflow as an ordered sequence of agents.

    Responsibilities:
    - resolve agents from AgentRegistry;
    - execute workflow steps sequentially;
    - maintain AgentState;
    - preserve the workflow execution identifier;
    - record every agent execution attempt;
    - apply workflow-level failure/recovery policy;
    - stop deterministically when recovery is exhausted.

    Non-responsibilities:
    - LLM/provider retry;
    - RAG retrieval;
    - tool discovery;
    - workflow planning.
    """

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        failure_policy: WorkflowFailurePolicy | None = None,
    ) -> None:
        if not isinstance(
            agent_registry,
            AgentRegistry,
        ):
            raise TypeError(
                "agent_registry must be an AgentRegistry."
            )

        if failure_policy is not None and not isinstance(
            failure_policy,
            WorkflowFailurePolicy,
        ):
            raise TypeError(
                "failure_policy must be a WorkflowFailurePolicy."
            )

        self._agent_registry = agent_registry
        self._failure_policy = (
            failure_policy
            if failure_policy is not None
            else WorkflowFailurePolicy()
        )

    @property
    def agent_registry(self) -> AgentRegistry:
        return self._agent_registry

    @property
    def failure_policy(self) -> WorkflowFailurePolicy:
        return self._failure_policy

    async def execute(
        self,
        request: WorkflowRequest,
        *,
        execution_id: UUID | None = None,
    ) -> WorkflowResult:
        """
        Execute all workflow steps in declared order.

        When execution_id is supplied by the persistence layer, the same
        identifier is preserved inside AgentState. Otherwise a new
        workflow execution identifier is generated.
        """

        if not isinstance(
            request,
            WorkflowRequest,
        ):
            raise TypeError(
                "request must be a WorkflowRequest."
            )

        resolved_execution_id = (
            execution_id
            if execution_id is not None
            else uuid4()
        )

        state = AgentState(
            execution_id=resolved_execution_id,
            status=AgentStatus.PENDING,
            metadata=dict(request.metadata),
        )

        state.set_status(
            AgentStatus.RUNNING,
        )

        current_input = request.input
        final_output = None

        for step in request.steps:
            agent = self._resolve_agent(
                step.agent_name,
            )

            state.set_current_agent(
                step.agent_name,
            )

            step_input = (
                current_input
                if step.input is None
                else step.input
            )

            result = await self._execute_step_with_recovery(
                agent=agent,
                step=step,
                step_input=step_input,
                request=request,
                state=state,
            )

            if result.status != AgentStatus.COMPLETED:
                state.set_status(
                    result.status,
                )

                # A cancelled workflow must retain the agent that was
                # executing when cancellation occurred. Failed workflows
                # also retain the failed agent for execution diagnostics.
                state.set_current_agent(
                    step.agent_name,
                )

                return WorkflowResult(
                    status=result.status,
                    output=result.output,
                    state=state,
                    metadata={
                        **request.metadata,
                        "failed_agent": (
                            step.agent_name
                            if result.status
                            == AgentStatus.FAILED
                            else None
                        ),
                    },
                    error=result.error,
                )

            final_output = result.output
            current_input = result.output

        state.set_status(
            AgentStatus.COMPLETED,
        )

        state.set_current_agent(
            None,
        )

        return WorkflowResult(
            status=AgentStatus.COMPLETED,
            output=final_output,
            state=state,
            metadata={
                **request.metadata,
                "agent_count": len(request.steps),
            },
        )

    async def _execute_step_with_recovery(
        self,
        *,
        agent: AgentPort,
        step: WorkflowStep,
        step_input: object,
        request: WorkflowRequest,
        state: AgentState,
    ) -> AgentResult:
        """
        Execute one workflow step according to the configured
        workflow-level failure policy.

        Failed attempts may be retried according to the configured
        failure policy.

        Cancellation is treated separately from failure:
        - it is recorded as CANCELLED;
        - it is never retried;
        - it terminates the workflow.
        """

        max_attempts = self._failure_policy.max_attempts

        last_result: AgentResult | None = None

        for attempt in range(
            1,
            max_attempts + 1,
        ):
            agent_request = AgentRequest(
                task=step.task,
                input=step_input,
                metadata={
                    **request.metadata,
                    **step.metadata,
                    "workflow_execution_id": str(
                        state.execution_id,
                    ),
                    "workflow_agent": step.agent_name,
                    "workflow_attempt": attempt,
                    "workflow_max_attempts": max_attempts,
                },
            )

            started_at = datetime.now(
                timezone.utc,
            )

            try:
                result = await agent.execute(
                    agent_request,
                )

            except asyncio.CancelledError:
                completed_at = datetime.now(
                    timezone.utc,
                )

                # Explicitly preserve the agent that was cancelled.
                state.set_current_agent(
                    step.agent_name,
                )

                state.add_step(
                    AgentStep(
                        agent_name=step.agent_name,
                        status=AgentStatus.CANCELLED,
                        started_at=started_at,
                        completed_at=completed_at,
                        metadata={
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "recovery_enabled": (
                                self._failure_policy
                                .retry_failed_agents
                            ),
                        },
                    )
                )

                return AgentResult(
                    status=AgentStatus.CANCELLED,
                    output=None,
                )

            except Exception as exc:
                completed_at = datetime.now(
                    timezone.utc,
                )

                state.add_step(
                    AgentStep(
                        agent_name=step.agent_name,
                        status=AgentStatus.FAILED,
                        started_at=started_at,
                        completed_at=completed_at,
                        error=str(exc),
                        metadata={
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "recovery_enabled": (
                                self._failure_policy
                                .retry_failed_agents
                            ),
                        },
                    )
                )

                last_result = None

                if not self._should_retry(
                    attempt=attempt,
                    status=AgentStatus.FAILED,
                ):
                    return self._exception_result(
                        error=str(exc),
                    )

                continue

            completed_at = datetime.now(
                timezone.utc,
            )

            state.add_step(
                AgentStep(
                    agent_name=step.agent_name,
                    status=result.status,
                    started_at=started_at,
                    completed_at=completed_at,
                    output=result.output,
                    error=result.error,
                    metadata={
                        **result.metadata,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "recovery_enabled": (
                            self._failure_policy
                            .retry_failed_agents
                        ),
                    },
                )
            )

            last_result = result

            if result.status == AgentStatus.COMPLETED:
                return result

            if result.status == AgentStatus.CANCELLED:
                state.set_current_agent(
                    step.agent_name,
                )
                return result

            if not self._should_retry(
                attempt=attempt,
                status=result.status,
            ):
                return result

        if last_result is not None:
            return last_result

        return self._exception_result(
            error="Workflow agent execution failed.",
        )

    def _should_retry(
        self,
        *,
        attempt: int,
        status: AgentStatus,
    ) -> bool:
        if not self._failure_policy.retry_failed_agents:
            return False

        if status != AgentStatus.FAILED:
            return False

        return (
            attempt
            < self._failure_policy.max_attempts
        )

    @staticmethod
    def _exception_result(
        *,
        error: str,
    ) -> AgentResult:
        return AgentResult(
            status=AgentStatus.FAILED,
            output=None,
            error=error,
        )

    def _resolve_agent(
        self,
        agent_name: str,
    ) -> AgentPort:
        return self._agent_registry.resolve(
            agent_name,
        )