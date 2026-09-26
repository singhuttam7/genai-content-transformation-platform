from __future__ import annotations

import pytest

from app.agents.transformation import (
    ArtifactEnvelope,
    TransformationAgentPort,
    TransformationRequest,
    TransformationResult,
    TransformationStatus,
    TransformationType,
)


class FakeTransformationAgent(TransformationAgentPort):
    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.ADVISORY

    async def execute(
        self,
        request: TransformationRequest,
    ) -> TransformationResult:
        return TransformationResult(
            status=TransformationStatus.COMPLETED,
            artifacts=[
                ArtifactEnvelope(
                    artifact_type="advisory",
                    title="Generated Advisory",
                    content=request.input,
                )
            ],
        )


class TestTransformationAgentPort:
    def test_port_is_abstract(self) -> None:
        assert TransformationAgentPort.__abstractmethods__ == {
            "transformation_type",
            "execute",
        }

    def test_fake_agent_is_valid_transformation_agent(
        self,
    ) -> None:
        agent = FakeTransformationAgent()

        assert isinstance(
            agent,
            TransformationAgentPort,
        )

    def test_transformation_type_is_exposed(self) -> None:
        agent = FakeTransformationAgent()

        assert (
            agent.transformation_type
            == TransformationType.ADVISORY
        )

    @pytest.mark.asyncio
    async def test_execute_accepts_transformation_request(
        self,
    ) -> None:
        agent = FakeTransformationAgent()

        request = TransformationRequest(
            transformation_type=TransformationType.ADVISORY,
            input="Source content",
        )

        result = await agent.execute(request)

        assert isinstance(
            result,
            TransformationResult,
        )

    @pytest.mark.asyncio
    async def test_execute_receives_original_request(
        self,
    ) -> None:
        received: list[TransformationRequest] = []

        class RecordingAgent(TransformationAgentPort):
            @property
            def transformation_type(
                self,
            ) -> TransformationType:
                return TransformationType.ADVISORY

            async def execute(
                self,
                request: TransformationRequest,
            ) -> TransformationResult:
                received.append(request)

                return TransformationResult(
                    status=TransformationStatus.COMPLETED,
                )

        agent = RecordingAgent()

        request = TransformationRequest(
            transformation_type=TransformationType.ADVISORY,
            input="Original content",
            objective="Create advisory",
        )

        await agent.execute(request)

        assert received == [request]
        assert received[0].input == "Original content"
        assert (
            received[0].objective
            == "Create advisory"
        )

    @pytest.mark.asyncio
    async def test_execute_returns_transformation_result(
        self,
    ) -> None:
        agent = FakeTransformationAgent()

        request = TransformationRequest(
            transformation_type=TransformationType.ADVISORY,
            input="Content",
        )

        result = await agent.execute(request)

        assert result.status == TransformationStatus.COMPLETED
        assert len(result.artifacts) == 1
        assert (
            result.artifacts[0].artifact_type
            == "advisory"
        )

    def test_port_is_provider_independent(self) -> None:
        module_name = TransformationAgentPort.__module__

        assert module_name == (
            "app.agents.transformation.port"
        )

    def test_port_does_not_depend_on_langgraph(self) -> None:
        source_module = TransformationAgentPort.__module__

        assert "langgraph" not in source_module

    def test_port_does_not_depend_on_sqlalchemy(self) -> None:
        source_module = TransformationAgentPort.__module__

        assert "sqlalchemy" not in source_module

    def test_transformation_agent_port_is_not_generic_agent_port(
        self,
    ) -> None:
        from app.agents.base import AgentPort

        assert not issubclass(
            TransformationAgentPort,
            AgentPort,
        )

    @pytest.mark.asyncio
    async def test_failed_transformation_result_uses_contract(
        self,
    ) -> None:
        class FailingAgent(TransformationAgentPort):
            @property
            def transformation_type(
                self,
            ) -> TransformationType:
                return TransformationType.VIDEO

            async def execute(
                self,
                request: TransformationRequest,
            ) -> TransformationResult:
                return TransformationResult(
                    status=TransformationStatus.FAILED,
                    error="Transformation failed.",
                )

        agent = FailingAgent()

        result = await agent.execute(
            TransformationRequest(
                transformation_type=TransformationType.VIDEO,
                input="Video source",
            )
        )

        assert result.status == TransformationStatus.FAILED
        assert result.error == "Transformation failed."