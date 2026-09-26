from __future__ import annotations

from app.agents.base.contracts import AgentRequest, AgentResult, AgentStatus
from app.agents.base.execution_context import AgentExecutionContext
from app.agents.base.llm_mapping import AgentLLMRequestMapper
from app.agents.base.rag_integration import AgentRAGContextIntegrator
from app.models_ai.gateway import LLMGateway
from app.models_ai.llm import LLMModelInfo


class AgentIntegrationService:
    """
    Coordinates the agent -> RAG -> LLM execution boundary.

    This service contains orchestration logic only. Concrete LLM providers,
    retrieval implementations, and individual agents remain behind their
    respective abstractions.
    """

    def __init__(
        self,
        *,
        rag_integrator: AgentRAGContextIntegrator,
        llm_gateway: LLMGateway,
    ) -> None:
        if not isinstance(
            rag_integrator,
            AgentRAGContextIntegrator,
        ):
            raise TypeError(
                "rag_integrator must be an AgentRAGContextIntegrator."
            )

        if not isinstance(
            llm_gateway,
            LLMGateway,
        ):
            raise TypeError(
                "llm_gateway must be an LLMGateway."
            )

        self._rag_integrator = rag_integrator
        self._llm_gateway = llm_gateway

    async def execute(
        self,
        request: AgentRequest,
        *,
        model: LLMModelInfo,
        use_rag: bool = True,
    ) -> AgentResult:
        """
        Execute an agent request through the RAG/LLM integration boundary.
        """

        if not isinstance(request, AgentRequest):
            raise TypeError(
                "request must be an AgentRequest."
            )

        if not isinstance(model, LLMModelInfo):
            raise TypeError(
                "model must be an LLMModelInfo."
            )

        context = AgentExecutionContext(
            request=request,
        )

        if use_rag:
            context = await self._rag_integrator.enrich(
                context,
            )

        llm_request = AgentLLMRequestMapper.to_request(
            context,
            model,
        )

        response = await self._llm_gateway.generate(
            llm_request,
        )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=response,
            metadata={
                **context.metadata,
                "rag_enabled": use_rag,
            },
        )