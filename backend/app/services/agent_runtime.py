from __future__ import annotations

from app.agents.base import (
    AgentIntegrationService,
    AgentRAGContextIntegrator,
    AgentRegistry,
)
from app.agents.transformation import (
    AdvisoryTransformationAgent,
    ExecutiveSummaryTransformationAgent,
    InfographicTransformationAgent,
    PresentationTransformationAgent,
    SocialMediaTransformationAgent,
    VideoTransformationAgent,
)
from app.agents.transformation.agent_port_adapter import (
    TransformationAgentPortAdapter,
)
from app.intelligence.retrieval.context_retriever import RAGLLMIntegrationService
from app.models_ai.gateway import LLMGateway
from app.models_ai.llm import LLMModelInfo
from app.rag.context import RAGContextAssembler
from app.rag.service import RAGRetrievalService


TRANSFORMATION_AGENT_NAMES = {
    "advisory": "AdvisoryTransformationAgent",
    "executive_summary": "ExecutiveSummaryTransformationAgent",
    "social_media": "SocialMediaTransformationAgent",
    "infographic": "InfographicTransformationAgent",
    "presentation": "PresentationTransformationAgent",
    "video": "VideoTransformationAgent",
}


def create_agent_registry(
    *,
    rag_retrieval_service: RAGRetrievalService,
    llm_gateway: LLMGateway,
    model: LLMModelInfo,
) -> AgentRegistry:
    if not isinstance(rag_retrieval_service, RAGRetrievalService):
        raise TypeError("rag_retrieval_service must be a RAGRetrievalService.")

    if not isinstance(llm_gateway, LLMGateway):
        raise TypeError("llm_gateway must be an LLMGateway.")

    if not isinstance(model, LLMModelInfo):
        raise TypeError("model must be an LLMModelInfo.")

    rag_llm_service = RAGLLMIntegrationService(
        rag_retrieval_service=rag_retrieval_service,
        context_assembler=RAGContextAssembler(),
        llm_gateway=llm_gateway,
    )

    rag_integrator = AgentRAGContextIntegrator(
        rag_service=rag_llm_service,
    )

    integration_service = AgentIntegrationService(
        rag_integrator=rag_integrator,
        llm_gateway=llm_gateway,
    )

    registry = AgentRegistry()

    transformation_agents = (
        (
            TRANSFORMATION_AGENT_NAMES["advisory"],
            AdvisoryTransformationAgent,
        ),
        (
            TRANSFORMATION_AGENT_NAMES["executive_summary"],
            ExecutiveSummaryTransformationAgent,
        ),
        (
            TRANSFORMATION_AGENT_NAMES["social_media"],
            SocialMediaTransformationAgent,
        ),
        (
            TRANSFORMATION_AGENT_NAMES["infographic"],
            InfographicTransformationAgent,
        ),
        (
            TRANSFORMATION_AGENT_NAMES["presentation"],
            PresentationTransformationAgent,
        ),
        (
            TRANSFORMATION_AGENT_NAMES["video"],
            VideoTransformationAgent,
        ),
    )

    for name, agent_class in transformation_agents:
        agent = agent_class(
            integration_service=integration_service,
            model=model,
        )

        registry.register(
            name,
            TransformationAgentPortAdapter(agent),
        )

    return registry