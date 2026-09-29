from unittest.mock import MagicMock

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentRegistry
from app.models_ai.gateway import LLMGateway
from app.models_ai.llm import LLMModelInfo
from app.services.agent_runtime import (
    TRANSFORMATION_AGENT_NAMES,
    create_agent_registry,
)
from app.services.rag_runtime import create_rag_retrieval_service


def test_create_agent_registry():
    session = MagicMock(spec=AsyncSession)

    rag_service = create_rag_retrieval_service(session)

    llm_gateway = MagicMock(spec=LLMGateway)

    model = MagicMock(spec=LLMModelInfo)

    registry = create_agent_registry(
        rag_retrieval_service=rag_service,
        llm_gateway=llm_gateway,
        model=model,
    )

    assert isinstance(registry, AgentRegistry)

    for agent_name in TRANSFORMATION_AGENT_NAMES.values():
        assert registry.resolve(agent_name) is not None