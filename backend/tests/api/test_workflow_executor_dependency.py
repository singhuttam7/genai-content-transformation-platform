from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.base import AgentRegistry
from app.api.dependencies import get_workflow_executor
from app.orchestration.execution.executor import WorkflowExecutor
from app.services.agent_runtime import TRANSFORMATION_AGENT_NAMES


@pytest.mark.asyncio
async def test_get_workflow_executor_builds_real_agent_registry():
    db = MagicMock(spec=AsyncSession)

    executor = await get_workflow_executor(db=db)

    assert isinstance(executor, WorkflowExecutor)
    assert isinstance(executor.agent_registry, AgentRegistry)

    for agent_name in TRANSFORMATION_AGENT_NAMES.values():
        assert executor.agent_registry.resolve(agent_name) is not None