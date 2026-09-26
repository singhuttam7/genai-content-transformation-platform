from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.agents.base.execution_context import AgentExecutionContext
from app.intelligence.retrieval.context_retriever import (
    RAGLLMIntegrationService,
)
from app.rag.schemas import RAGContext, RAGQuery


class AgentRAGContextIntegrator:
    """
    Integrate an agent execution context with the existing RAG pipeline.

    Responsibilities:
    - convert agent task/input into an application-level RAGQuery;
    - delegate retrieval and context assembly to the existing
      RAGLLMIntegrationService;
    - enrich AgentExecutionContext with deterministic RAG context;
    - preserve the original agent request.

    Non-responsibilities:
    - embedding;
    - vector retrieval;
    - context ranking;
    - context truncation;
    - LLM invocation;
    - provider selection;
    - prompt generation.
    """

    def __init__(
        self,
        *,
        rag_service: RAGLLMIntegrationService,
    ) -> None:
        if not isinstance(
            rag_service,
            RAGLLMIntegrationService,
        ):
            raise TypeError(
                "rag_service must be an RAGLLMIntegrationService."
            )

        self._rag_service = rag_service

    @property
    def rag_service(self) -> RAGLLMIntegrationService:
        """Return the configured RAG integration service."""
        return self._rag_service

    async def enrich(
        self,
        context: AgentExecutionContext,
    ) -> AgentExecutionContext:
        """
        Retrieve and attach RAG context to an agent execution context.

        The supplied context is never mutated. A new context instance is
        returned with the assembled RAG context and RAG metadata.
        """

        if not isinstance(
            context,
            AgentExecutionContext,
        ):
            raise TypeError(
                "context must be an AgentExecutionContext."
            )

        query = self.build_query(context)

        rag_context = await self._rag_service.retrieve_context(
            query,
        )

        return self._enrich_context(
            context=context,
            rag_context=rag_context,
        )

    @staticmethod
    def build_query(
        context: AgentExecutionContext,
    ) -> RAGQuery:
        """
        Convert an agent execution context into an application-level
        RAGQuery.

        The agent task and input are deliberately combined into the
        retrieval query so that retrieval has enough semantic context
        to find knowledge relevant to the agent's actual objective.
        """

        if not isinstance(
            context,
            AgentExecutionContext,
        ):
            raise TypeError(
                "context must be an AgentExecutionContext."
            )

        task = context.request.task.strip()

        if not task:
            raise ValueError(
                "Agent task must not be empty."
            )

        input_text = AgentRAGContextIntegrator._serialize_input(
            context.request.input,
        )

        return RAGQuery(
            text=(
                f"Task: {task}\n\n"
                f"Input:\n{input_text}"
            ),
        )

    @staticmethod
    def _serialize_input(
        value: object,
    ) -> str:
        """
        Convert agent input into deterministic retrieval text.
        """

        if isinstance(value, str):
            text = value.strip()

            if not text:
                raise ValueError(
                    "Agent input string must not be empty."
                )

            return text

        if value is None:
            raise ValueError(
                "Agent input must not be None."
            )

        return str(value)

    @staticmethod
    def _enrich_context(
        *,
        context: AgentExecutionContext,
        rag_context: RAGContext,
    ) -> AgentExecutionContext:
        """
        Create an enriched execution context from an assembled
        RAGContext.
        """

        if not isinstance(
            rag_context,
            RAGContext,
        ):
            raise TypeError(
                "rag_context must be a RAGContext."
            )

        metadata: dict[str, Any] = deepcopy(
            context.metadata,
        )

        metadata["rag"] = {
            "enabled": True,
            "chunk_count": len(rag_context.chunks),
            "source_chunk_count": rag_context.metadata.get(
                "source_chunk_count",
                len(rag_context.chunks),
            ),
            "context_metadata": deepcopy(
                rag_context.metadata,
            ),
        }

        return context.model_copy(
            update={
                "rag_context": rag_context.context_text,
                "metadata": metadata,
            },
            deep=True,
        )