from __future__ import annotations

from copy import deepcopy

from app.models_ai.gateway import LLMGateway
from app.models_ai.llm.schemas import (
    LLMMessage,
    LLMRequest,
    LLMResponse,
)
from app.rag.context import RAGContextAssembler
from app.rag.schemas import (
    RAGContext,
    RAGQuery,
)
from app.rag.service import RAGRetrievalService


class RAGLLMIntegrationService:
    """
    Integrate the existing RAG pipeline with the LLM gateway.

    Responsibilities:
    - execute RAG retrieval;
    - assemble deterministic bounded RAG context;
    - compose the context into an existing LLM request;
    - invoke the provider-independent LLM gateway.

    Non-responsibilities:
    - embedding;
    - vector retrieval;
    - context ranking;
    - context truncation;
    - provider-specific LLM behavior;
    - LLM retry policy;
    - structured-output parsing.
    """

    CONTEXT_INSTRUCTION = (
        "Use the retrieved knowledge context below when answering "
        "the user's request. Treat the retrieved context as reference "
        "material. Do not invent facts that are not supported by the "
        "request or retrieved context.\n\n"
    )

    def __init__(
        self,
        *,
        rag_retrieval_service: RAGRetrievalService,
        context_assembler: RAGContextAssembler,
        llm_gateway: LLMGateway,
    ) -> None:
        if not isinstance(
            rag_retrieval_service,
            RAGRetrievalService,
        ):
            raise TypeError(
                "rag_retrieval_service must be a "
                "RAGRetrievalService."
            )

        if not isinstance(
            context_assembler,
            RAGContextAssembler,
        ):
            raise TypeError(
                "context_assembler must be a "
                "RAGContextAssembler."
            )

        if not isinstance(
            llm_gateway,
            LLMGateway,
        ):
            raise TypeError(
                "llm_gateway must be an LLMGateway."
            )

        self._rag_retrieval_service = rag_retrieval_service
        self._context_assembler = context_assembler
        self._llm_gateway = llm_gateway

    @property
    def rag_retrieval_service(
        self,
    ) -> RAGRetrievalService:
        """Return the configured RAG retrieval service."""
        return self._rag_retrieval_service

    @property
    def context_assembler(
        self,
    ) -> RAGContextAssembler:
        """Return the configured RAG context assembler."""
        return self._context_assembler

    @property
    def llm_gateway(
        self,
    ) -> LLMGateway:
        """Return the configured LLM gateway."""
        return self._llm_gateway

    async def retrieve_context(
        self,
        query: RAGQuery,
    ) -> RAGContext:
        """
        Execute retrieval and deterministic context assembly.

        A5.7 remains responsible for all retrieval and context-limit
        behavior. This method only composes those existing services.
        """

        chunks = await self._rag_retrieval_service.retrieve(
            query,
        )

        return self._context_assembler.assemble(
            query=query,
            chunks=chunks,
        )

    async def generate(
        self,
        *,
        query: RAGQuery,
        request: LLMRequest,
    ) -> LLMResponse:
        """
        Generate an LLM response using retrieved RAG context.

        The supplied LLM request is never mutated. A new request is
        created with a deterministic context message inserted before
        the original conversation messages.
        """

        context = await self.retrieve_context(
            query,
        )

        contextual_request = self._build_contextual_request(
            request=request,
            context=context,
        )

        return await self._llm_gateway.generate(
            contextual_request,
        )

    @classmethod
    def _build_contextual_request(
        cls,
        *,
        request: LLMRequest,
        context: RAGContext,
    ) -> LLMRequest:
        """
        Build a new LLM request containing the assembled RAG context.

        The original request remains immutable and untouched.
        """

        context_message = cls._build_context_message(
            context,
        )

        messages = [
            context_message,
            *[
                message.model_copy(deep=True)
                for message in request.messages
            ],
        ]

        metadata = deepcopy(request.metadata)

        metadata["rag"] = {
            "enabled": True,
            "chunk_count": len(context.chunks),
            "source_chunk_count": context.metadata.get(
                "source_chunk_count",
                len(context.chunks),
            ),
            "context_metadata": deepcopy(
                context.metadata,
            ),
        }

        return request.model_copy(
            update={
                "messages": messages,
                "metadata": metadata,
            },
            deep=True,
        )

    @classmethod
    def _build_context_message(
        cls,
        context: RAGContext,
    ) -> LLMMessage:
        """
        Convert the assembled RAG context into one deterministic
        LLM message.
        """

        if not context.context_text:
            content = (
                cls.CONTEXT_INSTRUCTION
                + "No retrieved knowledge context was found."
            )
        else:
            content = (
                cls.CONTEXT_INSTRUCTION
                + context.context_text
            )

        return LLMMessage(
            role="system",
            content=content,
            metadata={
                "source": "rag",
                "chunk_count": len(context.chunks),
            },
        )