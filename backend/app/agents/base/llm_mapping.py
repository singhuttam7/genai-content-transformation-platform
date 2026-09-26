from __future__ import annotations

from app.agents.base.execution_context import AgentExecutionContext
from app.models_ai.llm import (
    LLMMessage,
    LLMModelInfo,
    LLMRequest,
)


class AgentLLMRequestMapper:
    """
    Convert an AgentExecutionContext into a provider-independent LLMRequest.

    The mapper belongs to the agent boundary and does not know about
    concrete providers such as Groq, OpenAI, Gemini, or Mistral.
    """

    @staticmethod
    def to_request(
        context: AgentExecutionContext,
        model: LLMModelInfo,
    ) -> LLMRequest:
        """
        Convert an agent execution context into an LLM request.

        The caller supplies the model explicitly so that model selection
        remains an orchestration/configuration responsibility rather than
        being hidden inside the mapper.
        """

        if not isinstance(context, AgentExecutionContext):
            raise TypeError(
                "context must be an AgentExecutionContext."
            )

        if not isinstance(model, LLMModelInfo):
            raise TypeError(
                "model must be an LLMModelInfo."
            )

        request = context.request

        task = request.task.strip()

        if not task:
            raise ValueError(
                "Agent task must not be empty."
            )

        messages: list[LLMMessage] = []

        if context.rag_context is not None:
            rag_context = context.rag_context.strip()

            if rag_context:
                messages.append(
                    LLMMessage(
                        role="system",
                        content=(
                            "Use the following retrieved knowledge "
                            "context when relevant.\n\n"
                            f"{rag_context}"
                        ),
                    )
                )

        input_text = AgentLLMRequestMapper._serialize_input(
            request.input,
        )

        messages.append(
            LLMMessage(
                role="user",
                content=(
                    f"Task: {task}\n\n"
                    f"Input:\n{input_text}"
                ),
            )
        )

        metadata = {
            **request.metadata,
            **context.metadata,
            "agent_task": task,
        }

        return LLMRequest(
            model=model,
            messages=messages,
            metadata=metadata,
        )

    @staticmethod
    def _serialize_input(value: object) -> str:
        """
        Convert agent input into deterministic text suitable for an LLM
        message.
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