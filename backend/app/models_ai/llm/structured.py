from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError


StructuredModelT = TypeVar(
    "StructuredModelT",
    bound=BaseModel,
)


class StructuredOutputParser:
    """
    Parse an LLM text response into a validated Pydantic model.

    The parser is provider-independent. Providers only need to return
    normal LLMResponse objects containing JSON text.
    """

    @staticmethod
    def parse(
        text: str,
        response_model: type[StructuredModelT],
    ) -> StructuredModelT:
        """
        Decode JSON text and validate it against the supplied
        Pydantic response model.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Structured LLM output cannot be blank."
            )

        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "LLM response is not valid JSON."
            ) from exc

        try:
            return response_model.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                "LLM response does not match the expected "
                "structured output schema."
            ) from exc