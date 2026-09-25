from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field

from app.models_ai.llm.structured import StructuredOutputParser


class SummaryOutput(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    title: str
    summary: str
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class TestStructuredOutputParser:
    def test_valid_json_is_parsed(self) -> None:
        result = StructuredOutputParser.parse(
            text=(
                '{"title":"RAG",'
                '"summary":"Retrieval augmented generation.",'
                '"confidence":0.95}'
            ),
            response_model=SummaryOutput,
        )

        assert isinstance(result, SummaryOutput)
        assert result.title == "RAG"
        assert result.summary == (
            "Retrieval augmented generation."
        )
        assert result.confidence == 0.95

    def test_json_numbers_are_validated(self) -> None:
        result = StructuredOutputParser.parse(
            text=(
                '{"title":"RAG",'
                '"summary":"Test",'
                '"confidence":0.5}'
            ),
            response_model=SummaryOutput,
        )

        assert result.confidence == 0.5

    def test_invalid_json_is_rejected(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            StructuredOutputParser.parse(
                text='{"title":"RAG",',
                response_model=SummaryOutput,
            )

        assert str(exc_info.value) == (
            "LLM response is not valid JSON."
        )

    def test_blank_output_is_rejected(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            StructuredOutputParser.parse(
                text="   ",
                response_model=SummaryOutput,
            )

        assert str(exc_info.value) == (
            "Structured LLM output cannot be blank."
        )

    def test_missing_required_field_is_rejected(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            StructuredOutputParser.parse(
                text=(
                    '{"title":"RAG",'
                    '"confidence":0.9}'
                ),
                response_model=SummaryOutput,
            )

        assert str(exc_info.value) == (
            "LLM response does not match the expected "
            "structured output schema."
        )

    def test_invalid_field_value_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            StructuredOutputParser.parse(
                text=(
                    '{"title":"RAG",'
                    '"summary":"Test",'
                    '"confidence":2.0}'
                ),
                response_model=SummaryOutput,
            )

    def test_extra_fields_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            StructuredOutputParser.parse(
                text=(
                    '{"title":"RAG",'
                    '"summary":"Test",'
                    '"confidence":0.9,'
                    '"unexpected":"field"}'
                ),
                response_model=SummaryOutput,
            )

    def test_nested_pydantic_model_is_supported(self) -> None:
        class Metadata(BaseModel):
            source: str
            page: int

        class Output(BaseModel):
            title: str
            metadata: Metadata

        result = StructuredOutputParser.parse(
            text=(
                '{"title":"RAG",'
                '"metadata":{"source":"paper.pdf","page":4}}'
            ),
            response_model=Output,
        )

        assert result.title == "RAG"
        assert result.metadata.source == "paper.pdf"
        assert result.metadata.page == 4