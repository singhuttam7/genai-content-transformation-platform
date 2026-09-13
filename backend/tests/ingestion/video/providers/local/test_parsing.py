from __future__ import annotations

import pytest

from app.ingestion.video.providers.local.parsing import (
    VisionResponseParser,
)
from app.ingestion.video.providers.local.runtime import (
    VisionRuntimeResponse,
)


@pytest.fixture
def parser() -> VisionResponseParser:
    return VisionResponseParser()


def build_response(text: str) -> VisionRuntimeResponse:
    return VisionRuntimeResponse(
        text=text,
        model="gemma3:4b",
        runtime_name="ollama",
        metadata={
            "runtime": "ollama",
        },
    )


# ===========================================================================
# Basic validation
# ===========================================================================


def test_parser_requires_response(
    parser: VisionResponseParser,
) -> None:
    with pytest.raises(
        ValueError,
        match="Vision runtime response must be provided",
    ):
        parser.parse(None)  # type: ignore[arg-type]


def test_parser_requires_correct_response_type(
    parser: VisionResponseParser,
) -> None:
    with pytest.raises(
        TypeError,
        match="VisionRuntimeResponse",
    ):
        parser.parse("invalid")  # type: ignore[arg-type]


# ===========================================================================
# JSON object
# ===========================================================================


def test_parser_parses_single_json_observation(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        {
            "timestamp_seconds": 2.5,
            "frame_index": 3,
            "description": "A person standing outdoors.",
            "objects": ["person", "tree"],
            "entities": [],
            "actions": ["standing"],
            "scene": "outdoor",
            "visible_text": null,
            "confidence": 0.91
        }
        """
    )

    result = parser.parse(response)

    assert result.parser_name == "default"
    assert result.source_format == "json"
    assert result.observation_count == 1
    assert not result.warnings

    observation = result.observations[0]

    assert observation.timestamp_seconds == 2.5
    assert observation.frame_index == 3
    assert observation.description == (
        "A person standing outdoors."
    )
    assert observation.objects == [
        "person",
        "tree",
    ]
    assert observation.actions == [
        "standing",
    ]
    assert observation.scene == "outdoor"
    assert observation.confidence == 0.91


# ===========================================================================
# JSON array
# ===========================================================================


def test_parser_parses_json_observation_array(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        [
            {
                "timestamp_seconds": 0,
                "frame_index": 0,
                "description": "A room.",
                "objects": ["chair"],
                "scene": "indoor"
            },
            {
                "timestamp_seconds": 2,
                "frame_index": 1,
                "description": "A person near a table.",
                "objects": ["person", "table"],
                "actions": ["standing"]
            }
        ]
        """
    )

    result = parser.parse(response)

    assert result.source_format == "json"
    assert result.observation_count == 2

    assert result.observations[0].frame_index == 0
    assert result.observations[1].frame_index == 1


# ===========================================================================
# observations wrapper
# ===========================================================================


def test_parser_parses_observations_wrapper(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        {
            "observations": [
                {
                    "timestamp_seconds": 1.5,
                    "frame_index": 2,
                    "description": "A building.",
                    "objects": ["building"]
                }
            ]
        }
        """
    )

    result = parser.parse(response)

    assert result.source_format == "json"
    assert result.observation_count == 1
    assert result.observations[0].description == (
        "A building."
    )


# ===========================================================================
# Empty observations regression
# ===========================================================================


def test_parser_preserves_json_format_for_empty_observations(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        {
            "observations": []
        }
        """
    )

    result = parser.parse(response)

    assert result.source_format == "json"
    assert result.observation_count == 0
    assert result.observations == ()


def test_parser_does_not_use_text_fallback_for_empty_json_observations(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        {
            "observations": []
        }
        """
    )

    result = parser.parse(response)

    assert result.source_format != "text"
    assert result.observation_count == 0
    assert not result.warnings


# ===========================================================================
# Markdown fenced JSON
# ===========================================================================


def test_parser_parses_markdown_json(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        ```json
        {
            "description": "A road with vehicles.",
            "objects": ["road", "car"],
            "scene": "street"
        }
        ```
        """
    )

    result = parser.parse(response)

    assert result.source_format == "json"
    assert result.observation_count == 1

    observation = result.observations[0]

    assert observation.description == (
        "A road with vehicles."
    )
    assert observation.objects == [
        "road",
        "car",
    ]


# ===========================================================================
# Embedded JSON
# ===========================================================================


def test_parser_extracts_embedded_json(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        Here is the visual analysis:

        {
            "description": "A laptop on a desk.",
            "objects": ["laptop", "desk"]
        }

        End of analysis.
        """
    )

    result = parser.parse(response)

    assert result.source_format == "json"
    assert result.observation_count == 1
    assert result.observations[0].description == (
        "A laptop on a desk."
    )


# ===========================================================================
# Plain-text fallback
# ===========================================================================


def test_parser_uses_plain_text_fallback(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        "A person is standing in an outdoor area."
    )

    result = parser.parse(response)

    assert result.source_format == "text"
    assert result.observation_count == 1

    observation = result.observations[0]

    assert observation.description == (
        "A person is standing in an outdoor area."
    )

    assert observation.objects == []
    assert observation.entities == []
    assert observation.actions == []
    assert observation.scene is None
    assert observation.visible_text is None
    assert observation.confidence is None

    assert observation.metadata["fallback"] is True

    assert len(result.warnings) == 1


# ===========================================================================
# Invalid JSON fallback
# ===========================================================================


def test_parser_falls_back_when_json_is_malformed(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        {
            "description": "A building",
            "objects": ["building"
        """
    )

    result = parser.parse(response)

    assert result.source_format == "text"
    assert result.observation_count == 1
    assert result.observations[0].description is not None


# ===========================================================================
# Invalid observation
# ===========================================================================


def test_parser_ignores_non_object_array_items(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        [
            "invalid",
            {
                "description": "A valid observation."
            }
        ]
        """
    )

    result = parser.parse(response)

    assert result.source_format == "json"
    assert result.observation_count == 1

    assert any(
        "was ignored" in warning
        for warning in result.warnings
    )


def test_parser_preserves_json_for_payload_without_observation(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        {
            "status": "completed",
            "message": "Analysis complete."
        }
        """
    )

    result = parser.parse(response)

    assert result.source_format == "json"
    assert result.observation_count == 0
    assert result.observations == ()
# ===========================================================================
# Type normalization
# ===========================================================================


def test_parser_normalizes_invalid_optional_types(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        {
            "timestamp_seconds": "invalid",
            "frame_index": "invalid",
            "description": "Visible content.",
            "objects": [
                "person",
                123,
                "",
                " car "
            ],
            "entities": "not-a-list",
            "actions": [
                " walking ",
                null
            ],
            "confidence": 5
        }
        """
    )

    result = parser.parse(response)

    observation = result.observations[0]

    assert observation.timestamp_seconds == 0.0
    assert observation.frame_index == 0

    assert observation.objects == [
        "person",
        "car",
    ]

    assert observation.entities == []

    assert observation.actions == [
        "walking",
    ]

    assert observation.confidence is None


# ===========================================================================
# Negative numeric values
# ===========================================================================


def test_parser_normalizes_negative_timestamp(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        {
            "timestamp_seconds": -10,
            "frame_index": -5,
            "description": "A frame."
        }
        """
    )

    result = parser.parse(response)

    observation = result.observations[0]

    assert observation.timestamp_seconds == 0.0
    assert observation.frame_index == 0


# ===========================================================================
# Confidence
# ===========================================================================


@pytest.mark.parametrize(
    "confidence",
    [
        -1,
        1.1,
        2,
        "0.5",
        None,
    ],
)
def test_parser_rejects_invalid_confidence_values(
    parser: VisionResponseParser,
    confidence: object,
) -> None:
    response = build_response(
        """
        {
            "description": "A visible object.",
            "confidence": CONFIDENCE
        }
        """.replace(
            "CONFIDENCE",
            (
                "null"
                if confidence is None
                else repr(confidence)
            ),
        )
    )

    result = parser.parse(response)

    observation = result.observations[0]

    assert observation.confidence is None


def test_parser_accepts_valid_confidence(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        {
            "description": "A visible object.",
            "confidence": 0.75
        }
        """
    )

    result = parser.parse(response)

    assert result.observations[0].confidence == 0.75


# ===========================================================================
# Metadata
# ===========================================================================


def test_parser_adds_parser_metadata(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        {
            "description": "A scene."
        }
        """
    )

    result = parser.parse(response)

    observation = result.observations[0]

    assert observation.metadata["parser"] == "default"
    assert observation.metadata["source_format"] == "json"


# ===========================================================================
# Nested JSON
# ===========================================================================


def test_parser_handles_nested_json(
    parser: VisionResponseParser,
) -> None:
    response = build_response(
        """
        {
            "observations": [
                {
                    "description": "A desk.",
                    "objects": [
                        "desk",
                        "laptop"
                    ],
                    "metadata": {
                        "additional": {
                            "source": "vision"
                        }
                    }
                }
            ]
        }
        """
    )

    result = parser.parse(response)

    assert result.observation_count == 1
    assert result.observations[0].objects == [
        "desk",
        "laptop",
    ]