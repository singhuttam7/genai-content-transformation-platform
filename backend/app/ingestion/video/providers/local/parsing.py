from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from app.ingestion.video.schemas import VisionObservation
from app.ingestion.video.providers.local.runtime import VisionRuntimeResponse


@dataclass(frozen=True, slots=True)
class ParsedVisionResponse:
    """
    Provider-independent representation of a parsed vision response.

    The parser converts model-generated text into structured
    VisionObservation objects without making any assumptions
    about the underlying model or runtime.
    """

    observations: tuple[VisionObservation, ...]
    parser_name: str
    source_format: str
    warnings: tuple[str, ...] = ()

    @property
    def observation_count(self) -> int:
        return len(self.observations)


class VisionResponseParser:
    """
    Parses model-generated vision responses into structured
    VisionObservation objects.

    Supported response formats:

    1. Structured JSON object
    2. Structured JSON array
    3. Markdown fenced JSON
    4. Plain-text model response

    The parser never performs model inference and never communicates
    with Ollama. Its responsibility is limited to interpreting and
    validating the textual runtime response.
    """

    name = "default"

    def parse(
        self,
        response: VisionRuntimeResponse,
    ) -> ParsedVisionResponse:
        """
        Parse a VisionRuntimeResponse.

        Raises:
            ValueError:
                If the response is missing or contains no usable text.
            TypeError:
                If the supplied response is not a VisionRuntimeResponse.
        """

        if response is None:
            raise ValueError(
                "Vision runtime response must be provided."
            )

        if not isinstance(response, VisionRuntimeResponse):
            raise TypeError(
                "Vision response parser requires a "
                "VisionRuntimeResponse."
            )

        text = response.text.strip()

        if not text:
            raise ValueError(
                "Vision runtime response text must not be empty."
            )

        # ---------------------------------------------------------------
        # First attempt: structured JSON
        # ---------------------------------------------------------------

        json_payload = self._extract_json_payload(text)

        if json_payload is not None:
            observations, warnings = self._parse_json_payload(
                json_payload
            )

            # IMPORTANT:
            #
            # Once valid JSON has been detected, preserve the JSON
            # source format even when zero observations were produced.
            #
            # This distinction is important for the provider layer:
            #
            #   valid JSON + empty observations
            #       -> "empty_observations"
            #
            # rather than:
            #
            #   valid JSON + empty observations
            #       -> "unstructured_response"
            #
            # Plain-text fallback must only happen when the response
            # cannot be interpreted as JSON at all.
            return ParsedVisionResponse(
                observations=tuple(observations),
                parser_name=self.name,
                source_format="json",
                warnings=tuple(warnings),
            )

        # ---------------------------------------------------------------
        # Fallback: plain text
        # ---------------------------------------------------------------

        observation = self._build_text_observation(text)

        return ParsedVisionResponse(
            observations=(observation,),
            parser_name=self.name,
            source_format="text",
            warnings=(
                "Vision model response was not structured JSON; "
                "plain-text fallback was used.",
            ),
        )

    # ===================================================================
    # JSON extraction
    # ===================================================================

    def _extract_json_payload(
        self,
        text: str,
    ) -> Any | None:
        """
        Attempt to extract JSON from a model response.

        Handles:

            {...}

        and:

            [...]

        as well as Markdown fenced JSON:

            ```json
            {...}
            ```
        """

        cleaned = text.strip()

        # ---------------------------------------------------------------
        # Markdown fenced JSON
        # ---------------------------------------------------------------

        if cleaned.startswith("```") and cleaned.endswith("```"):
            lines = cleaned.splitlines()

            if len(lines) >= 3:
                first_line = lines[0].strip().lower()

                if first_line in {
                    "```",
                    "```json",
                    "```jsonc",
                }:
                    cleaned = "\n".join(
                        lines[1:-1]
                    ).strip()

        # ---------------------------------------------------------------
        # Direct JSON
        # ---------------------------------------------------------------

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # ---------------------------------------------------------------
        # JSON embedded inside surrounding text
        # ---------------------------------------------------------------

        extracted = self._extract_embedded_json(cleaned)

        if extracted is None:
            return None

        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_embedded_json(
        text: str,
    ) -> str | None:
        """
        Find a JSON object or array embedded inside surrounding text.

        This intentionally uses bracket matching instead of a greedy
        regular expression so nested JSON structures are handled safely.
        """

        starts = []

        for index, character in enumerate(text):
            if character in "{[":
                starts.append(index)

        for start in starts:
            opening = text[start]

            if opening == "{":
                closing = "}"
            else:
                closing = "]"

            depth = 0
            in_string = False
            escaped = False

            for index in range(start, len(text)):
                character = text[index]

                if in_string:
                    if escaped:
                        escaped = False
                    elif character == "\\":
                        escaped = True
                    elif character == '"':
                        in_string = False

                    continue

                if character == '"':
                    in_string = True
                    continue

                if character == opening:
                    depth += 1

                elif character == closing:
                    depth -= 1

                    if depth == 0:
                        return text[start:index + 1]

        return None

    # ===================================================================
    # JSON interpretation
    # ===================================================================

    def _parse_json_payload(
        self,
        payload: Any,
    ) -> tuple[list[VisionObservation], list[str]]:
        """
        Convert supported JSON structures into observations.
        """

        warnings: list[str] = []

        # ---------------------------------------------------------------
        # JSON array
        # ---------------------------------------------------------------

        if isinstance(payload, list):
            observations: list[VisionObservation] = []

            for index, item in enumerate(payload):
                if not isinstance(item, dict):
                    warnings.append(
                        f"JSON observation at index {index} "
                        "was ignored because it was not an object."
                    )
                    continue

                observation = self._parse_observation(
                    item,
                    fallback_index=index,
                )

                if observation is not None:
                    observations.append(observation)

            return observations, warnings

        # ---------------------------------------------------------------
        # JSON object
        # ---------------------------------------------------------------

        if isinstance(payload, dict):
            # Common structured format:
            #
            # {
            #     "observations": [...]
            #     }
            observations_payload = payload.get(
                "observations"
            )

            if isinstance(observations_payload, list):
                observations: list[VisionObservation] = []

                for index, item in enumerate(
                    observations_payload
                ):
                    if not isinstance(item, dict):
                        warnings.append(
                            f"JSON observation at index {index} "
                            "was ignored because it was not an object."
                        )
                        continue

                    observation = self._parse_observation(
                        item,
                        fallback_index=index,
                    )

                    if observation is not None:
                        observations.append(observation)

                return observations, warnings

            # Single observation object.
            observation = self._parse_observation(
                payload,
                fallback_index=0,
            )

            if observation is not None:
                return [observation], warnings

            warnings.append(
                "JSON object did not contain a valid vision observation."
            )

        return [], warnings

    def _parse_observation(
        self,
        data: dict[str, Any],
        *,
        fallback_index: int,
    ) -> VisionObservation | None:
        """
        Normalize one model-generated observation.
        """

        description = self._string_or_none(
            data.get("description")
        )

        objects = self._string_list(
            data.get("objects")
        )

        entities = self._string_list(
            data.get("entities")
        )

        actions = self._string_list(
            data.get("actions")
        )

        scene = self._string_or_none(
            data.get("scene")
        )

        visible_text = self._string_or_none(
            data.get("visible_text")
        )

        timestamp_seconds = self._non_negative_float(
            data.get("timestamp_seconds"),
            default=0.0,
        )

        frame_index = self._non_negative_int(
            data.get("frame_index"),
            default=fallback_index,
        )

        confidence = self._confidence_or_none(
            data.get("confidence")
        )

        # ---------------------------------------------------------------
        # A completely empty structured object is not useful.
        # ---------------------------------------------------------------

        if not any(
            (
                description,
                objects,
                entities,
                actions,
                scene,
                visible_text,
            )
        ):
            return None

        metadata: dict[str, object] = {
            "parser": self.name,
            "source_format": "json",
        }

        return VisionObservation(
            timestamp_seconds=timestamp_seconds,
            frame_index=frame_index,
            description=description,
            objects=objects,
            entities=entities,
            actions=actions,
            scene=scene,
            visible_text=visible_text,
            confidence=confidence,
            metadata=metadata,
        )

    # ===================================================================
    # Plain-text fallback
    # ===================================================================

    def _build_text_observation(
        self,
        text: str,
    ) -> VisionObservation:
        """
        Convert an unstructured model response into a safe
        VisionObservation.

        We deliberately do not attempt unreliable NLP extraction here.

        The complete model response becomes the description and the
        semantic fields remain empty. More sophisticated structured
        generation belongs to A4.7.4.8.
        """

        return VisionObservation(
            timestamp_seconds=0.0,
            frame_index=0,
            description=text,
            objects=[],
            entities=[],
            actions=[],
            scene=None,
            visible_text=None,
            confidence=None,
            metadata={
                "parser": self.name,
                "source_format": "text",
                "fallback": True,
            },
        )

    # ===================================================================
    # Value normalization helpers
    # ===================================================================

    @staticmethod
    def _string_or_none(
        value: Any,
    ) -> str | None:
        if not isinstance(value, str):
            return None

        value = value.strip()

        return value or None

    @staticmethod
    def _string_list(
        value: Any,
    ) -> list[str]:
        if not isinstance(value, list):
            return []

        result: list[str] = []

        for item in value:
            if not isinstance(item, str):
                continue

            normalized = item.strip()

            if normalized:
                result.append(normalized)

        return result

    @staticmethod
    def _non_negative_float(
        value: Any,
        *,
        default: float,
    ) -> float:
        if isinstance(value, bool):
            return default

        if isinstance(value, (int, float)):
            normalized = float(value)

            if normalized >= 0:
                return normalized

        return default

    @staticmethod
    def _non_negative_int(
        value: Any,
        *,
        default: int,
    ) -> int:
        if isinstance(value, bool):
            return default

        if isinstance(value, int):
            if value >= 0:
                return value

        if isinstance(value, float):
            if value.is_integer() and value >= 0:
                return int(value)

        return default

    @staticmethod
    def _confidence_or_none(
        value: Any,
    ) -> float | None:
        if isinstance(value, bool):
            return None

        if not isinstance(value, (int, float)):
            return None

        normalized = float(value)

        if 0.0 <= normalized <= 1.0:
            return normalized

        return None