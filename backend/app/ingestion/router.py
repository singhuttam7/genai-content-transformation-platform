from __future__ import annotations

from collections.abc import Iterable

from app.ingestion.processor import ContentProcessor
from app.ingestion.schemas import InputType


class ProcessorRouter:
    """Routes input types to their registered content processors."""

    def __init__(
        self,
        processors: Iterable[ContentProcessor] | None = None,
    ) -> None:
        self._processors: dict[
            InputType,
            ContentProcessor,
        ] = {}

        if processors is not None:
            for processor in processors:
                self.register(processor)

    def register(
        self,
        processor: ContentProcessor,
    ) -> None:
        """Register a processor for its supported input types."""

        for input_type in processor.supported_types:
            if input_type in self._processors:
                existing = self._processors[input_type]

                raise ValueError(
                    f"Processor already registered for "
                    f"{input_type.value}: "
                    f"{type(existing).__name__}"
                )

            self._processors[input_type] = processor

    def get_processor(
        self,
        input_type: InputType,
    ) -> ContentProcessor:
        """Return the processor registered for an input type."""

        processor = self._processors.get(input_type)

        if processor is None:
            raise ValueError(
                f"No processor registered for "
                f"input type: {input_type.value}"
            )

        return processor

    def supports(
        self,
        input_type: InputType,
    ) -> bool:
        """Return whether a processor exists for an input type."""

        return input_type in self._processors

    @property
    def supported_types(self) -> tuple[InputType, ...]:
        """Return all currently supported input types."""

        return tuple(self._processors.keys())