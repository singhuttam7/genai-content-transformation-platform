from __future__ import annotations

from app.ingestion.detector import InputDetector
from app.ingestion.normalizer import ContentNormalizer
from app.ingestion.router import ProcessorRouter
from app.ingestion.schemas import (
    CanonicalContent,
    IngestionRequest,
)


class IngestionPipeline:
    """Orchestrates the content understanding pipeline.

    Pipeline:

        IngestionRequest
            ↓
        Input Detection
            ↓
        Processor Routing
            ↓
        Content Processing
            ↓
        Content Normalization
            ↓
        CanonicalContent
    """

    def __init__(
        self,
        detector: InputDetector,
        router: ProcessorRouter,
        normalizer: ContentNormalizer,
    ) -> None:
        self.detector = detector
        self.router = router
        self.normalizer = normalizer

    async def run(
        self,
        request: IngestionRequest,
    ) -> CanonicalContent:
        """Execute the complete ingestion pipeline."""

        # ---------------------------------------------------------
        # 1. Detect input type
        # ---------------------------------------------------------

        detected_type = self.detector.detect(
            request
        )

        # ---------------------------------------------------------
        # 2. Ensure request type matches detection
        # ---------------------------------------------------------

        if detected_type != request.input_type:
            request = request.model_copy(
                update={
                    "input_type": detected_type,
                }
            )

        # ---------------------------------------------------------
        # 3. Select processor
        # ---------------------------------------------------------

        processor = self.router.get_processor(
            detected_type
        )

        # ---------------------------------------------------------
        # 4. Extract content
        # ---------------------------------------------------------

        extracted_content = await processor.process(
            request
        )

        # ---------------------------------------------------------
        # 5. Normalize into canonical representation
        # ---------------------------------------------------------

        canonical_content = await self.normalizer.normalize(
            extracted_content
        )

        return canonical_content