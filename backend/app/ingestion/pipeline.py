from __future__ import annotations

from app.ingestion.detector import InputDetector
from app.ingestion.enrichment import ContentEnrichmentService
from app.ingestion.normalizer import ContentNormalizer
from app.ingestion.router import ProcessorRouter
from app.ingestion.schemas import (
    CanonicalContent,
    IngestionRequest,
)


class IngestionPipeline:
    """
    Orchestrates the content understanding pipeline.

    Pipeline:

        IngestionRequest
            ↓
        Input Detection
            ↓
        Processor Routing
            ↓
        Content Processing
            ↓
        Content Enrichment
            ↓
        Content Normalization
            ↓
        CanonicalContent
    """

    def __init__(
        self,
        *,
        detector: InputDetector,
        router: ProcessorRouter,
        normalizer: ContentNormalizer,
        enrichment: ContentEnrichmentService | None = None,
    ) -> None:
        self.detector = detector
        self.router = router
        self.normalizer = normalizer
        self.enrichment = enrichment

    async def run(
        self,
        request: IngestionRequest,
    ) -> CanonicalContent:
        """
        Execute the complete ingestion pipeline.
        """

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
        # 5. Enrich extracted content
        # ---------------------------------------------------------

        if self.enrichment is not None:
            extracted_content = (
                await self.enrichment.enrich(
                    request=request,
                    content=extracted_content,
                )
            )

        # ---------------------------------------------------------
        # 6. Normalize into canonical representation
        # ---------------------------------------------------------

        canonical_content = (
            await self.normalizer.normalize(
                extracted_content
            )
        )

        return canonical_content