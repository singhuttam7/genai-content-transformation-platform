from __future__ import annotations

from app.core.config import settings
from app.models_ai.embeddings.adapters.sentence_transformer import (
    SentenceTransformerAdapter,
)
from app.models_ai.embeddings.schemas import EmbeddingModelConfig
from app.models_ai.embeddings.service import EmbeddingService


EMBEDDING_PROVIDER = "sentence-transformers"
EMBEDDING_DIMENSION = 384


def create_embedding_service() -> EmbeddingService:
    """
    Construct the application embedding service using the
    project's local Sentence-Transformer adapter.
    """

    model_config = EmbeddingModelConfig(
        provider=EMBEDDING_PROVIDER,
        model_name=settings.embedding_model,
        expected_dimension=EMBEDDING_DIMENSION,
        normalized=False,
        batch_size=32,
        options={
            "device": "cpu",
            "encode": {
                "show_progress_bar": False,
            },
        },
    )

    adapter = SentenceTransformerAdapter(
        model_config=model_config,
    )

    return EmbeddingService(
        port=adapter,
        model_config=model_config,
    )