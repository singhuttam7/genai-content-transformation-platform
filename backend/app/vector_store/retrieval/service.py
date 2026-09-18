from __future__ import annotations

from math import isfinite
from uuid import UUID

from app.vector_store.retrieval.port import VectorRetrievalPort
from app.vector_store.retrieval.schemas import (
    VectorRetrievalRequest,
    VectorRetrievalResult,
)


class VectorRetrievalService:
    """Application-level service for vector similarity retrieval."""

    def __init__(
        self,
        port: VectorRetrievalPort,
    ) -> None:
        if not isinstance(port, VectorRetrievalPort):
            raise TypeError(
                "port must be a VectorRetrievalPort."
            )

        self._port = port

    @property
    def port(self) -> VectorRetrievalPort:
        """Return the configured retrieval port."""
        return self._port

    async def search(
        self,
        request: VectorRetrievalRequest,
    ) -> VectorRetrievalResult:
        """
        Execute vector retrieval through the configured port.

        The service validates application-level retrieval invariants
        before delegating provider-specific behavior to the port.
        """
        self._validate_request(request)

        return await self._port.search(request)

    def _validate_request(
        self,
        request: VectorRetrievalRequest,
    ) -> None:
        """Validate the retrieval request before delegation."""
        if not isinstance(request, VectorRetrievalRequest):
            raise TypeError(
                "request must be a VectorRetrievalRequest."
            )

        if not request.query_vector:
            raise ValueError(
                "query_vector must not be empty."
            )

        if request.top_k < 1:
            raise ValueError(
                "top_k must be greater than or equal to 1."
            )

        if request.similarity_threshold is not None:
            if not -1.0 <= request.similarity_threshold <= 1.0:
                raise ValueError(
                    "similarity_threshold must be between -1.0 and 1.0."
                )

        if not isinstance(request.metadata_filter, dict):
            raise TypeError(
                "metadata_filter must be a dictionary."
            )

        if request.project_id is not None and not isinstance(
            request.project_id,
            UUID,
        ):
            raise TypeError(
                "project_id must be a UUID or None."
            )

        self._validate_query_vector(
            request.query_vector,
        )

        self._validate_query_dimension(
            request,
        )

    def _validate_query_vector(
        self,
        query_vector: list[float],
    ) -> None:
        """Validate query vector values."""
        for value in query_vector:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    "query_vector must contain only numeric values."
                )

            if not isfinite(float(value)):
                raise ValueError(
                    "query_vector must contain only finite numeric values."
                )

    def _validate_query_dimension(
        self,
        request: VectorRetrievalRequest,
    ) -> None:
        """Validate query vector dimension against model dimension."""
        expected_dimension = request.model.dimension
        actual_dimension = len(request.query_vector)

        if actual_dimension != expected_dimension:
            raise ValueError(
                "query_vector dimension must match model dimension "
                f"({expected_dimension}); received {actual_dimension}."
            )