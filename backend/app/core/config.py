from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide configuration.

    Values are loaded from environment variables and the
    backend .env file.
    """

    # ---------------------------------------------------------
    # Application
    # ---------------------------------------------------------

    app_name: str = "GenAI Content Transformation Platform"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    # ---------------------------------------------------------
    # API
    # ---------------------------------------------------------

    api_v1_prefix: str = "/api/v1"

    # ---------------------------------------------------------
    # CORS
    # ---------------------------------------------------------

    frontend_url: str = "http://localhost:5173"

    # ---------------------------------------------------------
    # Database
    # ---------------------------------------------------------

    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/"
        "genai_content_platform"
    )

    # ---------------------------------------------------------
    # Redis
    # ---------------------------------------------------------

    redis_url: str = "redis://localhost:6379/0"

    # ---------------------------------------------------------
    # Storage
    # ---------------------------------------------------------

    storage_backend: str = "local"
    storage_root: str = "./data/storage"
    max_upload_size_mb: int = 100

    # ---------------------------------------------------------
    # AI Providers
    # ---------------------------------------------------------

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    # ---------------------------------------------------------
# OCR
# ---------------------------------------------------------

    ocr_enabled: bool = True
    ocr_provider: str = "tesseract"
    ocr_default_language: str = "eng"
    ocr_timeout_seconds: float = 30.0
    tesseract_executable_path: str | None = None


        # ---------------------------------------------------------
    # ASR / Speech Recognition
    # ---------------------------------------------------------

    asr_enabled: bool = True
    asr_provider: str = "faster_whisper"
    asr_default_language: str | None = None
    asr_timeout_seconds: float = 300.0

    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # ---------------------------------------------------------
    # RAG
    # ---------------------------------------------------------

    embedding_model: str = "text-embedding-3-small"
    vector_database_url: str | None = None

    # ---------------------------------------------------------
    # Security
    # ---------------------------------------------------------

    secret_key: str = "change-this-development-secret"
    access_token_expire_minutes: int = 30

    # ---------------------------------------------------------
    # URL Ingestion / Outbound HTTP
    # ---------------------------------------------------------

    url_fetch_timeout_seconds: float = 15.0
    url_max_response_size_mb: int = 10
    url_max_redirects: int = 5

    url_allowed_content_types: tuple[str, ...] = (
        "text/html",
        "application/xhtml+xml",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached application settings instance.
    """

    return Settings()


settings = get_settings()