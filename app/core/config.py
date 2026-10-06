from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Themiros API"
    app_version: str = "0.1.0"
    app_env: Literal["development", "test", "staging", "production"] = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_allowed_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    supabase_url: str | None = None
    supabase_service_role_key: str | None = Field(default=None, repr=False)
    supabase_jwt_audience: str = "authenticated"
    ai_provider: Literal["disabled", "openai"] = "disabled"
    openai_api_key: str | None = Field(default=None, repr=False)
    openai_model: str = "gpt-6-astra"
    openai_timeout_seconds: float = Field(default=30, gt=0, le=300)
    openai_input_cost_eur_per_million: Decimal = Field(default=Decimal("0"), ge=0)
    openai_output_cost_eur_per_million: Decimal = Field(default=Decimal("0"), ge=0)
    ai_max_cost_eur: Decimal = Field(default=Decimal("0"), ge=0)

    # E4 — ingestion documentaire. Les seuils métier (70/40, bonus, malus)
    # vivent dans la table `ingestion_settings` ; ici, l'infrastructure.
    ingestion_storage_bucket: str = "documents"
    ingestion_worker_poll_seconds: float = Field(default=2, gt=0, le=60)
    ingestion_worker_lease_seconds: int = Field(default=300, ge=60, le=3600)
    ingestion_worker_concurrency: int = Field(default=2, ge=1, le=16)
    # Embeddings calculés localement (fastembed, ONNX) par défaut ; `openai`
    # reste possible avec un modèle `text-embedding-3-*` réduit à 768.
    # Changer de modèle impose de réindexer tout le corpus.
    ingestion_embedding_provider: Literal["local", "openai"] = "local"
    # Version int8 du modèle : ~3 fois plus rapide sur CPU, vecteurs quasi
    # identiques (similarité ≥ 0,99 avec la version complète).
    ingestion_embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2-int8"
    )
    ingestion_embedding_dimensions: int = Field(default=768, gt=0)
    ingestion_embedding_batch_size: int = Field(default=32, ge=1, le=2048)
    ingestion_embedding_cache_dir: str | None = None
    ingestion_embedding_threads: int | None = Field(default=None, ge=1, le=64)
    ingestion_embedding_window_words: int = Field(default=80, ge=16, le=512)
    # Segments vectorisés pour le score de pertinence (§7.3), répartis dans
    # tout le document ; les autres ne le sont qu'à l'indexation.
    ingestion_relevance_sample_segments: int = Field(default=32, ge=1, le=1000)
    ingestion_ocr_base_languages: str = "fra+eng"
    ingestion_ocr_dpi: int = Field(default=300, ge=100, le=600)
    ingestion_ocr_page_timeout_seconds: int = Field(default=120, gt=0, le=600)
    ingestion_ocr_parallelism: int = Field(default=4, ge=1, le=16)
    ingestion_ner_model: str = "xx_ent_wiki_sm"

    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_allowed_origins.split(",")
            if origin.strip()
        ]

    @computed_field
    @property
    def supabase_jwks_url(self) -> str | None:
        if not self.supabase_url:
            return None
        return f"{self.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"

    @computed_field
    @property
    def supabase_jwt_issuer(self) -> str | None:
        if not self.supabase_url:
            return None
        return f"{self.supabase_url.rstrip('/')}/auth/v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
