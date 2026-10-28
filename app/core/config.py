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
    # Embeddings (`ingestion_embedding_model`) : un seul tarif, en entrée.
    openai_embedding_cost_eur_per_million: Decimal = Field(default=Decimal("0"), ge=0)
    ai_max_cost_eur: Decimal = Field(default=Decimal("0"), ge=0)

    # E4 — ingestion documentaire. Les seuils métier (70/40, bonus, malus)
    # vivent dans la table `ingestion_settings` ; ici, l'infrastructure.
    ingestion_storage_bucket: str = "documents"
    ingestion_worker_poll_seconds: float = Field(default=2, gt=0, le=60)
    ingestion_worker_lease_seconds: int = Field(default=300, ge=60, le=3600)
    ingestion_worker_concurrency: int = Field(default=2, ge=1, le=16)
    # OpenAI `text-embedding-3-large` réduit à 768 dimensions : entraîné pour
    # retrouver le passage qui répond à une question, y compris d'une langue à
    # l'autre (questions en français, corpus en anglais), ce que le modèle
    # local de paraphrases (`local`, fastembed) fait mal. Changer de modèle
    # impose de réindexer le corpus (`app.modules.ingestion.reindex`) et de
    # recalibrer les seuils de similarité.
    ingestion_embedding_provider: Literal["local", "openai"] = "openai"
    ingestion_embedding_model: str = "text-embedding-3-large"
    ingestion_embedding_dimensions: int = Field(default=768, gt=0)
    # Modèle local uniquement : OpenAI regroupe selon les limites de son API.
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
    # E5 — vectorisation des cadres validés pour la classification du corpus.
    classification_poll_seconds: float = Field(default=10, gt=0, le=300)
    # E6 — runs d'analyse. Similarité « forte » d'une détection : seuil de la
    # valeur 1 d'une variable, propre au modèle d'embeddings. 0,43 avec
    # text-embedding-3-large garde la même part de détections fortes (14 %)
    # que 0,55 avec l'ancien modèle local, que le nouveau n'atteint presque
    # jamais (toutes les variables restaient à 0,5).
    run_poll_seconds: float = Field(default=5, gt=0, le=300)
    run_lease_seconds: int = Field(default=300, ge=60, le=3600)
    run_strong_similarity: float = Field(default=0.43, gt=0, le=1)
    run_max_evidences_per_variable: int = Field(default=10, ge=1, le=100)
    # Un run attend la classification du corpus (cadre tout juste validé)
    # au plus ce délai, en se remettant en file toutes les run_defer_seconds.
    run_classification_wait_seconds: int = Field(default=3600, ge=60, le=86400)
    run_defer_seconds: int = Field(default=30, ge=5, le=600)
    # E9 — couche B. Similarité minimale d'un extrait avec une sous-question :
    # la spec dit 0,7, qu'aucun modèle n'atteint ; 0,32 avec
    # text-embedding-3-large laisse passer la même part du corpus que 0,40
    # avec l'ancien modèle local. L'IA écarte ensuite les extraits hors sujet.
    # Les extraits soumis à l'IA sont limités et tronqués pour contenir le
    # nombre de tokens ; au-delà du plafond d'entrée, les critères sont
    # répartis sur plusieurs appels. Avec 3 extraits de 600 caractères, l'IA
    # laissait un tiers des sous-questions sans preuve et notait surtout 1-2 ;
    # 6 extraits de 1 200 caractères (~32 000 tokens pour 51 sous-questions)
    # tiennent toujours en un seul appel.
    run_layer_b_min_similarity: float = Field(default=0.32, gt=0, le=1)
    run_layer_b_top_k: int = Field(default=8, ge=1, le=20)
    # Sous-questions par appel de recherche : la base coupe une requête
    # au-delà de 8 s.
    run_layer_b_match_batch_size: int = Field(default=8, ge=1, le=100)
    run_layer_b_excerpts_per_question: int = Field(default=6, ge=1, le=10)
    run_layer_b_excerpt_chars: int = Field(default=1200, ge=100, le=4000)
    run_layer_b_max_input_tokens: int = Field(default=80000, ge=1000, le=400000)
    run_layer_b_timeout_seconds: float = Field(default=120, gt=0, le=600)

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
