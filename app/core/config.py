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
