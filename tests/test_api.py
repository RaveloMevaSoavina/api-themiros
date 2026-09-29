from uuid import UUID

import httpx
from fastapi.testclient import TestClient

from app.api.dependencies import CurrentUser, get_current_user
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.main import app
from app.modules.frameworks.router import get_generation_service

TEST_USER_ID = UUID("11111111-1111-4111-8111-111111111111")


async def authenticated_user() -> CurrentUser:
    return CurrentUser(id=TEST_USER_ID, email="test@example.com", token="test")


class UnavailableGenerationService:
    async def generate(self, payload, user):
        raise ApiError(
            status_code=503,
            code="AI_PROVIDER_UNAVAILABLE",
            detail="Provider unavailable in test",
        )


def unavailable_generation_service() -> UnavailableGenerationService:
    return UnavailableGenerationService()


class ValidSupabaseAuthClient:
    async def get(self, url, *, headers, **options):
        assert url == "https://example.supabase.co/auth/v1/user"
        assert headers["Authorization"] == "Bearer valid-user-token"
        assert options["timeout"] == 5
        return httpx.Response(
            200,
            json={"id": str(TEST_USER_ID), "email": "test@example.com"},
        )


def configured_settings() -> Settings:
    return Settings(
        _env_file=None,
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="server-key",
    )


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Request-ID"]


def test_framework_endpoint_requires_authentication() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/frameworks/generate-pillars",
            json={
                "workspace_id": "11111111-1111-4111-8111-111111111111",
                "approach_id": "22222222-2222-4222-8222-222222222222",
            },
        )

    assert response.status_code == 401
    assert response.json()["code"] == "AUTHENTICATION_REQUIRED"


def test_framework_endpoint_accepts_supabase_user_token() -> None:
    app.dependency_overrides[get_settings] = configured_settings
    app.dependency_overrides[get_generation_service] = unavailable_generation_service
    try:
        with TestClient(app) as client:
            app.state.http_client = ValidSupabaseAuthClient()
            response = client.post(
                "/api/v1/frameworks/generate-pillars",
                headers={"Authorization": "Bearer valid-user-token"},
                json={
                    "workspace_id": "11111111-1111-4111-8111-111111111111",
                    "approach_id": "22222222-2222-4222-8222-222222222222",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["code"] == "AI_PROVIDER_UNAVAILABLE"


def test_complexity_endpoint_returns_deterministic_calculation() -> None:
    app.dependency_overrides[get_current_user] = authenticated_user
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/approach/complexity",
                json={
                    "scale": "national",
                    "actors": "2-3",
                    "themes": ["agriculture", "eau"],
                    "object_type": "program",
                    "budget": "inconnu",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["score"] == 5
    assert response.json()["value"] == "complique"


def test_unconfigured_generator_has_explicit_error() -> None:
    app.dependency_overrides[get_current_user] = authenticated_user
    app.dependency_overrides[get_generation_service] = unavailable_generation_service
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/frameworks/generate-pillars",
                json={
                    "workspace_id": "11111111-1111-4111-8111-111111111111",
                    "approach_id": "22222222-2222-4222-8222-222222222222",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["code"] == "AI_PROVIDER_UNAVAILABLE"
