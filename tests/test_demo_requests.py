from datetime import UTC, datetime
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.errors import ApiError
from app.infrastructure.supabase import SupabaseGateway
from app.main import app
from app.modules.demo_requests.guard import AbuseGuard, client_key
from app.modules.demo_requests.notifier import Mail, NotificationFailed
from app.modules.demo_requests.repository import DemoRequestRepository
from app.modules.demo_requests.router import get_demo_request_service
from app.modules.demo_requests.schemas import DemoRequestPayload
from app.modules.demo_requests.service import DemoRequestService

NOW = datetime(2026, 11, 4, 9, 30, tzinfo=UTC)

FORM = {
    "source": "landing/request-demo.html",
    "name": "  Jane Doe ",
    "organisation": "Ministère du Plan",
    "email": "Jane@Plan.gouv.fr",
    "context": "Évaluation à mi-parcours\nd'un programme",
    "website": "",
}


class FakeRepository:
    def __init__(self, *, recent: int = 0, rate_fails: bool = False) -> None:
        self.recent = recent
        self.rate_fails = rate_fails
        self.rows: list[dict] = []
        self.notified: list[UUID] = []
        self.rate_queries: list[tuple[str, datetime, int]] = []

    async def count_recent(self, ip, *, since, limit):
        self.rate_queries.append((ip, since, limit))
        if self.rate_fails:
            raise ApiError(status_code=503, code="DATABASE_UNAVAILABLE", detail="x")
        return self.recent

    async def insert(self, row):
        self.rows.append(row)

    async def mark_notified(self, request_id):
        self.notified.append(request_id)


class FakeNotifier:
    def __init__(self, *, fails: bool = False) -> None:
        self.fails = fails
        self.mails: list[Mail] = []

    async def send(self, mail):
        if self.fails:
            raise NotificationFailed("resend down")
        self.mails.append(mail)


def service(repository, notifier=None, **overrides) -> DemoRequestService:
    options = {
        "sender": "Themiros <demo@themiros.com>",
        "recipients": ["team@themiros.com"],
        "backoffice_url": "https://admin.themiros.com/",
        "clock": lambda: NOW,
        **overrides,
    }
    return DemoRequestService(repository, notifier, **options)


def payload(**overrides) -> DemoRequestPayload:
    return DemoRequestPayload.model_validate({**FORM, **overrides})


async def test_valid_request_is_stored_then_notified():
    repository, notifier = FakeRepository(), FakeNotifier()
    await service(repository, notifier).submit(
        payload(), ip="203.0.113.7", user_agent="Test/1.0"
    )

    [row] = repository.rows
    assert {key: value for key, value in row.items() if key != "id"} == {
        "name": "Jane Doe",
        "organisation": "Ministère du Plan",
        "email": "jane@plan.gouv.fr",
        "context": "Évaluation à mi-parcours\nd'un programme",
        "source": "landing/request-demo.html",
        "user_agent": "Test/1.0",
        "ip": "203.0.113.7",
    }
    assert repository.notified == [UUID(row["id"])]
    [mail] = notifier.mails
    assert mail.reply_to == "jane@plan.gouv.fr"
    assert mail.recipients == ["team@themiros.com"]
    assert mail.subject == "Demande de démo — Ministère du Plan"
    assert "Suivi : https://admin.themiros.com/demo-requests" in mail.text
    assert "04/11/2026 à 10:30" in mail.text
    assert repository.rate_queries == [
        ("203.0.113.7", datetime(2026, 11, 4, 9, 20, tzinfo=UTC), 3)
    ]


async def test_empty_context_is_stored_as_null():
    repository, notifier = FakeRepository(), FakeNotifier()
    await service(repository, notifier).submit(
        payload(context="   "), ip=None, user_agent=None
    )
    assert repository.rows[0]["context"] is None
    assert repository.rows[0]["user_agent"] is None
    assert "(non renseigné)" in notifier.mails[0].text
    assert repository.rate_queries == []


async def test_honeypot_drops_the_request_silently():
    repository, notifier = FakeRepository(), FakeNotifier()
    await service(repository, notifier).submit(
        payload(website="http://spam"), ip="203.0.113.7", user_agent=None
    )
    assert repository.rows == []
    assert notifier.mails == []


async def test_rate_limit_rejects_without_storing():
    repository = FakeRepository(recent=3)
    with pytest.raises(ApiError) as error:
        await service(repository, FakeNotifier()).submit(
            payload(), ip="203.0.113.7", user_agent=None
        )
    assert error.value.status_code == 429
    assert error.value.code == "DEMO_REQUEST_RATE_LIMITED"
    assert repository.rows == []


async def test_rate_check_failure_does_not_lose_the_request():
    repository = FakeRepository(rate_fails=True)
    await service(repository, FakeNotifier()).submit(
        payload(), ip="203.0.113.7", user_agent=None
    )
    assert len(repository.rows) == 1


async def test_notification_failure_keeps_the_request_unnotified():
    repository = FakeRepository()
    await service(repository, FakeNotifier(fails=True)).submit(
        payload(), ip=None, user_agent=None
    )
    assert len(repository.rows) == 1
    assert repository.notified == []


@pytest.mark.parametrize(
    "overrides", [{"sender": None}, {"recipients": []}], ids=["sender", "to"]
)
async def test_missing_mail_settings_store_without_sending(overrides):
    repository, notifier = FakeRepository(), FakeNotifier()
    await service(repository, notifier, **overrides).submit(
        payload(), ip=None, user_agent=None
    )
    assert len(repository.rows) == 1
    assert notifier.mails == []
    assert repository.notified == []


async def test_repository_sends_postgrest_queries():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=[{"id": "a"}, {"id": "b"}])
        return httpx.Response(201 if request.method == "POST" else 204)

    settings = Settings(
        _env_file=None,
        supabase_url="https://supabase.test",
        supabase_service_role_key="service-key",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        repository = DemoRequestRepository(SupabaseGateway(settings, client))
        assert await repository.count_recent("203.0.113.7", since=NOW, limit=3) == 2
        await repository.insert({"id": "x", "name": "Jane"})
        await repository.mark_notified(UUID("11111111-1111-4111-8111-111111111111"))

    select, insert, update = requests
    assert select.url.path == "/rest/v1/demo_requests"
    assert select.url.params["ip"] == "eq.203.0.113.7"
    assert select.url.params["created_at"] == "gte.2026-11-04T09:30:00+00:00"
    assert select.url.params["limit"] == "3"
    assert select.headers["authorization"] == "Bearer service-key"
    assert insert.method == "POST"
    assert update.method == "PATCH"
    assert update.url.params["id"] == "eq.11111111-1111-4111-8111-111111111111"


def post(json: dict, headers: dict | None = None, repository=None):
    repository = repository or FakeRepository()
    app.state.demo_request_guard = AbuseGuard()
    app.dependency_overrides[get_demo_request_service] = lambda: service(
        repository, FakeNotifier()
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/demo-requests", json=json, headers=headers or {}
            )
    finally:
        app.dependency_overrides.clear()
    return response, repository


def test_endpoint_is_public_and_returns_201():
    response, repository = post(
        FORM, headers={"X-Forwarded-For": "10.0.0.9, 198.51.100.4"}
    )
    assert response.status_code == 201
    assert response.json() == {"status": "received"}
    # Dernière entrée : celle ajoutée par Caddy, pas celle du client.
    assert repository.rows[0]["ip"] == "198.51.100.4"


def test_endpoint_ignores_a_malformed_forwarded_address():
    _, repository = post(FORM, headers={"X-Forwarded-For": "unknown"})
    assert repository.rows[0]["ip"] is None


def test_endpoint_rejects_invalid_fields():
    response, repository = post({**FORM, "name": "", "email": "pas-un-mail"})
    assert response.status_code == 422
    fields = {error["loc"][-1] for error in response.json()["context"]["errors"]}
    assert fields == {"name", "email"}
    assert repository.rows == []


def test_endpoint_answers_429_when_rate_limited():
    response, _ = post(
        FORM,
        headers={"X-Forwarded-For": "198.51.100.4"},
        repository=FakeRepository(recent=3),
    )
    assert response.status_code == 429
    assert response.json()["code"] == "DEMO_REQUEST_RATE_LIMITED"


def test_cors_preflight_from_the_landing_page(monkeypatch):
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://themiros.com")
    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            response = client.options(
                "/api/v1/demo-requests",
                headers={
                    "Origin": "https://themiros.com",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
    finally:
        get_settings.cache_clear()
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://themiros.com"


# --- Garde anti-robots ------------------------------------------------------


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_guard_bans_an_address_beyond_the_attempt_limit():
    clock = FakeClock()
    guard = AbuseGuard(
        attempt_limit=3, attempt_window_seconds=60, ban_seconds=600, clock=clock
    )
    assert [guard.blocked_for("1.2.3.4") for _ in range(3)] == [None] * 3
    assert guard.blocked_for("1.2.3.4") == 600
    # Les autres adresses ne sont pas touchées.
    assert guard.blocked_for("5.6.7.8") is None
    clock.now += 599
    assert guard.blocked_for("1.2.3.4") == 1
    clock.now += 2
    assert guard.blocked_for("1.2.3.4") is None


def test_guard_window_slides():
    clock = FakeClock()
    guard = AbuseGuard(attempt_limit=2, attempt_window_seconds=60, clock=clock)
    assert guard.blocked_for("k") is None
    assert guard.blocked_for("k") is None
    clock.now += 61
    assert guard.blocked_for("k") is None


def test_guard_global_limit_counts_stored_requests_per_hour():
    clock = FakeClock()
    guard = AbuseGuard(global_limit=2, clock=clock)
    guard.record_stored()
    clock.now += 100
    guard.record_stored()
    assert guard.global_retry_after() == 3500
    clock.now += 3500
    assert guard.global_retry_after() is None


def test_guard_memory_stays_bounded():
    guard = AbuseGuard(max_tracked=100)
    for index in range(1000):
        guard.blocked_for(f"10.0.{index // 256}.{index % 256}")
    assert len(guard._attempts) <= 101


def test_client_key_groups_ipv6_by_64():
    assert client_key("2001:db8:1:2:aaaa::1") == "2001:db8:1:2::/64"
    assert client_key("2001:db8:1:2:bbbb::9") == "2001:db8:1:2::/64"
    assert client_key("198.51.100.4") == "198.51.100.4"
    assert client_key(None) == "unknown"


@pytest.fixture
def landing_client(monkeypatch):
    """Application complète, CORS de la landing compris."""
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://themiros.com")
    from app.core.config import get_settings
    from app.main import create_app

    get_settings.cache_clear()
    application = create_app()
    repository = FakeRepository()
    application.dependency_overrides[get_demo_request_service] = lambda: service(
        repository, FakeNotifier()
    )
    try:
        with TestClient(application) as client:
            client.repository = repository
            client.headers.update({"Origin": "https://themiros.com"})
            yield client
    finally:
        get_settings.cache_clear()


def test_hammering_bot_is_blocked_with_cors_headers(landing_client):
    ip = {"X-Forwarded-For": "203.0.113.50"}
    statuses = [
        landing_client.post("/api/v1/demo-requests", json={}, headers=ip).status_code
        for _ in range(10)
    ]
    assert statuses == [422] * 10
    blocked = landing_client.post("/api/v1/demo-requests", json=FORM, headers=ip)
    assert blocked.status_code == 429
    assert blocked.json()["code"] == "DEMO_REQUEST_BLOCKED"
    assert blocked.headers["retry-after"] == "3600"
    # Sans en-tête CORS, le navigateur masquerait le 429 au formulaire.
    assert blocked.headers["access-control-allow-origin"] == "https://themiros.com"
    assert landing_client.repository.rows == []


def test_honeypot_bans_the_address(landing_client):
    ip = {"X-Forwarded-For": "203.0.113.51"}
    trap = landing_client.post(
        "/api/v1/demo-requests", json={**FORM, "website": "x"}, headers=ip
    )
    assert trap.status_code == 201
    retry = landing_client.post("/api/v1/demo-requests", json=FORM, headers=ip)
    assert retry.status_code == 429
    assert landing_client.repository.rows == []


def test_global_limit_stops_a_distributed_bot(landing_client):
    landing_client.app.state.demo_request_guard = AbuseGuard(global_limit=2)
    statuses = [
        landing_client.post(
            "/api/v1/demo-requests",
            json=FORM,
            headers={"X-Forwarded-For": f"203.0.113.{60 + index}"},
        ).status_code
        for index in range(3)
    ]
    assert statuses == [201, 201, 429]
    assert len(landing_client.repository.rows) == 2


def test_oversized_body_is_refused_before_parsing(landing_client):
    response = landing_client.post(
        "/api/v1/demo-requests",
        content=b"{" + b" " * 20_000 + b"}",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
