from uuid import UUID

import httpx
import pytest

from app.api.dependencies import CurrentUser
from app.core.config import Settings
from app.modules.frameworks.repository import SupabasePillarContextRepository
from app.modules.frameworks.schemas import GeneratePillarsRequest

USER = CurrentUser(
    id=UUID("44444444-4444-4444-8444-444444444444"),
    email="test@example.com",
    token="test-token",
)


def reference_row(locale: str) -> dict:
    return {
        "code": "theory_of_change",
        "name": "Theory of change" if locale == "en" else "Théorie du changement",
        "description": "Description",
        "criteria_codes": ["pertinence"],
        "observable_variables": [],
        "default_weight": "20",
    }


class FakeGateway:
    def __init__(self) -> None:
        self.rpc_calls: list[tuple[str, dict]] = []
        self.internal_calls: list[tuple[str, dict]] = []

    async def rest_get(self, path: str, *, token: str, params: dict) -> list[dict]:
        rows = {
            "workspaces": [
                {
                    "kind": "program",
                    "target_country": "MG",
                    "financiers": [],
                    "themes": [],
                    "declared_stage": "implementation",
                    "start_year": 2024,
                    "end_year": 2028,
                }
            ],
            "approaches": [
                {
                    "id": "22222222-2222-4222-8222-222222222222",
                    "status": "confirmed",
                    "cycle": "en_cours",
                    "instrument_module": "M2",
                    "instrument_subtype": "programme national",
                    "complexity_class": "simple",
                    "locale": "fr",
                    "ref_frameworks": {"code": "themiros"},
                }
            ],
            "approach_criteria": [
                {
                    "applicability": "obligatoire",
                    "weight": "100",
                    "ref_criteria": {"code": "pertinence"},
                }
            ],
        }
        return rows[path]

    async def rpc(self, function_name: str, *, token: str, payload: dict):
        self.rpc_calls.append((function_name, payload))
        return [reference_row(payload["requested_locale"])]

    async def internal_rpc(self, function_name: str, *, payload: dict):
        self.internal_calls.append((function_name, payload))
        return {
            "code": "generate_pillars",
            "version": "v1.1",
            "template": "Generate pillars",
            "model": "test-model",
            "parameters": {},
        }


@pytest.mark.asyncio
async def test_load_requests_reference_pillars_in_every_interface_locale() -> None:
    async with httpx.AsyncClient() as client:
        repository = SupabasePillarContextRepository(Settings(app_env="test"), client)
        gateway = FakeGateway()
        repository._gateway = gateway

        context = await repository.load(
            GeneratePillarsRequest(
                workspace_id=UUID("11111111-1111-4111-8111-111111111111")
            ),
            USER,
        )

    assert context.locale == "fr"
    assert context.translation_locales == ["en"]
    assert context.ref_pillars[0].name == "Théorie du changement"
    assert context.ref_pillar_translations["en"][0].name == "Theory of change"
    assert [payload["requested_locale"] for _, payload in gateway.rpc_calls] == [
        "fr",
        "en",
    ]
    assert gateway.internal_calls[0][1]["prompt_version"] == "v1.1"
