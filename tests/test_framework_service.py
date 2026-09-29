from decimal import Decimal
from uuid import UUID

import pytest

from app.api.dependencies import CurrentUser
from app.modules.frameworks.schemas import (
    ApplicableCriterion,
    GeneratedPillar,
    GeneratePillarsRequest,
    GeneratePillarsResponse,
    ObservableVariable,
    PillarGenerationContext,
    PromptDefinition,
    ReferencePillar,
)
from app.modules.frameworks.service import PillarGenerationService


def generation_context() -> PillarGenerationContext:
    return PillarGenerationContext(
        approach_id="22222222-2222-4222-8222-222222222222",
        object_type="program",
        country="MG",
        financiers=[],
        themes=["adaptation"],
        stage="implementation",
        start_year=2024,
        end_year=2028,
        cycle="en_cours",
        instrument_module="M2",
        instrument_subtype="programme national",
        complexity_class="simple",
        framework_code="themiros",
        applicable_criteria=[
            ApplicableCriterion(
                code="pertinence",
                applicability="obligatoire",
                weight=Decimal("100"),
            )
        ],
        ref_pillars=[
            ReferencePillar(
                code="reference",
                name="Référence",
                description="Description",
                criteria_codes=["pertinence"],
                observable_variables=[],
                default_weight=Decimal("20"),
            )
        ],
        prompt=PromptDefinition(
            code="generate_pillars",
            version="v1.0",
            template="Generate pillars",
            model="test-model",
            parameters={},
        ),
    )


def generation_result() -> GeneratePillarsResponse:
    return GeneratePillarsResponse(
        pillars=[
            GeneratedPillar(
                name=f"Pilier {index}",
                description="Description",
                ref_pillar_code="reference",
                criteria_codes=["pertinence"],
                observable_variables=[
                    ObservableVariable(
                        code=f"variable_{index}",
                        label=f"Variable {index}",
                    )
                ],
                weight=Decimal("20"),
            )
            for index in range(1, 6)
        ],
        rationale="Contexte",
        prompt_version="generate_pillars_v1.0",
        tokens_used=100,
        cost_eur=Decimal("0"),
        model="test-model",
        duration_ms=10,
        generation_mode="ai",
    )


class FakeRepository:
    def __init__(self) -> None:
        self.running = False
        self.persisted = None
        self.failed = None

    async def load(self, request, user):
        return generation_context()

    async def mark_running(self, request):
        self.running = True

    async def persist(self, request, approach_id, result):
        self.persisted = result

    async def mark_failed(self, request, message):
        self.failed = message


class SuccessfulGenerator:
    async def generate(self, context):
        return generation_result()


class FailingGenerator:
    async def generate(self, context):
        raise RuntimeError("provider failed")


REQUEST = GeneratePillarsRequest(
    workspace_id=UUID("11111111-1111-4111-8111-111111111111"),
    approach_id=UUID("22222222-2222-4222-8222-222222222222"),
    job_id=UUID("33333333-3333-4333-8333-333333333333"),
)
USER = CurrentUser(
    id=UUID("44444444-4444-4444-8444-444444444444"),
    email="test@example.com",
    token="test-token",
)


@pytest.mark.asyncio
async def test_service_persists_generated_pillars() -> None:
    repository = FakeRepository()
    service = PillarGenerationService(SuccessfulGenerator(), repository)

    result = await service.generate(REQUEST, USER)

    assert repository.running is True
    assert repository.persisted["pillars"][0]["name"] == "Pilier 1"
    assert result.tokens_used == 100


@pytest.mark.asyncio
async def test_service_marks_job_failed_when_provider_fails() -> None:
    repository = FakeRepository()
    service = PillarGenerationService(FailingGenerator(), repository)

    with pytest.raises(RuntimeError, match="provider failed"):
        await service.generate(REQUEST, USER)

    assert repository.failed == "provider failed"
