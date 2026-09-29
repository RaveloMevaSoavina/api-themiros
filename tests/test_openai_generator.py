from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.errors import ApiError
from app.modules.frameworks.generator import (
    LlmGeneratedPillar,
    LlmObservableVariable,
    LlmPillarOutput,
    OpenAIPillarGenerator,
    ResilientPillarGenerator,
    TemplatePillarGenerator,
)
from app.modules.frameworks.schemas import (
    ApplicableCriterion,
    PillarGenerationContext,
    PromptDefinition,
    ReferencePillar,
)


class FakeResponses:
    def __init__(self, output: LlmPillarOutput) -> None:
        self._output = output
        self.last_request = None

    async def parse(self, **kwargs):
        self.last_request = kwargs
        return SimpleNamespace(
            output_parsed=self._output,
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=500,
                total_tokens=1500,
            ),
        )


class FakeOpenAI:
    def __init__(self, output: LlmPillarOutput) -> None:
        self.responses = FakeResponses(output)


def context() -> PillarGenerationContext:
    return PillarGenerationContext(
        approach_id="22222222-2222-4222-8222-222222222222",
        object_type="program",
        country="MG",
        financiers=[{"name": "GCF", "principal": True}],
        themes=["adaptation"],
        stage="implementation",
        start_year=2024,
        end_year=2028,
        cycle="en_cours",
        instrument_module="M2",
        instrument_subtype="programme national",
        complexity_class="complique",
        framework_code="gcf",
        applicable_criteria=[
            ApplicableCriterion(
                code="pertinence",
                applicability="obligatoire",
                weight=Decimal("100"),
            )
        ],
        ref_pillars=[
            ReferencePillar(
                code=f"pillar_{index}",
                name=f"Pilier {index}",
                description="Description",
                criteria_codes=["pertinence"],
                observable_variables=[
                    {
                        "code": f"reference_variable_{index}",
                        "label": f"Variable {index}",
                        "description": "Variable observable",
                    }
                ],
                default_weight=Decimal("20"),
            )
            for index in range(1, 6)
        ],
        prompt=PromptDefinition(
            code="generate_pillars",
            version="v1.0",
            template="Generate structured pillars",
            model="gpt-6-astra",
            parameters={"temperature": 0.3, "max_tokens": 4000},
        ),
    )


def output(criteria_code: str = "pertinence") -> LlmPillarOutput:
    return LlmPillarOutput(
        pillars=[
            LlmGeneratedPillar(
                name=f"Pilier {index}",
                description="Description contextualisée",
                ref_pillar_code=f"pillar_{index}",
                criteria_codes=[criteria_code],
                observable_variables=[
                    LlmObservableVariable(
                        code=f"variable_{index}",
                        label=f"Variable {index}",
                        description="Variable observable",
                    )
                ],
                weight=20,
            )
            for index in range(1, 6)
        ],
        rationale="Adaptation au contexte",
    )


@pytest.mark.asyncio
async def test_openai_provider_returns_validated_structured_output() -> None:
    client = FakeOpenAI(output())
    settings = Settings(
        app_env="test",
        ai_provider="openai",
        openai_api_key="test-key",
        openai_input_cost_eur_per_million=Decimal("2"),
        openai_output_cost_eur_per_million=Decimal("8"),
    )
    generator = OpenAIPillarGenerator(settings, client=client)

    result = await generator.generate(context())

    assert len(result.pillars) == 5
    assert sum(pillar.weight for pillar in result.pillars) == Decimal("100")
    assert result.tokens_used == 1500
    assert result.cost_eur == Decimal("0.006000")
    assert result.generation_mode == "ai"
    assert client.responses.last_request["text_format"] is LlmPillarOutput


@pytest.mark.asyncio
async def test_openai_provider_rejects_non_applicable_criterion() -> None:
    generator = OpenAIPillarGenerator(
        Settings(app_env="test", openai_api_key="test-key"),
        client=FakeOpenAI(output("impact")),
    )

    with pytest.raises(ApiError) as captured:
        await generator.generate(context())

    assert captured.value.code == "AI_PROVIDER_RESPONSE_INVALID"


@pytest.mark.asyncio
async def test_provider_normalizes_weights_to_one_hundred() -> None:
    generated = output()
    for pillar in generated.pillars:
        pillar.weight = 10
    generator = OpenAIPillarGenerator(
        Settings(app_env="test", openai_api_key="test-key"),
        client=FakeOpenAI(generated),
    )

    result = await generator.generate(context())

    assert sum(pillar.weight for pillar in result.pillars) == Decimal("100")


@pytest.mark.asyncio
async def test_provider_uses_template_when_ai_output_is_invalid() -> None:
    resilient = ResilientPillarGenerator(
        OpenAIPillarGenerator(
            Settings(app_env="test", openai_api_key="test-key"),
            client=FakeOpenAI(output("impact")),
        ),
        TemplatePillarGenerator(),
    )

    result = await resilient.generate(context())

    assert result.generation_mode == "template"
    assert result.fallback_reason == "AI_PROVIDER_RESPONSE_INVALID"


@pytest.mark.asyncio
async def test_ex_ante_requires_expected_results_pillar() -> None:
    generator = OpenAIPillarGenerator(
        Settings(app_env="test", openai_api_key="test-key"),
        client=FakeOpenAI(output()),
    )
    ex_ante_context = context().model_copy(update={"cycle": "ex_ante"})

    with pytest.raises(ApiError) as captured:
        await generator.generate(ex_ante_context)

    assert captured.value.code == "AI_PROVIDER_RESPONSE_INVALID"


@pytest.mark.asyncio
async def test_ex_ante_accepts_expected_results_without_achieved_results() -> None:
    generated = output()
    generated.pillars[0].name = "Résultats attendus"
    generated.pillars[0].observable_variables[
        0
    ].code = "outcome_indicators_with_targets"
    generator = OpenAIPillarGenerator(
        Settings(app_env="test", openai_api_key="test-key"),
        client=FakeOpenAI(generated),
    )

    result = await generator.generate(context().model_copy(update={"cycle": "ex_ante"}))

    assert result.generation_mode == "ai"
