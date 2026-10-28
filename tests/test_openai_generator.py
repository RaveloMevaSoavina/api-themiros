from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.core.errors import ApiError
from app.modules.frameworks.generator import (
    LlmGeneratedPillar,
    LlmObservableVariable,
    LlmPillarOutput,
    LlmPillarTranslation,
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
                translations=[],
            )
            for index in range(1, 6)
        ],
        rationale="Adaptation au contexte",
    )


def reference_pillar(index: int, locale: str) -> ReferencePillar:
    english = locale == "en"
    return ReferencePillar(
        code=f"pillar_{index}",
        name=f"{'Pillar' if english else 'Pilier'} {index}",
        description="Description",
        criteria_codes=["pertinence"],
        observable_variables=[
            {
                "code": f"reference_variable_{index}",
                "label": f"{'Variable EN' if english else 'Variable'} {index}",
                "description": (
                    "Observable variable" if english else "Variable observable"
                ),
            }
        ],
        default_weight=Decimal("20"),
    )


def translated_context() -> PillarGenerationContext:
    return context().model_copy(
        update={
            "locale": "fr",
            "translation_locales": ["en"],
            "ref_pillar_translations": {
                "en": [reference_pillar(index, "en") for index in range(1, 6)]
            },
        }
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
    assert (result.input_tokens, result.output_tokens) == (1000, 500)
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


@pytest.mark.asyncio
async def test_provider_keeps_valid_translations_and_drops_invalid_ones() -> None:
    generated = output()
    generated.pillars[0].name = "Pilier adapté"
    generated.pillars[0].translations = [
        LlmPillarTranslation(
            locale="en",
            name="Adapted pillar",
            description="Contextualised description",
            observable_variables=[
                LlmObservableVariable(
                    code="variable_1",
                    label="Variable one",
                    description="Observable variable",
                ),
                LlmObservableVariable(
                    code="unknown_variable",
                    label="Unknown",
                    description="",
                ),
            ],
        ),
        LlmPillarTranslation(
            locale="de",
            name="Angepasste Säule",
            description="Beschreibung",
            observable_variables=[],
        ),
    ]
    generated.pillars[1].translations = [
        LlmPillarTranslation(
            locale="en",
            name="x" * 200,
            description="Too long name",
            observable_variables=[],
        )
    ]
    generator = OpenAIPillarGenerator(
        Settings(app_env="test", openai_api_key="test-key"),
        client=FakeOpenAI(generated),
    )

    result = await generator.generate(translated_context())

    first, second = result.pillars[0], result.pillars[1]
    assert result.generation_mode == "ai"
    assert [item.locale for item in first.translations] == ["en"]
    assert first.translations[0].name == "Adapted pillar"
    assert [item.code for item in first.translations[0].observable_variables] == [
        "variable_1"
    ]
    assert second.translations == []


@pytest.mark.asyncio
async def test_unchanged_reference_text_uses_reference_translation() -> None:
    generated = output()
    generated.pillars[0].description = "Description"
    generated.pillars[0].observable_variables = [
        LlmObservableVariable(
            code="reference_variable_1",
            label="Variable 1",
            description="Variable observable",
        )
    ]
    generated.pillars[0].translations = [
        LlmPillarTranslation(
            locale="en",
            name="Free translation",
            description="Free description",
            observable_variables=[],
        )
    ]
    generator = OpenAIPillarGenerator(
        Settings(app_env="test", openai_api_key="test-key"),
        client=FakeOpenAI(generated),
    )

    result = await generator.generate(translated_context())

    translation = result.pillars[0].translations[0]
    assert translation.name == "Pillar 1"
    assert translation.description == "Description"
    assert translation.observable_variables[0].label == "Variable EN 1"


@pytest.mark.asyncio
async def test_prompt_omits_reference_translations() -> None:
    client = FakeOpenAI(output())
    generator = OpenAIPillarGenerator(
        Settings(app_env="test", openai_api_key="test-key"),
        client=client,
    )

    await generator.generate(translated_context())

    user_prompt = client.responses.last_request["input"][1]["content"]
    assert '"translation_locales":["en"]' in user_prompt
    assert "ref_pillar_translations" not in user_prompt


@pytest.mark.asyncio
async def test_template_translates_from_reference() -> None:
    result = await TemplatePillarGenerator().generate(translated_context())

    for pillar in result.pillars:
        assert [item.locale for item in pillar.translations] == ["en"]
        translation = pillar.translations[0]
        assert translation.name == pillar.name.replace("Pilier", "Pillar")
        assert translation.observable_variables[0].label.startswith("Variable EN")


@pytest.mark.asyncio
async def test_template_writes_expected_results_pillar_in_approach_locale() -> None:
    english_context = translated_context()
    english_context = english_context.model_copy(
        update={
            "cycle": "ex_ante",
            "locale": "en",
            "translation_locales": ["fr"],
            "ref_pillars": [
                reference.model_copy(
                    update={
                        "observable_variables": [
                            {
                                "code": "outcome_indicators_with_targets",
                                "label": "Targets",
                                "description": "Targets are set",
                            }
                        ]
                    }
                )
                for reference in english_context.ref_pillars
            ],
            "ref_pillar_translations": {},
        }
    )

    result = await TemplatePillarGenerator().generate(english_context)

    assert result.pillars[0].name == "Expected results and measurement system"
    assert result.pillars[0].translations[0].locale == "fr"
    assert result.pillars[0].translations[0].name == (
        "Résultats attendus et dispositif de mesure"
    )
