from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

# Langues de l'interface (dashboard-themiros, supportedLanguages) : un pilier
# est rédigé dans la langue de l'approche et traduit dans les autres.
INTERFACE_LOCALES = ("fr", "en")


class GeneratePillarsRequest(BaseModel):
    workspace_id: UUID
    approach_id: UUID | None = None
    job_id: UUID | None = None


class ApplicableCriterion(BaseModel):
    code: str
    applicability: str
    weight: Decimal


class ReferencePillar(BaseModel):
    code: str
    name: str
    description: str
    criteria_codes: list[str]
    observable_variables: list[dict]
    default_weight: Decimal


class PromptDefinition(BaseModel):
    code: str
    version: str
    template: str
    model: str
    parameters: dict


class PillarGenerationContext(BaseModel):
    approach_id: UUID
    object_type: str
    country: str
    financiers: list[dict]
    themes: list[str]
    stage: str
    start_year: int
    end_year: int
    cycle: str
    instrument_module: str
    instrument_subtype: str
    complexity_class: str
    framework_code: str
    locale: str = "fr"
    translation_locales: list[str] = Field(default_factory=list)
    applicable_criteria: list[ApplicableCriterion]
    ref_pillars: list[ReferencePillar]
    # Piliers du référentiel dans chaque langue de translation_locales.
    ref_pillar_translations: dict[str, list[ReferencePillar]] = Field(
        default_factory=dict
    )
    prompt: PromptDefinition


class ObservableVariable(BaseModel):
    code: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    label: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=300)


class VariableTranslation(BaseModel):
    code: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    label: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=300)


class PillarTranslation(BaseModel):
    locale: str = Field(pattern=r"^(fr|en|pt|es)$")
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=200)
    observable_variables: list[VariableTranslation] = Field(default_factory=list)


class GeneratedPillar(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=200)
    ref_pillar_code: str = Field(min_length=1, max_length=80)
    criteria_codes: list[str] = Field(min_length=1)
    observable_variables: list[ObservableVariable] = Field(min_length=1)
    weight: Decimal = Field(gt=0, le=100, decimal_places=2)
    translations: list[PillarTranslation] = Field(default_factory=list)


class GeneratePillarsResponse(BaseModel):
    pillars: list[GeneratedPillar] = Field(min_length=5, max_length=8)
    rationale: str = Field(min_length=1)
    prompt_version: str
    tokens_used: int = Field(ge=0)
    # Détail facturé (registre ai_usage) ; tokens_used en reste la somme.
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_eur: Decimal = Field(ge=0, decimal_places=6)
    model: str
    duration_ms: int = Field(ge=0)
    generation_mode: str = Field(pattern=r"^(ai|template)$")
    fallback_reason: str | None = None

    @model_validator(mode="after")
    def weights_must_total_one_hundred(self):
        total = sum((pillar.weight for pillar in self.pillars), Decimal("0"))
        if total != Decimal("100"):
            raise ValueError("pillar weights must total exactly 100")
        return self
