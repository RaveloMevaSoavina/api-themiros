from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


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
    applicable_criteria: list[ApplicableCriterion]
    ref_pillars: list[ReferencePillar]
    prompt: PromptDefinition


class ObservableVariable(BaseModel):
    code: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    label: str = Field(min_length=1, max_length=160)
    description: str | None = Field(default=None, max_length=300)


class GeneratedPillar(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=200)
    ref_pillar_code: str = Field(min_length=1, max_length=80)
    criteria_codes: list[str] = Field(min_length=1)
    observable_variables: list[ObservableVariable] = Field(min_length=1)
    weight: Decimal = Field(gt=0, le=100, decimal_places=2)


class GeneratePillarsResponse(BaseModel):
    pillars: list[GeneratedPillar] = Field(min_length=5, max_length=8)
    rationale: str = Field(min_length=1)
    prompt_version: str
    tokens_used: int = Field(ge=0)
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
