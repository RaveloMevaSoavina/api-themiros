import json
import re
import time
import unicodedata
from decimal import ROUND_HALF_UP, Decimal
from typing import Protocol

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.modules.frameworks.schemas import (
    GeneratedPillar,
    GeneratePillarsResponse,
    ObservableVariable,
    PillarGenerationContext,
    ReferencePillar,
)

PROMPT_CODE = "generate_pillars"
PROMPT_VERSION = "v1.0"
EX_ANTE_FORBIDDEN_VARIABLES = {
    "activities_completed_share",
    "beneficiaries_reached",
    "disbursement_rate",
    "evidence_of_impact",
    "implementation_delay_months",
    "outcome_achievement_rate",
    "output_delivery_rate",
}
EXPECTED_RESULTS_VARIABLES = {
    "expected_outcomes",
    "expected_results",
    "outcome_indicators_with_targets",
}


class LlmObservableVariable(BaseModel):
    code: str
    label: str
    description: str


class LlmGeneratedPillar(BaseModel):
    name: str
    description: str
    ref_pillar_code: str
    criteria_codes: list[str]
    observable_variables: list[LlmObservableVariable]
    weight: float


class LlmPillarOutput(BaseModel):
    pillars: list[LlmGeneratedPillar]
    rationale: str


class PillarGenerator(Protocol):
    async def generate(
        self, context: PillarGenerationContext
    ) -> GeneratePillarsResponse: ...


class OpenAIPillarGenerator:
    def __init__(
        self,
        settings: Settings,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if not settings.openai_api_key and client is None:
            raise ApiError(
                status_code=503,
                code="AI_PROVIDER_NOT_CONFIGURED",
                detail="OPENAI_API_KEY n'est pas configurée",
            )
        self._settings = settings
        self._client = client or AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=2,
        )

    async def generate(
        self, context: PillarGenerationContext
    ) -> GeneratePillarsResponse:
        started_at = time.perf_counter()
        parameters = context.prompt.parameters
        model = context.prompt.model or self._settings.openai_model
        request_options = {
            "model": model,
            "input": [
                {"role": "system", "content": context.prompt.template},
                {"role": "user", "content": self._user_prompt(context)},
            ],
            "text_format": LlmPillarOutput,
            "max_output_tokens": int(parameters.get("max_tokens", 4000)),
        }
        if "temperature" in parameters:
            request_options["temperature"] = float(parameters["temperature"])

        try:
            response = await self._client.responses.parse(**request_options)
        except (openai.APIConnectionError, openai.APITimeoutError) as error:
            raise self._provider_error(
                "Connexion au fournisseur IA impossible"
            ) from error
        except openai.RateLimitError as error:
            raise self._provider_error("Limite du fournisseur IA atteinte") from error
        except openai.AuthenticationError as error:
            raise ApiError(
                status_code=503,
                code="AI_PROVIDER_AUTHENTICATION_FAILED",
                detail="La clé du fournisseur IA est invalide",
            ) from error
        except openai.APIError as error:
            raise self._provider_error(
                "Le fournisseur IA a rejeté la requête"
            ) from error

        parsed = response.output_parsed
        if parsed is None:
            raise ApiError(
                status_code=502,
                code="AI_PROVIDER_RESPONSE_INVALID",
                detail="Le fournisseur IA n'a retourné aucune sortie exploitable",
            )

        try:
            pillars = [
                GeneratedPillar.model_validate(pillar.model_dump())
                for pillar in parsed.pillars
            ]
            pillars = normalize_weights(pillars)
            validate_business_rules(pillars, context, rationale=parsed.rationale)
            usage = response.usage
            cost = self._cost(usage)
            if (
                self._settings.ai_max_cost_eur > 0
                and cost > self._settings.ai_max_cost_eur
            ):
                raise ApiError(
                    status_code=409,
                    code="AI_COST_LIMIT_EXCEEDED",
                    detail=(
                        "Le coût estimé de la génération dépasse le plafond configuré"
                    ),
                    context={"cost_eur": str(cost)},
                )
            return GeneratePillarsResponse(
                pillars=pillars,
                rationale=parsed.rationale,
                prompt_version=f"{context.prompt.code}_{context.prompt.version}",
                tokens_used=usage.total_tokens if usage else 0,
                cost_eur=cost,
                model=model,
                duration_ms=elapsed_ms(started_at),
                generation_mode="ai",
            )
        except ApiError:
            raise
        except (ValidationError, ValueError) as error:
            raise ApiError(
                status_code=502,
                code="AI_PROVIDER_RESPONSE_INVALID",
                detail="La proposition IA ne respecte pas les règles du cadre",
                context={"reason": str(error)},
            ) from error

    @staticmethod
    def _user_prompt(context: PillarGenerationContext) -> str:
        inputs = context.model_dump(mode="json", exclude={"prompt"})
        return "Contexte de génération (JSON) :\n" + json.dumps(
            inputs,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _cost(self, usage) -> Decimal:
        if usage is None:
            return Decimal("0")
        million = Decimal("1000000")
        input_cost = (
            Decimal(usage.input_tokens)
            * self._settings.openai_input_cost_eur_per_million
            / million
        )
        output_cost = (
            Decimal(usage.output_tokens)
            * self._settings.openai_output_cost_eur_per_million
            / million
        )
        return (input_cost + output_cost).quantize(Decimal("0.000001"))

    @staticmethod
    def _provider_error(detail: str) -> ApiError:
        return ApiError(
            status_code=503,
            code="AI_PROVIDER_UNAVAILABLE",
            detail=detail,
        )


class TemplatePillarGenerator:
    async def generate(
        self,
        context: PillarGenerationContext,
        fallback_reason: str | None = None,
    ) -> GeneratePillarsResponse:
        started_at = time.perf_counter()
        selected = self._select_references(context)
        pillars = [self._to_generated(reference, context) for reference in selected]
        pillars = normalize_weights(pillars)
        validate_business_rules(pillars, context)
        return GeneratePillarsResponse(
            pillars=pillars,
            rationale=(
                "Gabarit de secours issu du référentiel, filtré par cycle, "
                "critères applicables et contexte déclaré."
            ),
            prompt_version=f"{context.prompt.code}_{context.prompt.version}",
            tokens_used=0,
            cost_eur=Decimal("0"),
            model="template",
            duration_ms=elapsed_ms(started_at),
            generation_mode="template",
            fallback_reason=fallback_reason,
        )

    def _select_references(
        self, context: PillarGenerationContext
    ) -> list[ReferencePillar]:
        allowed = {criterion.code for criterion in context.applicable_criteria}
        candidates = [
            pillar
            for pillar in context.ref_pillars
            if set(pillar.criteria_codes) & allowed
        ]
        if len(candidates) < 5:
            raise ApiError(
                status_code=422,
                code="REFERENCE_PILLARS_INSUFFICIENT",
                detail=(
                    "Le référentiel filtré contient moins de cinq piliers utilisables"
                ),
            )
        return sorted(
            candidates,
            key=lambda pillar: self._score(pillar, context),
            reverse=True,
        )[:8]

    @staticmethod
    def _score(pillar: ReferencePillar, context: PillarGenerationContext) -> Decimal:
        applicability_score = {
            "obligatoire": Decimal("4"),
            "prospectif": Decimal("3"),
            "optionnel": Decimal("2"),
        }
        criteria = {item.code: item for item in context.applicable_criteria}
        criterion_score = sum(
            (
                applicability_score.get(criteria[code].applicability, Decimal("1"))
                for code in pillar.criteria_codes
                if code in criteria
            ),
            Decimal("0"),
        )
        context_text = " ".join(
            [
                context.country,
                context.framework_code,
                context.instrument_subtype,
                *context.themes,
                *(str(item.get("name", "")) for item in context.financiers),
            ]
        )
        pillar_text = " ".join(
            [
                pillar.name,
                pillar.description,
                *(str(item.get("label", "")) for item in pillar.observable_variables),
            ]
        )
        overlap = len(tokens(context_text) & tokens(pillar_text))
        return criterion_score + Decimal(overlap * 2) + pillar.default_weight / 100

    @staticmethod
    def _to_generated(
        reference: ReferencePillar, context: PillarGenerationContext
    ) -> GeneratedPillar:
        allowed = {criterion.code for criterion in context.applicable_criteria}
        variables = [
            ObservableVariable.model_validate(variable)
            for variable in reference.observable_variables
            if not (
                context.cycle == "ex_ante"
                and variable.get("code") in EX_ANTE_FORBIDDEN_VARIABLES
            )
        ]
        name = reference.name
        description = reference.description
        if context.cycle == "ex_ante" and any(
            variable.code in EXPECTED_RESULTS_VARIABLES for variable in variables
        ):
            name = "Résultats attendus et dispositif de mesure"
            description = (
                "Évalue la formulation des résultats attendus, leurs cibles et "
                "le dispositif prévu pour les mesurer."
            )
        return GeneratedPillar(
            name=name,
            description=description,
            ref_pillar_code=reference.code,
            criteria_codes=[
                code for code in reference.criteria_codes if code in allowed
            ],
            observable_variables=variables,
            weight=reference.default_weight,
        )


class ResilientPillarGenerator:
    def __init__(
        self,
        primary: PillarGenerator,
        fallback: TemplatePillarGenerator,
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    async def generate(
        self, context: PillarGenerationContext
    ) -> GeneratePillarsResponse:
        try:
            return await self._primary.generate(context)
        except ApiError as error:
            if error.code == "AI_COST_LIMIT_EXCEEDED":
                raise
            return await self._fallback.generate(context, fallback_reason=error.code)
        except Exception as error:
            return await self._fallback.generate(
                context,
                fallback_reason=type(error).__name__,
            )


def normalize_weights(pillars: list[GeneratedPillar]) -> list[GeneratedPillar]:
    total = sum((pillar.weight for pillar in pillars), Decimal("0"))
    if total <= 0:
        raise ValueError("pillar weights must have a positive total")
    normalized: list[GeneratedPillar] = []
    for pillar in pillars:
        weight = (pillar.weight * 100 / total).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        normalized.append(pillar.model_copy(update={"weight": weight}))
    difference = Decimal("100") - sum(
        (pillar.weight for pillar in normalized), Decimal("0")
    )
    if difference:
        index = max(range(len(normalized)), key=lambda item: normalized[item].weight)
        normalized[index] = normalized[index].model_copy(
            update={"weight": normalized[index].weight + difference}
        )
    return normalized


def validate_business_rules(
    pillars: list[GeneratedPillar],
    context: PillarGenerationContext,
    rationale: str | None = None,
) -> None:
    if not 5 <= len(pillars) <= 8:
        raise ValueError("between five and eight pillars are required")
    allowed_criteria = {criterion.code for criterion in context.applicable_criteria}
    allowed_references = {pillar.code for pillar in context.ref_pillars} | {"new"}
    for pillar in pillars:
        invalid_criteria = set(pillar.criteria_codes) - allowed_criteria
        if invalid_criteria:
            raise ValueError(f"criteria not applicable: {sorted(invalid_criteria)}")
        if pillar.ref_pillar_code not in allowed_references:
            raise ValueError(f"unknown ref_pillar_code: {pillar.ref_pillar_code}")
    if any(pillar.ref_pillar_code == "new" for pillar in pillars) and (
        rationale is None or len(rationale.strip()) < 20
    ):
        raise ValueError("new pillars require a contextual rationale")

    if context.cycle == "ex_ante":
        codes = {
            variable.code
            for pillar in pillars
            for variable in pillar.observable_variables
        }
        forbidden = codes & EX_ANTE_FORBIDDEN_VARIABLES
        if forbidden:
            raise ValueError(
                f"ex-ante contains achieved-result variables: {sorted(forbidden)}"
            )
        expected_names = {
            "resultats attendus",
            "expected results",
            "expected outcomes",
        }
        has_expected_results = bool(codes & EXPECTED_RESULTS_VARIABLES) or any(
            any(label in normalized_text(pillar.name) for label in expected_names)
            for pillar in pillars
        )
        if not has_expected_results:
            raise ValueError("ex-ante requires an expected-results pillar")


def tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized_text(value))
        if len(token) > 2
    }


def normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    ).lower()


def elapsed_ms(started_at: float) -> int:
    return max(0, round((time.perf_counter() - started_at) * 1000))


def get_pillar_generator() -> PillarGenerator:
    settings = get_settings()
    fallback = TemplatePillarGenerator()
    if settings.ai_provider == "openai":
        return ResilientPillarGenerator(OpenAIPillarGenerator(settings), fallback)
    return fallback
