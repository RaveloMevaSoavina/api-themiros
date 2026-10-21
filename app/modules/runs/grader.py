"""Appels IA du moteur de runs, en sortie structurée :

- notation des sous-questions de la couche B (prompt grade_sub_questions) ;
- qualification des écarts de la couche C (prompt detect_incoherence).

Un appel par lot (un seul lot en pratique). Toute défaillance (IA
désactivée, réseau, réponse invalide, plafond de coût) lève
`GradingUnavailable` : les critères du lot restent non conclus, les écarts
gardent leur qualification par défaut ; le run continue.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.modules.runs.layer_b import GradingBatch, RawGrade, RawGrading
from app.modules.runs.layer_c import QualificationBatch, RawQualification

MILLION = Decimal("1000000")


class GradingUnavailable(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# Schéma de sortie compact : identifiants courts, une lettre par champ.
class LlmGrade(BaseModel):
    k: str
    n: int | None
    e: list[str]
    c: bool


class LlmCriterionJustification(BaseModel):
    k: str
    j: str


class LlmGradingOutput(BaseModel):
    q: list[LlmGrade]
    c: list[LlmCriterionJustification]


class LlmAlert(BaseModel):
    k: str
    ok: bool
    g: str
    m: str
    e: list[str]


class LlmQualificationOutput(BaseModel):
    a: list[LlmAlert]


@dataclass(frozen=True, slots=True)
class AiUsage:
    model: str
    input_tokens: int
    output_tokens: int
    cost_eur: Decimal


class SubQuestionGrader(Protocol):
    async def grade(
        self, prompt: dict[str, Any], batch: GradingBatch
    ) -> tuple[RawGrading, AiUsage]: ...


class AlertQualifier(Protocol):
    async def qualify(
        self, prompt: dict[str, Any], batch: QualificationBatch
    ) -> tuple[list[RawQualification], AiUsage]: ...


def ai_cost(settings: Settings, input_tokens: int, output_tokens: int) -> Decimal:
    return (
        Decimal(input_tokens) * settings.openai_input_cost_eur_per_million / MILLION
        + Decimal(output_tokens) * settings.openai_output_cost_eur_per_million / MILLION
    ).quantize(Decimal("0.000001"))


class OpenAIStructuredCall:
    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self._settings = settings
        # Un seul nouvel essai : le run tient un bail, et une IA indisponible
        # ne fait pas échouer le run.
        self._client = client or AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.run_layer_b_timeout_seconds,
            max_retries=1,
        )

    async def _parse[T: BaseModel](
        self,
        prompt: dict[str, Any],
        message: str,
        text_format: type[T],
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        default_max_tokens: int,
    ) -> tuple[T, AiUsage]:
        parameters = prompt.get("parameters") or {}
        model = prompt.get("model") or self._settings.openai_model
        # Plafond du prompt, pas l'estimation du lot : le raisonnement du
        # modèle consomme le même budget et tronquerait la sortie JSON.
        max_output_tokens = max(
            int(parameters.get("max_tokens", default_max_tokens)),
            estimated_output_tokens + 500,
        )
        estimate = ai_cost(self._settings, estimated_input_tokens, max_output_tokens)
        if self._over_budget(estimate):
            raise GradingUnavailable(
                "AI_COST_LIMIT_EXCEEDED",
                "Le coût estimé de l'appel IA dépasse le plafond configuré",
            )

        request: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": prompt["template"]},
                {"role": "user", "content": message},
            ],
            "text_format": text_format,
            "max_output_tokens": max_output_tokens,
        }
        if "temperature" in parameters:
            request["temperature"] = float(parameters["temperature"])

        try:
            response = await self._client.responses.parse(**request)
        except (openai.APIConnectionError, openai.APITimeoutError) as error:
            raise GradingUnavailable(
                "AI_PROVIDER_UNAVAILABLE", "Connexion au fournisseur IA impossible"
            ) from error
        except openai.RateLimitError as error:
            raise GradingUnavailable(
                "AI_PROVIDER_UNAVAILABLE", "Limite du fournisseur IA atteinte"
            ) from error
        except openai.AuthenticationError as error:
            raise GradingUnavailable(
                "AI_PROVIDER_AUTHENTICATION_FAILED",
                "La clé du fournisseur IA est invalide",
            ) from error
        except openai.APIError as error:
            raise GradingUnavailable(
                "AI_PROVIDER_REJECTED",
                f"Le fournisseur IA a rejeté la requête : {error.message}"[:300],
            ) from error
        except ValidationError as error:
            # Le SDK valide le JSON produit : une sortie coupée échoue ici.
            raise GradingUnavailable(
                "AI_OUTPUT_TRUNCATED",
                "La sortie IA est incomplète ou ne suit pas le schéma",
            ) from error

        parsed = response.output_parsed
        if getattr(response, "status", None) == "incomplete":
            details = getattr(response, "incomplete_details", None)
            raise GradingUnavailable(
                "AI_OUTPUT_TRUNCATED",
                f"La sortie IA est incomplète ({getattr(details, 'reason', None)})",
            )
        if parsed is None:
            raise GradingUnavailable(
                "AI_PROVIDER_RESPONSE_INVALID",
                "Le fournisseur IA n'a retourné aucune sortie exploitable",
            )
        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        return parsed, AiUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_eur=ai_cost(self._settings, input_tokens, output_tokens),
        )

    def _over_budget(self, cost: Decimal) -> bool:
        return (
            self._settings.ai_max_cost_eur > 0 and cost > self._settings.ai_max_cost_eur
        )


class OpenAISubQuestionGrader(OpenAIStructuredCall):
    async def grade(
        self, prompt: dict[str, Any], batch: GradingBatch
    ) -> tuple[RawGrading, AiUsage]:
        parsed, usage = await self._parse(
            prompt,
            batch.message(),
            LlmGradingOutput,
            batch.estimated_input_tokens,
            batch.max_output_tokens,
            6000,
        )
        return (
            RawGrading(
                grades=tuple(
                    RawGrade(item.k, item.n, tuple(item.e), item.c) for item in parsed.q
                ),
                justifications={item.k: item.j for item in parsed.c},
            ),
            usage,
        )


class OpenAIAlertQualifier(OpenAIStructuredCall):
    async def qualify(
        self, prompt: dict[str, Any], batch: QualificationBatch
    ) -> tuple[list[RawQualification], AiUsage]:
        parsed, usage = await self._parse(
            prompt,
            batch.message(),
            LlmQualificationOutput,
            batch.estimated_input_tokens,
            batch.max_output_tokens,
            2000,
        )
        return (
            [
                RawQualification(item.k, item.ok, item.g, item.m, tuple(item.e))
                for item in parsed.a
            ],
            usage,
        )


def build_sub_question_grader(settings: Settings) -> SubQuestionGrader | None:
    """`None` quand l'IA est désactivée : la couche B ne note rien."""
    if settings.ai_provider != "openai" or not settings.openai_api_key:
        return None
    return OpenAISubQuestionGrader(settings)


def build_alert_qualifier(settings: Settings) -> AlertQualifier | None:
    """`None` quand l'IA est désactivée : les écarts gardent leur
    qualification par défaut."""
    if settings.ai_provider != "openai" or not settings.openai_api_key:
        return None
    return OpenAIAlertQualifier(settings)
