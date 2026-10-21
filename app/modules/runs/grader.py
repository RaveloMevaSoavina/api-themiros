"""Notation IA des sous-questions de la couche B (prompt grade_sub_questions).

Un appel par lot (`layer_b.build_batches`, un seul lot en pratique), en
sortie structurée. Toute défaillance (IA désactivée, réseau, réponse
invalide, plafond de coût) lève `GradingUnavailable` : les critères du lot
restent non conclus, le run continue.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.modules.runs.layer_b import GradingBatch, RawGrade, RawGrading

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


def ai_cost(settings: Settings, input_tokens: int, output_tokens: int) -> Decimal:
    return (
        Decimal(input_tokens) * settings.openai_input_cost_eur_per_million / MILLION
        + Decimal(output_tokens) * settings.openai_output_cost_eur_per_million / MILLION
    ).quantize(Decimal("0.000001"))


class OpenAISubQuestionGrader:
    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self._settings = settings
        # Un seul nouvel essai : le run tient un bail, et une IA indisponible
        # laisse simplement les critères non conclus.
        self._client = client or AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.run_layer_b_timeout_seconds,
            max_retries=1,
        )

    async def grade(
        self, prompt: dict[str, Any], batch: GradingBatch
    ) -> tuple[RawGrading, AiUsage]:
        parameters = prompt.get("parameters") or {}
        model = prompt.get("model") or self._settings.openai_model
        max_output_tokens = min(
            int(parameters.get("max_tokens", 6000)),
            # Marge pour la structure JSON de la réponse.
            batch.max_output_tokens + 500,
        )
        estimate = ai_cost(
            self._settings, batch.estimated_input_tokens, max_output_tokens
        )
        if self._over_budget(estimate):
            raise GradingUnavailable(
                "AI_COST_LIMIT_EXCEEDED",
                "Le coût estimé de la notation dépasse le plafond configuré",
            )

        request: dict[str, Any] = {
            "model": model,
            "input": [
                {"role": "system", "content": prompt["template"]},
                {"role": "user", "content": batch.message()},
            ],
            "text_format": LlmGradingOutput,
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
        except (openai.APIError, ValidationError) as error:
            raise GradingUnavailable(
                "AI_PROVIDER_RESPONSE_INVALID",
                "Le fournisseur IA a rejeté la requête ou rendu une sortie invalide",
            ) from error

        parsed = response.output_parsed
        if parsed is None or getattr(response, "status", None) == "incomplete":
            raise GradingUnavailable(
                "AI_PROVIDER_RESPONSE_INVALID",
                "Le fournisseur IA n'a retourné aucune sortie exploitable",
            )
        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0
        return (
            RawGrading(
                grades=tuple(
                    RawGrade(item.k, item.n, tuple(item.e), item.c) for item in parsed.q
                ),
                justifications={item.k: item.j for item in parsed.c},
            ),
            AiUsage(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_eur=ai_cost(self._settings, input_tokens, output_tokens),
            ),
        )

    def _over_budget(self, cost: Decimal) -> bool:
        return (
            self._settings.ai_max_cost_eur > 0 and cost > self._settings.ai_max_cost_eur
        )


def build_sub_question_grader(settings: Settings) -> SubQuestionGrader | None:
    """`None` quand l'IA est désactivée : la couche B ne note rien."""
    if settings.ai_provider != "openai" or not settings.openai_api_key:
        return None
    return OpenAISubQuestionGrader(settings)
