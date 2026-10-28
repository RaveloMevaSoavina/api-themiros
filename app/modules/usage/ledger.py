"""Registre de la consommation IA (`ai_usage`, migration 20261028000002).

Chaque traitement qui appelle OpenAI y inscrit une ligne : son espace, son
run ou son document, la source (`ingestion`, `classification`,
`layer_b_search`, `layer_b`, `layer_c`, `restitution` ; `pillar_generation`
est inscrite en SQL), le modèle, le nombre d'appels, les tokens facturés
(chiffres de l'API, pas des estimations) et le coût.

Les embeddings sont comptés par `OpenAIEmbedder` dans le compteur du
contexte courant : `UsageLedger.embeddings` en ouvre un pour un document, un
cadre ou un run, et inscrit le total à la sortie, même en cas d'échec (les
appels réussis sont facturés). Les tâches lancées dans ce contexte
(`asyncio.gather`) partagent le compteur ; deux documents traités en
parallèle ont chacun le leur.

Une inscription qui échoue est journalisée sans interrompre le traitement :
le compte de la consommation ne doit pas faire échouer l'analyse.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal
from uuid import UUID

from app.core.config import Settings
from app.infrastructure.supabase import SupabaseGateway

logger = logging.getLogger(__name__)

MILLION = Decimal(1_000_000)
UsageKind = Literal["completion", "embedding"]


@dataclass(slots=True)
class EmbeddingMeter:
    calls: int = 0
    tokens: int = 0


_meter: ContextVar[EmbeddingMeter | None] = ContextVar("embedding_meter", default=None)


def count_embedding(tokens: int) -> None:
    """Appel d'embeddings réussi, compté dans le compteur ouvert s'il y en a."""
    meter = _meter.get()
    if meter is not None:
        meter.calls += 1
        meter.tokens += tokens


def embedding_cost(settings: Settings, tokens: int) -> Decimal:
    return (
        Decimal(tokens) * settings.openai_embedding_cost_eur_per_million / MILLION
    ).quantize(Decimal("0.000001"))


class UsageLedger:
    """Sans passerelle (tests, outils), rien n'est inscrit."""

    def __init__(self, gateway: SupabaseGateway | None, settings: Settings) -> None:
        self._gateway = gateway
        self._settings = settings

    async def record(
        self,
        *,
        workspace_id: UUID,
        source: str,
        kind: UsageKind,
        model: str | None,
        calls: int,
        input_tokens: int,
        output_tokens: int,
        cost_eur: Decimal,
        run_id: UUID | None = None,
        document_id: UUID | None = None,
    ) -> None:
        if self._gateway is None or calls == 0:
            return
        try:
            await self._gateway.service_insert(
                "ai_usage",
                payload={
                    "workspace_id": str(workspace_id),
                    "run_id": str(run_id) if run_id else None,
                    "document_id": str(document_id) if document_id else None,
                    "source": source,
                    "kind": kind,
                    "model": model,
                    "calls": calls,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_eur": str(cost_eur),
                },
            )
        except Exception:
            logger.exception(
                "Unable to record AI usage (%s, workspace %s)", source, workspace_id
            )

    @asynccontextmanager
    async def embeddings(
        self,
        *,
        workspace_id: UUID,
        source: str,
        model: str,
        run_id: UUID | None = None,
        document_id: UUID | None = None,
    ) -> AsyncIterator[EmbeddingMeter]:
        meter = EmbeddingMeter()
        token = _meter.set(meter)
        try:
            yield meter
        finally:
            _meter.reset(token)
            await self.record(
                workspace_id=workspace_id,
                source=source,
                kind="embedding",
                model=model,
                calls=meter.calls,
                input_tokens=meter.tokens,
                output_tokens=0,
                cost_eur=embedding_cost(self._settings, meter.tokens),
                run_id=run_id,
                document_id=document_id,
            )
