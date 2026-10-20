"""Exécution des runs d'analyse (E6) : classification figée, variables
intermédiaires, couche A, restitution.

Tourne dans le processus du worker d'ingestion. Les couches B et C arrivent
aux sprints suivants : leurs étapes sont marquées « skipped ». Un échec passe
le run en `failed` sans rien publier (RG-6.1).
"""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime

from app.core.config import Settings
from app.core.errors import ApiError
from app.modules.runs.domain import (
    ENGINE_VERSION,
    RunInputs,
    ScoringRules,
    compute_layer_a,
)
from app.modules.runs.repository import ClaimedRun, RunRepository

logger = logging.getLogger(__name__)

NOT_AVAILABLE = {"reason": "not_available"}


class RunFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class RunWorker:
    def __init__(
        self, repository: RunRepository, settings: Settings, *, worker_id: str
    ) -> None:
        self._repository = repository
        self._settings = settings
        self._worker_id = f"{worker_id}#runs"
        self._stopping = asyncio.Event()

    def stop(self) -> None:
        self._stopping.set()

    async def run(self) -> None:
        while not self._stopping.is_set():
            try:
                worked = await self.run_once()
            except Exception:
                logger.exception("Unable to claim an analysis run")
                worked = False
            if worked:
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=self._settings.run_poll_seconds
                )

    async def run_once(self) -> bool:
        """Traite un run s'il y en a un ; `False` si la file est vide."""
        run = await self._repository.claim(self._worker_id, ENGINE_VERSION)
        if run is None:
            return False
        await self.process(run)
        return True

    def rules(self, run: ClaimedRun) -> ScoringRules:
        """Seuils du moteur, et règle des sources indépendantes figée au
        lancement avec les paramètres de masse critique de l'espace."""
        settings = run.rules_applied.get("critical_mass_settings") or {}
        return ScoringRules(
            strong_similarity=self._settings.run_strong_similarity,
            min_independent_sources=int(settings.get("min_independent_sources") or 2),
            max_evidences_per_variable=self._settings.run_max_evidences_per_variable,
        )

    async def process(self, run: ClaimedRun) -> None:
        step = "classification"
        try:
            await self._repository.step(run, step, "running")
            inputs = await self._repository.inputs(run)
            if inputs.state == "pending" and self._may_wait(inputs):
                await self._repository.defer(
                    run, self._settings.run_defer_seconds, "classification_pending"
                )
                logger.info("Run %s waits for the corpus classification", run.run_id)
                return
            self._check_classification(inputs)
            await self._repository.step(
                run,
                step,
                "completed",
                {
                    **inputs.segments,
                    "unclassified": inputs.segments.get("published", 0)
                    - inputs.segments.get("classified", 0),
                    "unassigned_documents": inputs.unassigned_documents,
                },
            )

            step = "variables"
            await self._repository.step(run, step, "running")
            rules = self.rules(run)
            results = compute_layer_a(inputs, rules)
            summary = results.summary()
            await self._repository.step(run, step, "completed", summary["variables"])

            step = "layer_a"
            await self._repository.step(run, step, "completed", summary["layer_a"])
            await self._repository.step(run, "layer_b", "skipped", NOT_AVAILABLE)
            await self._repository.step(run, "layer_c", "skipped", NOT_AVAILABLE)

            step = "restitution"
            await self._repository.step(run, step, "running")
            written = await self._repository.complete(run, results.payload(rules))
            logger.info(
                "Run %s completed: %s variables, %s evidences, %s scores",
                run.run_id,
                written.get("variables"),
                written.get("evidences"),
                written.get("scores"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if isinstance(error, ApiError) and error.code == "RUN_LEASE_LOST":
                logger.warning("Run %s taken over by another worker", run.run_id)
                return
            if isinstance(error, RunFailure):
                code, message = error.code, error.message
            elif isinstance(error, ApiError):
                code, message = error.code, error.detail
            else:
                logger.exception("Run %s failed at step %s", run.run_id, step)
                code, message = "RUN_FAILED", "Le calcul de l'analyse a échoué"
            logger.warning("Run %s failed at step %s: %s", run.run_id, step, code)
            try:
                await self._repository.fail(run, step, code, message)
            except Exception:
                logger.exception("Unable to record the failure of run %s", run.run_id)

    def _may_wait(self, inputs: RunInputs) -> bool:
        waited = (datetime.now(UTC) - inputs.created_at).total_seconds()
        return waited < self._settings.run_classification_wait_seconds

    @staticmethod
    def _check_classification(inputs: RunInputs) -> None:
        if inputs.state == "pending":
            raise RunFailure(
                "CLASSIFICATION_TIMEOUT",
                "La classification du corpus n'a pas abouti dans le délai imparti",
            )
        if inputs.state == "failed":
            raise RunFailure(
                "CLASSIFICATION_FAILED", "La classification du corpus a échoué"
            )
        if inputs.state == "framework_changed":
            raise RunFailure(
                "FRAMEWORK_CHANGED",
                "Le cadre a été revalidé depuis le lancement : relancez l'analyse",
            )
