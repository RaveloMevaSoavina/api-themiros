"""Exécution des runs d'analyse (E6) : classification figée, variables
intermédiaires, couche A, couche B, couche C, restitution.

Tourne dans le processus du worker d'ingestion, avec son modèle
d'embeddings (recherche des extraits de la couche B). Un échec passe le run
en `failed` sans rien publier (RG-6.1) ; une IA indisponible n'est pas un
échec : les critères de la couche B restent non conclus, les écarts de la
couche C gardent leur qualification par défaut, la synthèse ses libellés
factuels, sans conclusion ni plan d'action.
"""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime
from decimal import Decimal

from app.core.config import Settings
from app.core.errors import ApiError
from app.modules.ingestion.analysis.embeddings import Embedder
from app.modules.runs.domain import (
    ENGINE_VERSION,
    LayerAResults,
    RunInputs,
    ScoringRules,
    compute_layer_a,
)
from app.modules.runs.grader import (
    AiUsage,
    AlertQualifier,
    GradingUnavailable,
    RestitutionWriter,
    SubQuestionGrader,
)
from app.modules.runs.layer_b import (
    LAYER_B_VERSION,
    PROMPT_CODE,
    PROMPT_VERSION,
    Grading,
    GradingBatch,
    LayerBResults,
    LayerBRules,
    build_batches,
    finalize,
    interpret,
    retrieve,
)
from app.modules.runs.layer_c import (
    LAYER_C_VERSION,
    LayerCInputs,
    LayerCResults,
    LayerCRules,
    detect,
    known_texts,
)
from app.modules.runs.layer_c import PROMPT_CODE as ALERT_PROMPT_CODE
from app.modules.runs.layer_c import PROMPT_VERSION as ALERT_PROMPT_VERSION
from app.modules.runs.layer_c import build_batches as build_alert_batches
from app.modules.runs.layer_c import interpret as interpret_alerts
from app.modules.runs.repository import ClaimedRun, RunRepository
from app.modules.runs.restitution import PROMPT_CODE as RESTITUTION_PROMPT_CODE
from app.modules.runs.restitution import PROMPT_VERSION as RESTITUTION_PROMPT_VERSION
from app.modules.runs.restitution import (
    RESTITUTION_VERSION,
    Restitution,
    RestitutionRules,
    select,
)
from app.modules.runs.restitution import build_batch as build_restitution_batch
from app.modules.runs.restitution import interpret as interpret_restitution
from app.modules.usage.ledger import UsageLedger

logger = logging.getLogger(__name__)

NOT_AVAILABLE = {"reason": "not_available"}
RUN_ENGINE_VERSION = (
    f"{ENGINE_VERSION}+{LAYER_B_VERSION}+{LAYER_C_VERSION}+{RESTITUTION_VERSION}"
)


class RunFailure(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class RunWorker:
    def __init__(
        self,
        repository: RunRepository,
        settings: Settings,
        *,
        worker_id: str,
        embedder: Embedder | None = None,
        grader: SubQuestionGrader | None = None,
        qualifier: AlertQualifier | None = None,
        writer: RestitutionWriter | None = None,
        ledger: UsageLedger | None = None,
    ) -> None:
        self._repository = repository
        self._settings = settings
        self._ledger = ledger or UsageLedger(None, settings)
        self._worker_id = f"{worker_id}#runs"
        self._stopping = asyncio.Event()
        # Sans modèle d'embeddings, la couche B n'est pas calculée ; sans
        # IA, ses critères restent non conclus, les écarts de la couche C ne
        # sont pas qualifiés et la synthèse n'est pas rédigée.
        self._embedder = embedder
        self._grader = grader
        self._qualifier = qualifier
        self._writer = writer

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
        run = await self._repository.claim(self._worker_id, RUN_ENGINE_VERSION)
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

    def layer_b_rules(self) -> LayerBRules:
        return LayerBRules(
            min_similarity=self._settings.run_layer_b_min_similarity,
            top_k=self._settings.run_layer_b_top_k,
            excerpts_per_question=self._settings.run_layer_b_excerpts_per_question,
            excerpt_chars=self._settings.run_layer_b_excerpt_chars,
            max_input_tokens=self._settings.run_layer_b_max_input_tokens,
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

            step = "layer_b"
            payload = results.payload(rules)
            layer_b: LayerBResults | None = None
            if self._embedder is None:
                await self._repository.step(run, step, "skipped", NOT_AVAILABLE)
            else:
                await self._repository.step(run, step, "running")
                layer_b_rules = self.layer_b_rules()
                layer_b = await self._layer_b(run, layer_b_rules)
                await self._repository.step(run, step, "completed", layer_b.summary())
                payload["layer_b"] = layer_b.payload()
                payload["rules"] = {
                    **payload["rules"],
                    "layer_b": {**layer_b_rules.as_dict(), "ai": layer_b.ai},
                }

            step = "layer_c"
            await self._repository.step(run, step, "running")
            layer_c_inputs = await self._repository.layer_c_inputs(run)
            layer_c_rules = LayerCRules()
            layer_c = await self._layer_c(
                run, layer_c_inputs, results, layer_b, layer_c_rules
            )
            await self._repository.step(run, step, "completed", layer_c.summary())
            payload["layer_c"] = layer_c.payload()
            payload["rules"] = {
                **payload["rules"],
                "layer_c": {**layer_c_rules.as_dict(), "ai": layer_c.ai},
            }

            step = "restitution"
            await self._repository.step(run, step, "running")
            restitution_rules = RestitutionRules()
            restitution = await self._restitution(
                run, layer_c_inputs, results, layer_b, layer_c, restitution_rules
            )
            payload["restitution"] = restitution.payload()
            payload["summary"] = restitution.summary()
            payload["rules"] = {
                **payload["rules"],
                "restitution": {**restitution_rules.as_dict(), "ai": restitution.ai},
            }
            written = await self._repository.complete(run, payload)
            logger.info(
                "Run %s completed: %s variables, %s evidences, %s scores, %s alerts, "
                "%s synthesis items, %s actions",
                run.run_id,
                written.get("variables"),
                written.get("evidences"),
                written.get("scores"),
                written.get("alerts"),
                written.get("synthesis_items"),
                written.get("actions"),
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
            logger.warning(
                "Run %s failed at step %s: %s %s",
                run.run_id,
                step,
                code,
                error.context if isinstance(error, ApiError) else "",
            )
            try:
                await self._repository.fail(run, step, code, message)
            except Exception:
                logger.exception("Unable to record the failure of run %s", run.run_id)

    async def _layer_b(self, run: ClaimedRun, rules: LayerBRules) -> LayerBResults:
        assert self._embedder is not None
        inputs = await self._repository.layer_b_inputs(run)
        queries = inputs.queries()
        candidates = []
        if queries:
            async with self._ledger.embeddings(
                workspace_id=run.workspace_id,
                run_id=run.run_id,
                source="layer_b_search",
                model=self._embedder.model,
            ):
                vectors = await self._embedder.embed([text for _, text in queries])
            candidates = await self._repository.sub_question_matches(
                run,
                self._embedder.model,
                [
                    (question_id, vector)
                    for (question_id, _), vector in zip(queries, vectors, strict=True)
                ],
                rules.top_k,
            )
        retrieved = retrieve(candidates, rules)
        batches = build_batches(inputs, retrieved, rules)
        grading, ai = await self._grade(run, batches)
        results = finalize(inputs, retrieved, grading, rules)
        results.ai = ai
        return results

    async def _grade(
        self, run: ClaimedRun, batches: list[GradingBatch]
    ) -> tuple[Grading, dict]:
        """Un appel IA par lot (un seul en pratique). Un lot en échec laisse
        ses critères non conclus ; les autres gardent leurs notes."""
        grading = Grading()
        ai: dict = {
            "prompt": f"{PROMPT_CODE}_{PROMPT_VERSION}",
            "batches": len(batches),
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_eur": "0",
            "errors": [],
        }
        if not batches:
            return grading, {**ai, "status": "not_needed"}
        if self._grader is None:
            return grading, {**ai, "status": "unavailable", "errors": ["AI_DISABLED"]}
        prompt = await self._prompt(run, PROMPT_CODE, PROMPT_VERSION)
        if prompt is None:
            return grading, {
                **ai,
                "status": "unavailable",
                "errors": ["PROMPT_UNAVAILABLE"],
            }

        usages: list[AiUsage] = []
        for index, batch in enumerate(batches):
            if index:
                # Prolonge le bail entre deux appels.
                await self._repository.step(run, "layer_b", "running")
            try:
                raw, usage = await self._grader.grade(prompt, batch)
            except GradingUnavailable as error:
                logger.warning(
                    "Run %s: layer B grading unavailable (%s: %s)",
                    run.run_id,
                    error.code,
                    error.message,
                )
                ai["errors"].append(error.code)
                continue
            usages.append(usage)
            interpret(batch, raw, grading)

        return grading, await self._account(run, "layer_b", ai, usages, len(batches))

    async def _layer_c(
        self,
        run: ClaimedRun,
        inputs: LayerCInputs,
        layer_a: LayerAResults,
        layer_b: LayerBResults | None,
        rules: LayerCRules,
    ) -> LayerCResults:
        alerts = detect(inputs, layer_a, layer_b, rules)
        ai: dict = {
            "prompt": f"{ALERT_PROMPT_CODE}_{ALERT_PROMPT_VERSION}",
            "batches": 0,
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_eur": "0",
            "errors": [],
        }
        if not alerts:
            return LayerCResults([], 0, {**ai, "status": "not_needed"})
        if self._qualifier is None:
            return LayerCResults(
                alerts,
                len(alerts),
                {**ai, "status": "unavailable", "errors": ["AI_DISABLED"]},
            )
        prompt = await self._prompt(run, ALERT_PROMPT_CODE, ALERT_PROMPT_VERSION)
        if prompt is None:
            return LayerCResults(
                alerts,
                len(alerts),
                {**ai, "status": "unavailable", "errors": ["PROMPT_UNAVAILABLE"]},
            )

        # Texte des extraits : ceux de la couche B sont déjà lus, ceux de la
        # couche A sont demandés en une fois.
        texts = known_texts(layer_b)
        missing = list(
            dict.fromkeys(
                segment
                for alert in alerts
                for segment in alert.segment_ids
                if segment not in texts
            )
        )
        texts.update(await self._repository.segment_texts(run, missing))
        batches = build_alert_batches(inputs, alerts, texts, rules)
        ai["batches"] = len(batches)

        infirmed: set = set()
        usages: list[AiUsage] = []
        for index, batch in enumerate(batches):
            if index:
                # Prolonge le bail entre deux appels.
                await self._repository.step(run, "layer_c", "running")
            try:
                raw, usage = await self._qualifier.qualify(prompt, batch)
            except GradingUnavailable as error:
                logger.warning(
                    "Run %s: layer C qualification unavailable (%s: %s)",
                    run.run_id,
                    error.code,
                    error.message,
                )
                ai["errors"].append(error.code)
                continue
            usages.append(usage)
            infirmed |= interpret_alerts(batch, raw, rules)

        return LayerCResults(
            [alert for alert in alerts if alert.id not in infirmed],
            len(alerts),
            await self._account(run, "layer_c", ai, usages, len(batches)),
        )

    async def _restitution(
        self,
        run: ClaimedRun,
        inputs: LayerCInputs,
        layer_a: LayerAResults,
        layer_b: LayerBResults | None,
        layer_c: LayerCResults,
        rules: RestitutionRules,
    ) -> Restitution:
        restitution = select(inputs, layer_a, layer_b, layer_c.alerts, rules)
        ai: dict = {
            "prompt": f"{RESTITUTION_PROMPT_CODE}_{RESTITUTION_PROMPT_VERSION}",
            "batches": 0,
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_eur": "0",
            "errors": [],
        }
        if not restitution.findings:
            restitution.ai = {**ai, "status": "not_needed"}
            return restitution
        if self._writer is None:
            restitution.ai = {**ai, "status": "unavailable", "errors": ["AI_DISABLED"]}
            return restitution
        prompt = await self._prompt(
            run, RESTITUTION_PROMPT_CODE, RESTITUTION_PROMPT_VERSION
        )
        if prompt is None:
            restitution.ai = {
                **ai,
                "status": "unavailable",
                "errors": ["PROMPT_UNAVAILABLE"],
            }
            return restitution

        texts = known_texts(layer_b)
        missing = list(
            dict.fromkeys(
                segment
                for finding in restitution.findings
                for segment in finding.segment_ids
                if segment not in texts
            )
        )
        texts.update(await self._repository.segment_texts(run, missing))
        batch = build_restitution_batch(inputs.locale, restitution, texts, rules)
        assert batch is not None
        usages: list[AiUsage] = []
        try:
            raw, usage = await self._writer.write(prompt, batch)
        except GradingUnavailable as error:
            logger.warning(
                "Run %s: restitution writing unavailable (%s: %s)",
                run.run_id,
                error.code,
                error.message,
            )
            ai["errors"].append(error.code)
        else:
            usages.append(usage)
            interpret_restitution(restitution, batch, raw)
        restitution.ai = await self._account(run, "restitution", ai, usages, 1)
        return restitution

    async def _prompt(self, run: ClaimedRun, code: str, version: str) -> dict | None:
        """Prompt actif, ou `None` s'il est introuvable : l'étape continue
        sans IA."""
        try:
            prompt = await self._repository.prompt(code, version)
        except ApiError as error:
            if error.code == "RUN_LEASE_LOST":
                raise
            logger.warning("Run %s: prompt %s unavailable", run.run_id, code)
            return None
        return prompt if prompt.get("template") else None

    async def _account(
        self,
        run: ClaimedRun,
        source: str,
        ai: dict,
        usages: list[AiUsage],
        batches: int,
    ) -> dict:
        """Résumé de l'étape (rules_applied) et inscription au registre de
        consommation, que le run aboutisse ou non."""
        summary = self._usage(ai, usages, batches)
        await self._ledger.record(
            workspace_id=run.workspace_id,
            run_id=run.run_id,
            source=source,
            kind="completion",
            model=summary.get("model"),
            calls=summary["calls"],
            input_tokens=summary["input_tokens"],
            output_tokens=summary["output_tokens"],
            cost_eur=Decimal(summary["cost_eur"]),
        )
        return summary

    @staticmethod
    def _usage(ai: dict, usages: list[AiUsage], batches: int) -> dict:
        ai["calls"] = len(usages)
        ai["input_tokens"] = sum(usage.input_tokens for usage in usages)
        ai["output_tokens"] = sum(usage.output_tokens for usage in usages)
        ai["cost_eur"] = str(sum((usage.cost_eur for usage in usages), Decimal("0")))
        if usages:
            ai["model"] = usages[0].model
        ai["status"] = (
            "completed"
            if len(usages) == batches
            else "partial"
            if usages
            else "unavailable"
        )
        return ai

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
