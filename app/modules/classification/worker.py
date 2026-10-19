"""Vectorisation des sondes des cadres validés (RG-5.5).

Tourne dans le processus du worker d'ingestion, avec le même modèle
d'embeddings que les segments : les similarités ne se comparent qu'entre
vecteurs d'un même modèle. La classification elle-même est faite en SQL.
"""

import asyncio
import contextlib
import logging

from app.core.config import Settings
from app.core.errors import ApiError
from app.modules.classification.repository import (
    ClaimedFramework,
    ClassificationRepository,
)
from app.modules.ingestion.analysis.embeddings import Embedder

logger = logging.getLogger(__name__)


class ClassificationWorker:
    def __init__(
        self,
        repository: ClassificationRepository,
        embedder: Embedder,
        settings: Settings,
        *,
        worker_id: str,
    ) -> None:
        self._repository = repository
        self._embedder = embedder
        self._settings = settings
        self._worker_id = f"{worker_id}#classification"
        self._stopping = asyncio.Event()

    def stop(self) -> None:
        self._stopping.set()

    async def run(self) -> None:
        while not self._stopping.is_set():
            try:
                worked = await self.run_once()
            except Exception:
                logger.exception("Unable to claim a framework classification")
                worked = False
            if worked:
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._stopping.wait(),
                    timeout=self._settings.classification_poll_seconds,
                )

    async def run_once(self) -> bool:
        """Traite un cadre s'il y en a un ; `False` si la file est vide."""
        job = await self._repository.claim(self._worker_id, self._embedder.model)
        if job is None:
            return False
        await self.process(job)
        return True

    async def process(self, job: ClaimedFramework) -> None:
        try:
            vectors = await self._embedder.embed([probe.text for probe in job.probes])
            result = await self._repository.complete(
                job, self._embedder.model, list(zip(job.probes, vectors, strict=True))
            )
            logger.info(
                "Framework %s classified: %s probes, %s documents",
                job.framework_id,
                result.get("probes"),
                result.get("documents"),
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if (
                isinstance(error, ApiError)
                and error.code == "CLASSIFICATION_LEASE_LOST"
            ):
                logger.warning(
                    "Framework %s taken over by another worker", job.framework_id
                )
                return
            code, message = (
                (error.code, error.detail)
                if isinstance(error, ApiError)
                else ("CLASSIFICATION_FAILED", "La classification du corpus a échoué")
            )
            logger.warning(
                "Framework %s classification failed: %s", job.framework_id, code
            )
            try:
                await self._repository.fail(job, code, message)
            except Exception:
                logger.exception(
                    "Unable to record the failure of framework %s", job.framework_id
                )
