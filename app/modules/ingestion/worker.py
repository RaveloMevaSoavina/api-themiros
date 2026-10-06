"""Worker d'ingestion : consomme la file `ingestion_jobs` (spec §12.3).

Lancement : `python -m app.modules.ingestion.worker`
"""

import asyncio
import contextlib
import logging
import signal
import socket
from uuid import uuid4

import httpx

from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.infrastructure.supabase import SupabaseGateway
from app.modules.ingestion.analysis.country import get_country_detector
from app.modules.ingestion.analysis.embeddings import build_embedder
from app.modules.ingestion.errors import is_abandon, is_retryable
from app.modules.ingestion.pipeline import IngestionPipeline, lease_lost
from app.modules.ingestion.repository import ClaimedJob, IngestionRepository

logger = logging.getLogger(__name__)


def describe_error(error: BaseException) -> tuple[str, str]:
    if isinstance(error, ApiError):
        return error.code, error.detail
    return "INGESTION_FAILED", "Le traitement du document a échoué"


class IngestionWorker:
    def __init__(
        self,
        repository: IngestionRepository,
        pipeline: IngestionPipeline,
        settings: Settings,
        *,
        worker_id: str | None = None,
    ) -> None:
        self._repository = repository
        self._pipeline = pipeline
        self._settings = settings
        self._worker_id = worker_id or f"{socket.gethostname()}-{uuid4().hex[:8]}"
        self._stopping = asyncio.Event()

    def stop(self) -> None:
        self._stopping.set()

    async def run(self) -> None:
        logger.info(
            "Ingestion worker %s started (%s slots)",
            self._worker_id,
            self._settings.ingestion_worker_concurrency,
        )
        await asyncio.gather(
            *(
                self._slot(index)
                for index in range(self._settings.ingestion_worker_concurrency)
            )
        )
        logger.info("Ingestion worker %s stopped", self._worker_id)

    async def _slot(self, index: int) -> None:
        slot_id = f"{self._worker_id}#{index}"
        while not self._stopping.is_set():
            try:
                job = await self._repository.claim(slot_id)
            except Exception:
                logger.exception("Unable to claim an ingestion job")
                job = None
            if job is not None:
                await self.process(job)
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._stopping.wait(),
                    timeout=self._settings.ingestion_worker_poll_seconds,
                )

    async def process(self, job: ClaimedJob) -> None:
        keep_alive = asyncio.create_task(self._keep_alive(job))

        async def on_stage(stage: str) -> None:
            if not await self._repository.heartbeat(job, stage):
                raise lease_lost()

        try:
            result = await self._pipeline.run(job, on_stage)
            await self._repository.complete(job, result)
            logger.info("Ingestion job %s (%s) completed", job.id, job.kind)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self._handle_failure(job, error)
        finally:
            keep_alive.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await keep_alive

    async def _handle_failure(self, job: ClaimedJob, error: Exception) -> None:
        if is_abandon(error):
            # Un autre worker a repris la tâche, ou un humain a tranché.
            logger.warning(
                "Ingestion job %s abandoned: %s", job.id, describe_error(error)[0]
            )
            return
        code, message = describe_error(error)
        retry = is_retryable(error)
        if isinstance(error, ApiError):
            logger.warning("Ingestion job %s failed: %s", job.id, code)
        else:
            logger.exception("Ingestion job %s failed unexpectedly", job.id)
        try:
            status = await self._repository.fail(job, code, message, retry=retry)
            logger.info("Ingestion job %s is now %s", job.id, status)
        except ApiError as fail_error:
            if not is_abandon(fail_error):
                logger.exception("Unable to record the failure of job %s", job.id)

    async def _keep_alive(self, job: ClaimedJob) -> None:
        """Prolonge le bail pendant les étapes longues (OCR)."""
        interval = max(10, self._settings.ingestion_worker_lease_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            try:
                if not await self._repository.heartbeat(job):
                    return
            except Exception:
                logger.warning("Heartbeat failed for ingestion job %s", job.id)


def build_pipeline(
    settings: Settings, client: httpx.AsyncClient
) -> tuple[IngestionRepository, IngestionPipeline]:
    repository = IngestionRepository(SupabaseGateway(settings, client), settings)
    pipeline = IngestionPipeline(
        repository,
        build_embedder(settings),
        get_country_detector(settings.ingestion_ner_model),
        settings,
    )
    return repository, pipeline


async def _main() -> None:
    settings = get_settings()
    async with httpx.AsyncClient() as client:
        repository, pipeline = build_pipeline(settings, client)
        worker = IngestionWorker(repository, pipeline, settings)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, worker.stop)
        await worker.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    asyncio.run(_main())
