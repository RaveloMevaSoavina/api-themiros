from uuid import UUID, uuid4

from app.core.config import Settings
from app.core.errors import ApiError
from app.modules.classification.repository import ClaimedFramework, Probe
from app.modules.classification.worker import ClassificationWorker
from app.modules.ingestion.errors import embedding_unavailable

PILLAR_ID = UUID("22222222-2222-4222-8222-222222222222")
VARIABLE_ID = UUID("33333333-3333-4333-8333-333333333333")


def _job() -> ClaimedFramework:
    return ClaimedFramework.from_row(
        {
            "framework_id": str(uuid4()),
            "workspace_id": str(uuid4()),
            "lease_token": str(uuid4()),
            "probes": [
                {
                    "pillar_id": str(PILLAR_ID),
                    "pillar_variable_id": None,
                    "text": "Budget. Exécution budgétaire",
                    "source_hash": "h1",
                },
                {
                    "pillar_id": str(PILLAR_ID),
                    "pillar_variable_id": str(VARIABLE_ID),
                    "text": "Budget : Taux de décaissement",
                    "source_hash": "h2",
                },
            ],
        }
    )


class FakeEmbedder:
    model = "fake-embedding"

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.texts: list[str] = []

    async def embed(self, texts):
        if self.error:
            raise self.error
        self.texts.extend(texts)
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class FakeRepository:
    def __init__(self, jobs, *, complete_error: Exception | None = None) -> None:
        self.jobs = list(jobs)
        self.complete_error = complete_error
        self.claims: list[tuple[str, str]] = []
        self.completed: list[tuple[ClaimedFramework, str, list]] = []
        self.failures: list[tuple[str, str]] = []

    async def claim(self, worker_id, model):
        self.claims.append((worker_id, model))
        return self.jobs.pop(0) if self.jobs else None

    async def complete(self, job, model, vectors):
        if self.complete_error:
            raise self.complete_error
        self.completed.append((job, model, vectors))
        return {"probes": len(vectors), "documents": 3}

    async def fail(self, job, code, message):
        self.failures.append((code, message))


def _worker(repository, embedder) -> ClassificationWorker:
    return ClassificationWorker(repository, embedder, Settings(), worker_id="host-1")


def test_claimed_framework_parses_pillar_and_variable_probes() -> None:
    job = _job()
    assert job is not None
    assert job.probes[0] == Probe(PILLAR_ID, None, "Budget. Exécution budgétaire", "h1")
    assert job.probes[1].pillar_variable_id == VARIABLE_ID
    assert ClaimedFramework.from_row(None) is None


async def test_empty_queue_does_nothing() -> None:
    repository = FakeRepository([])
    assert await _worker(repository, FakeEmbedder()).run_once() is False
    assert repository.claims == [("host-1#classification", "fake-embedding")]


async def test_probes_are_embedded_with_the_segment_model() -> None:
    job = _job()
    repository = FakeRepository([job])
    embedder = FakeEmbedder()

    assert await _worker(repository, embedder).run_once() is True

    assert embedder.texts == [probe.text for probe in job.probes]
    completed_job, model, vectors = repository.completed[0]
    assert completed_job is job
    assert model == "fake-embedding"
    assert [probe.source_hash for probe, _ in vectors] == ["h1", "h2"]
    assert vectors[1][1] == [1.0, 1.0]
    assert repository.failures == []


async def test_embedding_failure_is_recorded_for_retry() -> None:
    repository = FakeRepository([_job()])

    await _worker(repository, FakeEmbedder(embedding_unavailable())).run_once()

    assert repository.completed == []
    assert repository.failures[0][0] == "EMBEDDING_SERVICE_UNAVAILABLE"


async def test_lost_lease_is_abandoned_without_failure() -> None:
    lost = ApiError(status_code=409, code="CLASSIFICATION_LEASE_LOST", detail="lost")
    repository = FakeRepository([_job()], complete_error=lost)

    await _worker(repository, FakeEmbedder()).run_once()

    assert repository.failures == []
