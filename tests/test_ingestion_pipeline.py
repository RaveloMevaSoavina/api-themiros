import hashlib
from dataclasses import replace
from typing import Any
from uuid import UUID, uuid4

import pymupdf

from app.core.config import Settings
from app.core.errors import ApiError
from app.modules.ingestion.analysis.country import CountryDetector
from app.modules.ingestion.analysis.segmenter import PageText
from app.modules.ingestion.domain import IngestionRules, WorkspaceProfile
from app.modules.ingestion.errors import embedding_unavailable
from app.modules.ingestion.pipeline import IngestionPipeline
from app.modules.ingestion.repository import ClaimedJob, DocumentRecord, StoredPages
from app.modules.ingestion.worker import IngestionWorker

WORKSPACE_ID = UUID("11111111-1111-4111-8111-111111111111")
RELEVANT = (
    "Programme d'adaptation agricole en Ethiopie finance par le FIDA. "
    "Les activites portent sur l'irrigation et la gestion de l'eau en Ethiopie. "
) * 12
OFF_TOPIC = (
    "TECHNICAL OFFER. This technical offer describes the construction of a "
    "road network in Kenya for the ministry of transport. "
) * 12


def _pdf(text: str, pages: int = 1) -> bytes:
    document = pymupdf.open()
    for _ in range(pages):
        page = document.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=10)
    return document.tobytes()


class FakeEmbedder:
    """Aligné avec l'empreinte, sauf pour une « technical offer »."""

    model = "fake-embedding"

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts):
        self.calls += 1
        return [
            [0.0, 1.0] if "technical offer" in text.lower() else [1.0, 0.0]
            for text in texts
        ]


class FakeRepository:
    def __init__(self, content: bytes, *, status: str = "a_verifier") -> None:
        self.content = content
        self.document_id = uuid4()
        self.status = status
        self.stages: list[str] = []
        self.pages: list[dict[str, Any]] = []
        self.qualification: dict[str, Any] | None = None
        self.segments: list[dict[str, Any]] = []
        self.published = 0
        self.fingerprint: tuple[str, list[float]] | None = None
        self.fingerprint_saves = 0
        self.overrides: dict[str, str] = {}
        self.completed: dict[str, Any] | None = None
        self.failures: list[tuple[str, bool]] = []
        self.lose_human_lock = False

    def job(self, kind: str = "full") -> ClaimedJob:
        return ClaimedJob(
            id=uuid4(),
            document_id=self.document_id,
            workspace_id=WORKSPACE_ID,
            kind=kind,
            lease_token=uuid4(),
            attempts=1,
            max_attempts=4,
        )

    async def load_document(self, document_id):
        return DocumentRecord(
            id=document_id,
            workspace_id=WORKSPACE_ID,
            original_filename="prodoc.pdf",
            storage_path=f"{WORKSPACE_ID}/{document_id}/prodoc.pdf",
            file_hash=hashlib.sha256(self.content).hexdigest(),
            mime_type="application/pdf",
            status=self.status,
            status_reason_code=None,
            detected_language="fr",
        )

    async def load_profile(self, workspace_id):
        return (
            WorkspaceProfile(
                id=str(workspace_id),
                kind="programme",
                target_country="ET",
                financiers=("FIDA",),
                themes=("agriculture", "water"),
                expected_languages=("fr", "en"),
                stage="implementation",
                start_year=2022,
                end_year=2028,
            ),
            ["FIDA", "IFAD"],
        )

    async def load_rules(self, workspace_id):
        return IngestionRules(off_topic_keywords=("technical offer",))

    async def load_overrides(self, document_id):
        return self.overrides

    async def load_pages(self, document_id):
        return StoredPages(
            pages=[
                PageText(page["page_number"], page["text"], page["char_start"])
                for page in self.pages
            ],
            total_pages=len(self.pages),
            exploitable_pages=sum(page["is_exploitable"] for page in self.pages),
        )

    async def download(self, storage_path):
        return self.content

    async def load_fingerprint(self, workspace_id):
        return self.fingerprint

    async def save_fingerprint(
        self, profile, *, source_text, source_hash, vector, model
    ):
        self.fingerprint = (source_hash, vector)
        self.fingerprint_saves += 1

    async def commit_extraction(self, job, *, pages, pdf_type, metadata):
        self.pages = pages

    async def commit_qualification(self, job, result):
        if self.lose_human_lock:
            raise ApiError(status_code=409, code="HUMAN_DECISION_LOCKED", detail="x")
        self.qualification = result
        self.status = result["status"]
        return {"status": result["status"]}

    async def append_segments(self, job, generation, rows):
        self.segments.extend(rows)
        return len(rows)

    async def publish_index(self, job, generation, model):
        self.published = len(self.segments)
        return self.published

    async def heartbeat(self, job, stage=None):
        if stage:
            self.stages.append(stage)
        return True

    async def complete(self, job, result):
        self.completed = result

    async def fail(self, job, code, message, *, retry):
        self.failures.append((code, retry))
        return "queued" if retry else "failed"


def _pipeline(
    repository: FakeRepository,
    embedder: FakeEmbedder | None = None,
    *,
    sample_segments: int | None = None,
):
    settings = Settings(_env_file=None)
    if sample_segments is not None:
        settings = settings.model_copy(
            update={"ingestion_relevance_sample_segments": sample_segments}
        )
    return IngestionPipeline(
        repository,
        embedder or FakeEmbedder(),
        CountryDetector(ner_model=None),
        settings,
    )


def _worker(repository: FakeRepository, pipeline: IngestionPipeline) -> IngestionWorker:
    return IngestionWorker(
        repository, pipeline, Settings(_env_file=None), worker_id="t"
    )


async def test_full_ingestion_of_a_relevant_document_indexes_it() -> None:
    repository = FakeRepository(_pdf(RELEVANT))
    await _worker(repository, _pipeline(repository)).process(repository.job())

    assert repository.failures == []
    assert repository.stages == ["extraction", "detection", "pertinence", "indexation"]
    qualification = repository.qualification
    assert qualification is not None
    assert qualification["status"] == "conforme"
    assert qualification["language"] == "fr"
    assert qualification["country"] == "ET"
    assert qualification["relevance_details"]["adjustments"] == {
        "country": 10,
        "financier": 5,
        "theme": 5,
        "language": 5,
    }
    assert repository.published > 0
    assert repository.segments[0]["embedding"] == [1.0, 0.0]
    assert repository.completed is not None
    assert repository.completed["indexation"]["segments_count"] == repository.published


async def test_off_topic_document_is_rejected_and_never_indexed() -> None:
    repository = FakeRepository(_pdf(OFF_TOPIC))
    await _worker(repository, _pipeline(repository)).process(repository.job())

    qualification = repository.qualification
    assert qualification is not None
    assert qualification["status"] == "rejete"
    assert qualification["relevance_score"] < 40
    assert qualification["status_reason_code"] == "country_mismatch"
    assert qualification["relevance_details"]["signals"]["off_topic_keywords"] == [
        "technical offer"
    ]
    assert repository.segments == []
    assert "indexation" not in repository.stages


class RecordingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[str]] = []

    async def embed(self, texts):
        self.batches.append(list(texts))
        return await super().embed(texts)


async def test_relevance_uses_a_sample_and_indexing_embeds_the_rest() -> None:
    repository = FakeRepository(_pdf(RELEVANT, pages=6))
    embedder = RecordingEmbedder()
    pipeline = _pipeline(repository, embedder, sample_segments=2)
    await _worker(repository, pipeline).process(repository.job())

    qualification = repository.qualification
    assert qualification is not None
    details = qualification["relevance_details"]
    total = details["segments_count"]
    assert total > 2
    assert details["segments_sampled"] == 2
    # Empreinte, échantillon, puis uniquement les segments restants.
    assert [len(batch) for batch in embedder.batches] == [1, 2, total - 2]
    assert repository.published == total
    # Échantillon = premier et dernier segment ; l'index garde l'ordre du
    # document.
    first, last = embedder.batches[1]
    assert [row["text"] for row in repository.segments] == [
        first,
        *embedder.batches[2],
        last,
    ]
    assert all(row["embedding"] == [1.0, 0.0] for row in repository.segments)


async def test_rejected_document_only_embeds_the_sample() -> None:
    repository = FakeRepository(_pdf(OFF_TOPIC, pages=6))
    embedder = RecordingEmbedder()
    pipeline = _pipeline(repository, embedder, sample_segments=2)
    await _worker(repository, pipeline).process(repository.job())

    assert repository.qualification is not None
    assert repository.qualification["status"] == "rejete"
    assert [len(batch) for batch in embedder.batches] == [1, 2]
    assert repository.segments == []


async def test_fingerprint_embedding_is_cached_between_documents() -> None:
    repository = FakeRepository(_pdf(RELEVANT))
    embedder = FakeEmbedder()
    pipeline = _pipeline(repository, embedder)
    await _worker(repository, pipeline).process(repository.job())
    calls_after_first = embedder.calls
    await _worker(repository, pipeline).process(repository.job("requalify"))

    assert repository.fingerprint_saves == 1
    # Requalification : un seul appel (segments), l'empreinte vient du cache.
    assert embedder.calls == calls_after_first + 1


async def test_human_correction_of_country_takes_precedence() -> None:
    repository = FakeRepository(_pdf(RELEVANT))
    await _worker(repository, _pipeline(repository)).process(repository.job())
    repository.overrides = {"country": "KE"}
    await _worker(repository, _pipeline(repository)).process(
        repository.job("requalify")
    )

    qualification = repository.qualification
    assert qualification is not None
    assert qualification["country"] == "KE"
    assert qualification["country_source"] == "user"
    assert qualification["status"] == "a_verifier"
    assert qualification["status_reason_code"] == "country_mismatch"


async def test_index_job_after_human_integration_reuses_stored_text() -> None:
    repository = FakeRepository(_pdf(OFF_TOPIC))
    pipeline = _pipeline(repository)
    await _worker(repository, pipeline).process(repository.job())
    assert repository.segments == []

    repository.status = "integre_decision_humaine"
    await _worker(repository, pipeline).process(repository.job("index"))
    assert repository.published > 0
    assert repository.failures == []


async def test_file_that_does_not_match_its_hash_fails_permanently() -> None:
    repository = FakeRepository(_pdf(RELEVANT))
    pipeline = _pipeline(repository)
    original = repository.load_document

    async def tampered(document_id):
        record = await original(document_id)
        return replace(record, file_hash="0" * 64)

    repository.load_document = tampered
    await _worker(repository, pipeline).process(repository.job())
    assert repository.failures == [("FILE_INTEGRITY_ERROR", False)]


async def test_corrupted_file_is_not_retried() -> None:
    repository = FakeRepository(b"%PDF-1.7 broken")
    await _worker(repository, _pipeline(repository)).process(repository.job())
    assert repository.failures == [("CORRUPTED_FILE", False)]


async def test_embedding_outage_is_retried() -> None:
    class DownEmbedder(FakeEmbedder):
        async def embed(self, texts):
            raise embedding_unavailable()

    repository = FakeRepository(_pdf(RELEVANT))
    pipeline = _pipeline(repository, DownEmbedder())
    await _worker(repository, pipeline).process(repository.job())
    assert repository.failures == [("EMBEDDING_SERVICE_UNAVAILABLE", True)]


async def test_human_decision_taken_meanwhile_is_never_overwritten() -> None:
    repository = FakeRepository(_pdf(RELEVANT))
    repository.lose_human_lock = True
    await _worker(repository, _pipeline(repository)).process(repository.job())
    assert repository.failures == []
    assert repository.completed is None


async def test_unexpected_errors_fail_without_leaking_details() -> None:
    repository = FakeRepository(_pdf(RELEVANT))

    async def broken(document_id):
        raise ValueError("boom")

    repository.load_document = broken
    await _worker(repository, _pipeline(repository)).process(repository.job())
    assert repository.failures == [("INGESTION_FAILED", False)]
