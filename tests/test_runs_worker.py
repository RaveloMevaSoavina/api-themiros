from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from app.core.config import Settings
from app.core.errors import ApiError
from app.modules.runs.domain import (
    Match,
    Pillar,
    ProgramVersion,
    RunInputs,
    Variable,
)
from app.modules.runs.repository import ClaimedRun
from app.modules.runs.worker import RunWorker

VERSION = UUID("55555555-5555-4555-8555-555555555555")
VARIABLE = UUID("33333333-3333-4333-8333-333333333333")
DOCUMENTS = [uuid4(), uuid4()]


def _run(min_sources: int = 2) -> ClaimedRun:
    return ClaimedRun.from_row(
        {
            "run_id": str(uuid4()),
            "workspace_id": str(uuid4()),
            "lease_token": str(uuid4()),
            "attempts": 1,
            "rules_applied": {
                "critical_mass_settings": {"min_independent_sources": min_sources}
            },
        }
    )


def _inputs(state: str = "ready", age: timedelta = timedelta()) -> RunInputs:
    return RunInputs(
        state=state,
        created_at=datetime.now(UTC) - age,
        versions=(ProgramVersion(VERSION, 1),),
        document_versions={document: frozenset({VERSION}) for document in DOCUMENTS},
        unassigned_documents=0,
        pillars=(Pillar(uuid4(), (Variable(VARIABLE, "has_budget_execution", 1),)),),
        segments={"published": 5, "classified": 3, "low_confidence": 1},
        matches=tuple(
            Match(uuid4(), document, VARIABLE, 0.6) for document in DOCUMENTS
        ),
    )


class FakeRepository:
    def __init__(
        self,
        runs,
        inputs: RunInputs,
        *,
        complete_error: Exception | None = None,
    ) -> None:
        self.runs = list(runs)
        self.inputs_value = inputs
        self.complete_error = complete_error
        self.claims: list[tuple[str, str]] = []
        self.steps: list[tuple[str, str, dict | None]] = []
        self.completed: list[dict] = []
        self.deferred: list[tuple[int, str]] = []
        self.failures: list[tuple[str | None, str]] = []

    async def claim(self, worker_id, engine_version):
        self.claims.append((worker_id, engine_version))
        return self.runs.pop(0) if self.runs else None

    async def inputs(self, run):
        return self.inputs_value

    async def step(self, run, step_name, status, summary=None):
        self.steps.append((step_name, status, summary))

    async def complete(self, run, results):
        if self.complete_error:
            raise self.complete_error
        self.completed.append(results)
        return {"variables": 1, "evidences": 2, "scores": 1}

    async def defer(self, run, delay_seconds, reason):
        self.deferred.append((delay_seconds, reason))

    async def fail(self, run, step_name, code, message):
        self.failures.append((step_name, code))


def _worker(repository) -> RunWorker:
    return RunWorker(repository, Settings(), worker_id="host-1")


async def test_empty_queue_does_nothing() -> None:
    repository = FakeRepository([], _inputs())
    assert await _worker(repository).run_once() is False
    assert repository.claims == [("host-1#runs", "layer-a/1.0.0")]


async def test_a_run_goes_through_the_named_steps_and_publishes_layer_a() -> None:
    repository = FakeRepository([_run()], _inputs())

    assert await _worker(repository).run_once() is True

    assert [(name, status) for name, status, _ in repository.steps] == [
        ("classification", "running"),
        ("classification", "completed"),
        ("variables", "running"),
        ("variables", "completed"),
        ("layer_a", "completed"),
        ("layer_b", "skipped"),
        ("layer_c", "skipped"),
        ("restitution", "running"),
    ]
    assert repository.steps[1][2] == {
        "published": 5,
        "classified": 3,
        "low_confidence": 1,
        "unclassified": 2,
        "unassigned_documents": 0,
    }
    (results,) = repository.completed
    assert results["variables"][0]["state"] == "present"
    assert results["scores"][0]["score"] == 100.0
    assert results["rules"]["min_independent_sources"] == 2
    assert repository.failures == []


async def test_the_independent_sources_rule_comes_from_the_run_settings() -> None:
    repository = FakeRepository([_run(min_sources=3)], _inputs())

    await _worker(repository).run_once()

    (results,) = repository.completed
    assert results["variables"][0]["state"] == "partiel"
    assert results["rules"]["min_independent_sources"] == 3


async def test_a_run_waits_for_a_pending_classification() -> None:
    repository = FakeRepository([_run()], _inputs("pending"))

    await _worker(repository).run_once()

    assert repository.deferred == [(30, "classification_pending")]
    assert repository.completed == []
    assert repository.failures == []


async def test_a_run_fails_when_the_classification_never_ends() -> None:
    repository = FakeRepository([_run()], _inputs("pending", timedelta(hours=2)))

    await _worker(repository).run_once()

    assert repository.deferred == []
    assert repository.failures == [("classification", "CLASSIFICATION_TIMEOUT")]


async def test_a_revalidated_framework_fails_the_run() -> None:
    repository = FakeRepository([_run()], _inputs("framework_changed"))

    await _worker(repository).run_once()

    assert repository.failures == [("classification", "FRAMEWORK_CHANGED")]


async def test_a_write_failure_fails_the_run_at_restitution() -> None:
    invalid = ApiError(status_code=409, code="RUN_RESULTS_INVALID", detail="invalid")
    repository = FakeRepository([_run()], _inputs(), complete_error=invalid)

    await _worker(repository).run_once()

    assert repository.failures == [("restitution", "RUN_RESULTS_INVALID")]


async def test_an_unexpected_error_is_recorded_without_its_details() -> None:
    inputs = replace(_inputs(), versions=None)  # provoque une erreur de calcul
    repository = FakeRepository([_run()], inputs)

    await _worker(repository).run_once()

    assert repository.failures == [("variables", "RUN_FAILED")]


async def test_lost_lease_is_abandoned_without_failure() -> None:
    lost = ApiError(status_code=409, code="RUN_LEASE_LOST", detail="lost")
    repository = FakeRepository([_run()], _inputs(), complete_error=lost)

    await _worker(repository).run_once()

    assert repository.failures == []
