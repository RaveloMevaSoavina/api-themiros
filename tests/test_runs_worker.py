from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
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
from app.modules.runs.grader import AiUsage, GradingUnavailable
from app.modules.runs.layer_b import (
    Candidate,
    Criterion,
    LayerBInputs,
    RawGrade,
    RawGrading,
    SubQuestion,
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
        self.layer_b_value = LayerBInputs(locale="fr", cycle="mi_parcours", criteria=())
        self.matches: list[Candidate] = []
        self.queries: list = []

    async def layer_b_inputs(self, run):
        return self.layer_b_value

    async def sub_question_matches(self, run, model, queries, top_k):
        self.queries.append((model, [question_id for question_id, _ in queries], top_k))
        return self.matches

    async def prompt(self, code, version):
        return {"code": code, "version": version, "template": "Noter", "parameters": {}}

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
    assert repository.claims == [("host-1#runs", "layer-a/1.0.0+layer-b/1.0.0")]


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


QUESTIONS = tuple(SubQuestion(uuid4(), f"Question {index} ?") for index in range(2))
CRITERION = Criterion(
    uuid4(), "pertinence", "Pertinence", None, "obligatoire", QUESTIONS
)


class FakeEmbedder:
    model = "test-model"

    async def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeGrader:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    async def grade(self, prompt, batch):
        self.calls += 1
        if self.error:
            raise self.error
        return (
            RawGrading(
                grades=tuple(
                    RawGrade(key, 3, tuple(sorted(batch.allowed[question_id])), False)
                    for key, question_id in batch.question_keys.items()
                ),
                justifications={"C1": "Alignement net avec les priorités."},
            ),
            AiUsage("test-llm", 1200, 80, Decimal("0.0021")),
        )


def _layer_b_repository() -> FakeRepository:
    repository = FakeRepository([_run()], _inputs())
    repository.layer_b_value = LayerBInputs("fr", "mi_parcours", (CRITERION,))
    repository.matches = [
        Candidate(question.id, uuid4(), DOCUMENTS[0], 0.62, "Extrait du corpus.")
        for question in QUESTIONS
    ]
    return repository


def _layer_b_worker(repository, grader) -> RunWorker:
    return RunWorker(
        repository,
        Settings(),
        worker_id="host-1",
        embedder=FakeEmbedder(),
        grader=grader,
    )


async def test_layer_b_rates_the_criteria_in_a_single_ai_call() -> None:
    repository = _layer_b_repository()
    grader = FakeGrader()

    await _layer_b_worker(repository, grader).run_once()

    assert grader.calls == 1
    steps = [(name, status) for name, status, _ in repository.steps]
    assert ("layer_b", "running") in steps and ("layer_b", "completed") in steps
    assert ("layer_c", "skipped") in steps
    assert repository.queries == [("test-model", [q.id for q in QUESTIONS], 8)]
    (results,) = repository.completed
    (note,) = results["layer_b"]["notes"]
    assert note["status"] == "conclu" and note["note"] == 3.0
    assert note["justification"] == "Alignement net avec les priorités."
    assert len(results["layer_b"]["evidences"]) == 2
    ai = results["rules"]["layer_b"]["ai"]
    assert ai["status"] == "completed" and ai["calls"] == 1
    assert ai["prompt"] == "grade_sub_questions_v1.0"
    assert (ai["input_tokens"], ai["output_tokens"]) == (1200, 80)
    summary = next(
        s for n, st, s in repository.steps if (n, st) == ("layer_b", "completed")
    )
    assert summary["concluded"] == 1 and summary["ai_calls"] == 1
    assert repository.failures == []


async def test_without_ai_the_criteria_are_not_concluded() -> None:
    repository = _layer_b_repository()

    await _layer_b_worker(repository, None).run_once()

    (results,) = repository.completed
    (note,) = results["layer_b"]["notes"]
    assert note["status"] == "non_conclu" and note["note"] is None
    ai = results["rules"]["layer_b"]["ai"]
    assert ai["status"] == "unavailable" and ai["errors"] == ["AI_DISABLED"]
    assert repository.failures == []


async def test_an_ai_failure_does_not_fail_the_run() -> None:
    repository = _layer_b_repository()
    grader = FakeGrader(GradingUnavailable("AI_PROVIDER_UNAVAILABLE", "down"))

    await _layer_b_worker(repository, grader).run_once()

    (results,) = repository.completed
    assert results["layer_b"]["notes"][0]["status"] == "non_conclu"
    assert results["rules"]["layer_b"]["ai"]["errors"] == ["AI_PROVIDER_UNAVAILABLE"]
    assert repository.failures == []


async def test_no_ai_call_when_no_criterion_is_documented_enough() -> None:
    repository = _layer_b_repository()
    repository.matches = []
    grader = FakeGrader()

    await _layer_b_worker(repository, grader).run_once()

    assert grader.calls == 0
    (results,) = repository.completed
    assert results["rules"]["layer_b"]["ai"]["status"] == "not_needed"
    assert results["layer_b"]["notes"][0]["documented_sub_questions"] == 0
