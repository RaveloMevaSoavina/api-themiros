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
from app.modules.runs.layer_c import LayerCInputs, PillarCriterion, RawQualification
from app.modules.runs.repository import ClaimedRun
from app.modules.runs.restitution import (
    RawAction,
    RawItem,
    RawProjection,
    RawRestitution,
)
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
        self.layer_c_value = LayerCInputs(locale="fr", pillar_names={}, links=())
        self.text_requests: list[list[UUID]] = []
        self.texts: dict[UUID, str] = {}

    async def layer_c_inputs(self, run):
        return self.layer_c_value

    async def segment_texts(self, run, segment_ids):
        self.text_requests.append(list(segment_ids))
        return {
            segment: self.texts.get(segment, "Extrait A.") for segment in segment_ids
        }

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
    assert repository.claims == [
        ("host-1#runs", "layer-a/1.0.0+layer-b/1.0.0+layer-c/1.0.0+restitution/1.1.0")
    ]


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
        ("layer_c", "running"),
        ("layer_c", "completed"),
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
    def __init__(self, error: Exception | None = None, note: int = 3) -> None:
        self.error = error
        self.note = note
        self.calls = 0

    async def grade(self, prompt, batch):
        self.calls += 1
        if self.error:
            raise self.error
        return (
            RawGrading(
                grades=tuple(
                    RawGrade(
                        key,
                        self.note,
                        tuple(sorted(batch.allowed[question_id])),
                        False,
                    )
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
    assert ("layer_c", "completed") in steps
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


class FakeQualifier:
    def __init__(self, error: Exception | None = None, confirmed: bool = True) -> None:
        self.error = error
        self.confirmed = confirmed
        self.batches: list = []

    async def qualify(self, prompt, batch):
        self.batches.append(batch)
        if self.error:
            raise self.error
        return (
            [
                RawQualification(
                    key,
                    self.confirmed,
                    "majeure",
                    "Budget bien documenté, pertinence faiblement étayée.",
                    tuple(batch.allowed[alert_id])[:1],
                )
                for key, alert_id in batch.alert_keys.items()
            ],
            AiUsage("test-llm", 900, 60, Decimal("0.0012")),
        )


def _layer_c_repository() -> FakeRepository:
    """Score A de 100 pour le pilier, note B de 1 pour son critère : un
    écart ALERT-01."""
    repository = _layer_b_repository()
    pillar = repository.inputs_value.pillars[0].id
    repository.layer_c_value = LayerCInputs(
        "fr",
        {pillar: "Budget et absorption"},
        (PillarCriterion(pillar, CRITERION.id, CRITERION.code, CRITERION.name),),
    )
    return repository


def _layer_c_worker(repository, qualifier) -> RunWorker:
    return RunWorker(
        repository,
        Settings(),
        worker_id="host-1",
        embedder=FakeEmbedder(),
        grader=FakeGrader(note=1),
        qualifier=qualifier,
    )


async def test_layer_c_qualifies_every_gap_in_a_single_ai_call() -> None:
    repository = _layer_c_repository()
    qualifier = FakeQualifier()

    await _layer_c_worker(repository, qualifier).run_once()

    assert len(qualifier.batches) == 1
    # Les extraits de la couche B sont déjà lus ; seuls ceux de la couche A
    # sont demandés.
    (requested,) = repository.text_requests
    b_segments = {candidate.segment_id for candidate in repository.matches}
    assert requested and not set(requested) & b_segments
    (results,) = repository.completed
    (alert,) = results["layer_c"]["alerts"]
    assert alert["rule_code"] == "ALERT-01" and alert["severity"] == "majeure"
    assert alert["qualified"] is True
    assert alert["message"] == "Budget bien documenté, pertinence faiblement étayée."
    assert len(alert["segment_ids"]) == 1
    ai = results["rules"]["layer_c"]["ai"]
    assert ai["status"] == "completed" and ai["calls"] == 1
    assert ai["prompt"] == "detect_incoherence_v1.0"
    assert results["rules"]["layer_c"]["engine_version"] == "layer-c/1.0.0"
    summary = next(
        s for n, st, s in repository.steps if (n, st) == ("layer_c", "completed")
    )
    assert summary["alerts"] == 1 and summary["qualified"] == 1
    assert repository.failures == []


async def test_without_ai_the_gaps_are_published_unqualified() -> None:
    repository = _layer_c_repository()

    await _layer_c_worker(repository, None).run_once()

    (results,) = repository.completed
    (alert,) = results["layer_c"]["alerts"]
    assert alert["qualified"] is False and alert["severity"] == "majeure"
    assert alert["message"].startswith("Budget et absorption : amélioration")
    # Deux extraits du critère (couche B), puis deux du pilier (couche A).
    assert len(alert["segment_ids"]) == 4
    ai = results["rules"]["layer_c"]["ai"]
    assert ai["status"] == "unavailable" and ai["errors"] == ["AI_DISABLED"]
    assert repository.text_requests == []


async def test_a_gap_infirmed_by_the_ai_is_not_published() -> None:
    repository = _layer_c_repository()

    await _layer_c_worker(repository, FakeQualifier(confirmed=False)).run_once()

    (results,) = repository.completed
    assert results["layer_c"]["alerts"] == []
    summary = next(
        s for n, st, s in repository.steps if (n, st) == ("layer_c", "completed")
    )
    assert summary["detected"] == 1 and summary["infirmed"] == 1


async def test_a_qualification_failure_does_not_fail_the_run() -> None:
    repository = _layer_c_repository()
    qualifier = FakeQualifier(GradingUnavailable("AI_PROVIDER_UNAVAILABLE", "down"))

    await _layer_c_worker(repository, qualifier).run_once()

    (results,) = repository.completed
    (alert,) = results["layer_c"]["alerts"]
    assert alert["qualified"] is False
    assert results["rules"]["layer_c"]["ai"]["errors"] == ["AI_PROVIDER_UNAVAILABLE"]
    assert repository.failures == []


async def test_no_ai_call_when_the_layers_agree() -> None:
    repository = _layer_c_repository()
    qualifier = FakeQualifier()
    worker = RunWorker(
        repository,
        Settings(),
        worker_id="host-1",
        embedder=FakeEmbedder(),
        grader=FakeGrader(note=3),
        qualifier=qualifier,
    )

    await worker.run_once()

    assert qualifier.batches == []
    (results,) = repository.completed
    assert results["layer_c"]["alerts"] == []
    assert results["rules"]["layer_c"]["ai"]["status"] == "not_needed"


class FakeWriter:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.batches: list = []

    async def write(self, prompt, batch):
        self.batches.append(batch)
        if self.error:
            raise self.error
        projections = {
            persona: RawProjection(f"Action ({persona}).", "le critère est faible")
            for persona in ("decideur", "pmu", "analyste")
        }
        return (
            RawRestitution(
                items=tuple(
                    RawItem(key, f"Constat {key}.", tuple(batch.allowed[key])[:1])
                    for key, finding in batch.findings.items()
                    if finding.section is not None
                ),
                conclusion="Les résultats suggèrent une pertinence à étayer.",
                actions=tuple(
                    RawAction(key, tuple(batch.allowed[key])[:1], projections)
                    for key in batch.origins
                ),
            ),
            AiUsage("test-llm", 1500, 400, Decimal("0.0030")),
        )


def _restitution_worker(repository, writer) -> RunWorker:
    return RunWorker(
        repository,
        Settings(),
        worker_id="host-1",
        embedder=FakeEmbedder(),
        grader=FakeGrader(note=1),
        qualifier=FakeQualifier(),
        writer=writer,
    )


async def test_the_restitution_is_written_in_a_single_ai_call() -> None:
    repository = _layer_c_repository()
    writer = FakeWriter()

    await _restitution_worker(repository, writer).run_once()

    assert len(writer.batches) == 1
    (results,) = repository.completed
    restitution = results["restitution"]
    # Pertinence (note 1) est une faiblesse ; l'alerte ALERT-01 est à vérifier.
    assert [item["section"] for item in restitution["items"]] == [
        "faiblesse",
        "a_verifier",
    ]
    assert restitution["items"][0]["text"] == "Constat W1."
    assert restitution["synthesis"]["qualified"] is True
    assert restitution["synthesis"]["status"] == "completed"
    assert restitution["synthesis"]["conclusion"].startswith("Les résultats")
    assert [action["origin_type"] for action in restitution["actions"]] == [
        "criterion",
        "alert",
    ]
    assert all(len(action["projections"]) == 3 for action in restitution["actions"])
    rules = results["rules"]["restitution"]
    assert rules["engine_version"] == "restitution/1.1.0"
    assert rules["ai"]["status"] == "completed" and rules["ai"]["calls"] == 1
    assert rules["ai"]["prompt"] == "formulate_restitution_v1.0"
    assert results["summary"]["actions"] == 2
    assert repository.failures == []


async def test_without_ai_the_synthesis_keeps_factual_labels() -> None:
    repository = _layer_c_repository()

    await _restitution_worker(repository, None).run_once()

    (results,) = repository.completed
    restitution = results["restitution"]
    assert restitution["items"][0]["text"] == "Pertinence : note 1.0/3."
    assert restitution["synthesis"] == {
        "conclusion": None,
        "limits": restitution["synthesis"]["limits"],
        "qualified": False,
        "status": "disabled",
    }
    assert restitution["actions"] == []
    ai = results["rules"]["restitution"]["ai"]
    assert ai["status"] == "unavailable" and ai["errors"] == ["AI_DISABLED"]


async def test_a_writing_failure_does_not_fail_the_run() -> None:
    repository = _layer_c_repository()
    writer = FakeWriter(GradingUnavailable("AI_OUTPUT_TRUNCATED", "cut"))

    await _restitution_worker(repository, writer).run_once()

    (results,) = repository.completed
    assert results["restitution"]["synthesis"]["qualified"] is False
    assert results["restitution"]["synthesis"]["status"] == "failed"
    assert results["restitution"]["actions"] == []
    assert results["rules"]["restitution"]["ai"]["errors"] == ["AI_OUTPUT_TRUNCATED"]
    assert repository.failures == []


async def test_an_empty_restitution_needs_no_ai_call() -> None:
    repository = FakeRepository([_run()], _inputs())
    writer = FakeWriter()

    await _restitution_worker(repository, writer).run_once()

    assert writer.batches == []
    (results,) = repository.completed
    assert results["restitution"]["items"] == []
    assert results["restitution"]["synthesis"]["status"] == "not_needed"
    assert results["rules"]["restitution"]["ai"]["status"] == "not_needed"
