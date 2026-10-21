from uuid import UUID, uuid4

from app.modules.runs.domain import LayerAResults, Match, PillarScore, VariableResult
from app.modules.runs.layer_b import (
    Candidate,
    CriterionNote,
    LayerBResults,
    SubAnswerResult,
)
from app.modules.runs.layer_c import (
    LayerCInputs,
    LayerCResults,
    LayerCRules,
    PillarCriterion,
    RawQualification,
    build_batches,
    detect,
    interpret,
    known_texts,
)

PILLAR = UUID("d0000000-0000-4000-8000-000000000001")
CRITERION = UUID("c1000000-0000-4000-8000-000000000001")
V1 = UUID("90000000-0000-4000-8000-000000000001")
V2 = UUID("90000000-0000-4000-8000-000000000002")
A_SEGMENT = UUID("0a000000-0000-4000-8000-000000000001")
B_SEGMENT = UUID("0b000000-0000-4000-8000-000000000001")
RULES = LayerCRules()


def _inputs(code: str = "pertinence", locale: str = "fr") -> LayerCInputs:
    return LayerCInputs(
        locale=locale,
        pillar_names={PILLAR: "Budget et absorption"},
        links=(PillarCriterion(PILLAR, CRITERION, code, "Pertinence"),),
    )


def _score(
    score: float | None,
    version: UUID = V2,
    confidence: float = 0.8,
    delta: float | None = None,
    label: str | None = None,
) -> PillarScore:
    return PillarScore(
        pillar_id=PILLAR,
        program_version_id=version,
        score=score,
        corpus_confidence=confidence,
        documents_count=3,
        delta=delta,
        evolution_label=label,
    )


def _layer_a(*scores: PillarScore, evidences: bool = True) -> LayerAResults:
    variables = [
        VariableResult(
            id=uuid4(),
            pillar_id=PILLAR,
            variable_code="has_budget_execution",
            program_version_id=score.program_version_id,
            value=1.0,
            state="present",
            documents_count=2,
            corpus_confidence=score.corpus_confidence,
            evidences=([Match(A_SEGMENT, uuid4(), uuid4(), 0.6)] if evidences else []),
        )
        for score in scores
    ]
    return LayerAResults(variables=variables, scores=list(scores))


def _layer_b(note: float | None, evidences: bool = True) -> LayerBResults:
    return LayerBResults(
        notes=[
            CriterionNote(
                criterion_id=CRITERION,
                note=note,
                status="conclu" if note is not None else "non_conclu",
                documented=2,
                total=3,
                justification="",
                ambiguous=False,
            )
        ],
        answers=[
            SubAnswerResult(
                id=uuid4(),
                sub_question_id=uuid4(),
                status="documentee",
                note=1,
                contradictory=False,
                evidences=(
                    [Candidate(uuid4(), B_SEGMENT, uuid4(), 0.5, "Extrait B.")]
                    if evidences
                    else []
                ),
                criterion_id=CRITERION,
            )
        ],
    )


def test_high_documentary_score_with_a_weak_rating_is_a_major_alert() -> None:
    (alert,) = detect(_inputs(), _layer_a(_score(62)), _layer_b(1.0), RULES)

    assert alert.rule.code == "ALERT-01"
    assert alert.rule.alert_type == "documentaire_sans_credibilite"
    assert alert.severity == "majeure"
    assert (alert.pillar_id, alert.criterion_id) == (PILLAR, CRITERION)
    # Preuves du critère d'abord, puis celles du pilier.
    assert alert.segment_ids == [B_SEGMENT, A_SEGMENT]
    assert alert.message.startswith("Budget et absorption : amélioration")
    assert "62/100" in alert.message and "Pertinence 1.0/3" in alert.message
    assert alert.qualified is False


def test_low_documentary_score_with_a_strong_rating_is_to_verify() -> None:
    (alert,) = detect(_inputs(), _layer_a(_score(35)), _layer_b(2.0), RULES)

    assert alert.rule.code == "ALERT-02" and alert.severity == "majeure"


def test_consistent_layers_raise_nothing() -> None:
    assert detect(_inputs(), _layer_a(_score(62)), _layer_b(2.5), RULES) == []
    assert detect(_inputs(), _layer_a(_score(50)), _layer_b(1.0), RULES) == []


def test_a_criterion_not_concluded_raises_no_alert() -> None:
    assert detect(_inputs(), _layer_a(_score(62)), _layer_b(None), RULES) == []
    assert detect(_inputs(), _layer_a(_score(62)), None, RULES) == []


def test_a_poor_corpus_makes_the_alert_minor() -> None:
    (alert,) = detect(
        _inputs(), _layer_a(_score(62, confidence=0.4)), _layer_b(1.0), RULES
    )

    assert alert.rule.code == "ALERT-01" and alert.severity == "mineure"


def test_the_criterion_rule_comes_first() -> None:
    (above,) = detect(_inputs("efficience"), _layer_a(_score(65)), _layer_b(0.5), RULES)
    (between,) = detect(
        _inputs("efficience"), _layer_a(_score(55)), _layer_b(0.5), RULES
    )
    (equity,) = detect(_inputs("equite"), _layer_a(_score(45)), _layer_b(1.0), RULES)
    (monitoring,) = detect(
        _inputs("gestion_adaptative"), _layer_a(_score(52)), _layer_b(1.0), RULES
    )

    assert above.rule.code == between.rule.code == "ALERT-03"
    assert above.severity == "mineure"
    assert equity.rule.code == "ALERT-04"
    assert monitoring.rule.code == "ALERT-05"


def test_the_latest_scored_version_is_compared() -> None:
    gap = detect(
        _inputs(), _layer_a(_score(30, V1), _score(70, V2)), _layer_b(1.0), RULES
    )
    unscored_latest = detect(
        _inputs(), _layer_a(_score(70, V1), _score(None, V2)), _layer_b(1.0), RULES
    )

    assert [(a.rule.code, a.program_version_id) for a in gap] == [("ALERT-01", V2)]
    assert [(a.rule.code, a.program_version_id) for a in unscored_latest] == [
        ("ALERT-01", V1)
    ]


def test_progress_without_corpus_confidence_is_flagged_on_the_pillar() -> None:
    (alert,) = detect(
        _inputs(),
        _layer_a(
            _score(40, V1, confidence=0.6),
            _score(55, V2, confidence=0.3, delta=15, label="renforce"),
        ),
        _layer_b(2.0),
        RULES,
    )

    assert alert.rule.code == "ALERT-06"
    assert alert.criterion_id is None and alert.severity == "mineure"
    assert alert.segment_ids == [A_SEGMENT]
    assert "+15 points" in alert.message and "30 %" in alert.message


def test_an_alert_without_evidence_is_not_created() -> None:
    assert (
        detect(
            _inputs(),
            _layer_a(_score(62), evidences=False),
            _layer_b(1.0, evidences=False),
            RULES,
        )
        == []
    )


def test_default_messages_follow_the_locale() -> None:
    (alert,) = detect(_inputs(locale="en"), _layer_a(_score(35)), _layer_b(2.0), RULES)

    assert alert.message.startswith("Budget et absorption: evaluative judgement")


def _alerts():
    return detect(
        _inputs(),
        _layer_a(
            _score(40, V1, confidence=0.6),
            _score(62, V2, confidence=0.45, delta=22, label="renforce"),
        ),
        _layer_b(1.0),
        RULES,
    )


def test_all_alerts_fit_in_one_call_with_shared_extracts() -> None:
    alerts = _alerts()
    texts = known_texts(_layer_b(1.0)) | {A_SEGMENT: "Extrait A."}

    (batch,) = build_batches(_inputs(), alerts, texts, RULES)

    assert [entry["r"] for entry in batch.content["a"]] == ["ALERT-01", "ALERT-06"]
    assert batch.content["x"] == {"E1": "Extrait B.", "E2": "Extrait A."}
    assert batch.content["a"][0]["e"] == ["E1", "E2"]
    assert batch.content["a"][1]["e"] == ["E2"]
    assert batch.content["a"][0]["n"] == 1.0 and "d" not in batch.content["a"][0]
    assert batch.content["a"][1]["d"] == 22 and "c" not in batch.content["a"][1]
    assert batch.content["lang"] == "fr"


def test_an_extract_without_text_is_not_sent() -> None:
    (batch,) = build_batches(_inputs(), _alerts(), {}, RULES)

    assert batch.content["x"] == {}
    assert all(entry["e"] == [] for entry in batch.content["a"])


def test_alerts_are_split_only_beyond_the_output_ceiling() -> None:
    rules = LayerCRules(max_output_tokens=100)

    batches = build_batches(_inputs(), _alerts(), {}, rules)

    assert [len(batch.alerts) for batch in batches] == [1, 1]
    assert [entry["k"] for entry in batches[1].content["a"]] == ["A1"]


def test_the_ai_confirms_qualifies_or_infirms_each_gap() -> None:
    alerts = _alerts()
    texts = known_texts(_layer_b(1.0)) | {A_SEGMENT: "Extrait A."}
    (batch,) = build_batches(_inputs(), alerts, texts, RULES)

    infirmed = interpret(
        batch,
        [
            RawQualification(
                "A1", True, "majeure", "  Message rédigé par l'IA.  ", ("E9", "E2")
            ),
            RawQualification("A1", False, "mineure", "doublon ignoré", ()),
            RawQualification("A2", False, "mineure", "", ()),
        ],
        RULES,
    )

    first, second = alerts
    assert infirmed == {second.id}
    # Corpus pauvre (0,45) : mineure quoi que dise l'IA.
    assert first.severity == "mineure"
    assert first.message == "Message rédigé par l'IA."
    # Seul l'extrait proposé pour l'écart est retenu.
    assert first.segment_ids == [A_SEGMENT]
    assert first.qualified is True


def test_an_unusable_answer_keeps_the_default_qualification() -> None:
    (alert,) = detect(_inputs(), _layer_a(_score(62)), _layer_b(1.0), RULES)
    (batch,) = build_batches(_inputs(), [alert], {}, RULES)
    message = alert.message

    interpret(batch, [RawQualification("A1", True, "grave", " ", ("E1",))], RULES)

    assert alert.severity == "majeure"
    assert alert.message == message
    assert alert.segment_ids == [B_SEGMENT, A_SEGMENT]


def test_payload_matches_complete_run() -> None:
    (alert,) = detect(_inputs(), _layer_a(_score(62)), _layer_b(1.0), RULES)
    results = LayerCResults([alert], detected=2, ai={"status": "completed", "calls": 1})

    (item,) = results.payload()["alerts"]

    assert item == {
        "id": str(alert.id),
        "pillar_id": str(PILLAR),
        "criterion_id": str(CRITERION),
        "program_version_id": str(V2),
        "rule_code": "ALERT-01",
        "alert_type": "documentaire_sans_credibilite",
        "severity": "majeure",
        "message": alert.message,
        "qualified": False,
        "segment_ids": [str(B_SEGMENT), str(A_SEGMENT)],
    }
    assert results.summary() == {
        "detected": 2,
        "alerts": 1,
        "major": 1,
        "infirmed": 1,
        "qualified": 0,
        "ai_status": "completed",
        "ai_calls": 1,
    }
