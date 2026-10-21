import json
from uuid import UUID, uuid4

from app.modules.runs.layer_b import (
    Candidate,
    Criterion,
    Grading,
    LayerBInputs,
    LayerBRules,
    RawGrade,
    RawGrading,
    SubQuestion,
    build_batches,
    finalize,
    interpret,
    retrieve,
    truncate,
)

RULES = LayerBRules(min_similarity=0.4)
DOCUMENTS = [uuid4(), uuid4()]


def _criterion(name: str, questions: int, applicability: str = "obligatoire"):
    return Criterion(
        id=uuid4(),
        code=name.lower(),
        name=name,
        definition=f"Définition de {name}",
        applicability=applicability,
        questions=tuple(
            SubQuestion(uuid4(), f"{name} question {index}?", "Rapport de suivi")
            for index in range(questions)
        ),
    )


def _candidate(question: SubQuestion, similarity=0.6, segment: UUID | None = None):
    return Candidate(
        question.id,
        segment or uuid4(),
        DOCUMENTS[0],
        similarity,
        f"Extrait pour {question.text} " * 3,
    )


def _inputs(*criteria: Criterion) -> LayerBInputs:
    return LayerBInputs(locale="fr", cycle="mi_parcours", criteria=criteria)


def _grade_all(batch, note=2, contradictory=False) -> RawGrading:
    return RawGrading(
        grades=tuple(
            RawGrade(
                key, note, (next(iter(sorted(batch.allowed[qid]))),), contradictory
            )
            for key, qid in batch.question_keys.items()
        ),
        justifications={key: "Justification fondée." for key in batch.criterion_keys},
    )


def test_retrieve_keeps_the_closest_extracts_above_the_threshold() -> None:
    question = SubQuestion(uuid4(), "Q?")
    segment = uuid4()
    candidates = [
        _candidate(question, 0.39),
        _candidate(question, 0.5, segment),
        _candidate(question, 0.7, segment),
        *[_candidate(question, 0.41 + index / 100) for index in range(6)],
    ]

    kept = retrieve(candidates, RULES)[question.id]

    assert len(kept) == RULES.top_k
    assert kept[0].segment_id == segment and kept[0].similarity == 0.7
    assert all(candidate.similarity >= 0.4 for candidate in kept)
    assert len({candidate.segment_id for candidate in kept}) == len(kept)


def test_a_criterion_under_half_documented_is_not_sent_to_the_ai() -> None:
    pertinence = _criterion("Pertinence", 3)
    impact = _criterion("Impact", 3)
    candidates = [_candidate(question) for question in pertinence.questions[:2]]
    candidates.append(_candidate(impact.questions[0]))

    batches = build_batches(
        _inputs(pertinence, impact), retrieve(candidates, RULES), RULES
    )

    (batch,) = batches
    assert batch.criteria == [pertinence]
    assert len(batch.question_keys) == 2


def test_the_whole_run_fits_in_one_call_with_shared_extracts() -> None:
    first = _criterion("Pertinence", 2)
    second = _criterion("Cohérence", 2)
    shared = uuid4()
    candidates = [
        _candidate(question, 0.6, shared)
        for question in (*first.questions, *second.questions)
    ]

    (batch,) = build_batches(_inputs(first, second), retrieve(candidates, RULES), RULES)

    content = json.loads(batch.message())
    assert list(content["x"]) == ["E1"]  # un extrait partagé n'est envoyé qu'une fois
    assert [criterion["k"] for criterion in content["c"]] == ["C1", "C2"]
    assert content["c"][0]["q"][0] == {
        "k": "Q1",
        "t": first.questions[0].text,
        "e": ["E1"],
        "p": "Rapport de suivi",
    }
    assert batch.estimated_input_tokens > 0
    assert batch.max_output_tokens == 4 * 30 + 2 * 120


def test_extracts_are_capped_and_truncated() -> None:
    criterion = _criterion("Pertinence", 1)
    question = criterion.questions[0]
    long_text = "mot " * 400
    candidates = [
        Candidate(question.id, uuid4(), DOCUMENTS[0], 0.6 - index / 100, long_text)
        for index in range(5)
    ]
    rules = LayerBRules(min_similarity=0.4, excerpts_per_question=2, excerpt_chars=100)

    (batch,) = build_batches(_inputs(criterion), retrieve(candidates, rules), rules)

    content = json.loads(batch.message())
    assert len(content["x"]) == 2
    assert all(len(text) <= 100 for text in content["x"].values())


def test_criteria_are_split_only_beyond_the_token_ceiling() -> None:
    criteria = [_criterion(f"Critère {index}", 2) for index in range(4)]
    candidates = [
        _candidate(question)
        for criterion in criteria
        for question in criterion.questions
    ]
    retrieved = retrieve(candidates, RULES)
    small = LayerBRules(min_similarity=0.4, max_input_tokens=1200)

    assert len(build_batches(_inputs(*criteria), retrieved, RULES)) == 1
    batches = build_batches(_inputs(*criteria), retrieved, small)
    assert len(batches) > 1
    assert sum(len(batch.criteria) for batch in batches) == 4


def test_ratings_average_the_documented_sub_questions() -> None:
    criterion = _criterion("Pertinence", 3)
    candidates = [_candidate(question) for question in criterion.questions]
    retrieved = retrieve(candidates, RULES)
    (batch,) = build_batches(_inputs(criterion), retrieved, RULES)
    keys = list(batch.question_keys)
    grading = Grading()
    raw = RawGrading(
        grades=(
            RawGrade(keys[0], 3, ("E1",), False),
            RawGrade(keys[1], 2, ("E2",), False),
            RawGrade(keys[2], None, (), False),
        ),
        justifications={"C1": "Les extraits montrent un alignement net."},
    )

    interpret(batch, raw, grading)
    results = finalize(_inputs(criterion), retrieved, grading, RULES)

    (note,) = results.notes
    assert note.status == "conclu"
    assert note.note == 2.5
    assert (note.documented, note.total) == (2, 3)
    assert note.justification == "Les extraits montrent un alignement net."
    assert not note.ambiguous
    statuses = [answer.status for answer in results.answers]
    assert statuses == ["documentee", "documentee", "non_documentee"]
    assert results.answers[2].evidences == []
    assert [c.segment_id for c in results.answers[0].evidences] == [
        retrieved[criterion.questions[0].id][0].segment_id
    ]


def test_a_rating_must_cite_one_of_its_own_extracts() -> None:
    criterion = _criterion("Pertinence", 2)
    candidates = [_candidate(question) for question in criterion.questions]
    retrieved = retrieve(candidates, RULES)
    (batch,) = build_batches(_inputs(criterion), retrieved, RULES)
    keys = list(batch.question_keys)
    grading = Grading()
    raw = RawGrading(
        grades=(
            RawGrade(keys[0], 3, ("E2",), False),  # extrait d'une autre question
            RawGrade(keys[1], 5, ("E2",), False),  # hors barème
        ),
        justifications={},
    )

    interpret(batch, raw, grading)
    results = finalize(_inputs(criterion), retrieved, grading, RULES)

    (note,) = results.notes
    assert note.status == "non_conclu" and note.note is None
    assert note.documented == 0
    assert "Moins de 50 %" in note.justification


def test_contradictory_or_dispersed_ratings_are_flagged() -> None:
    criterion = _criterion("Pertinence", 2)
    retrieved = retrieve([_candidate(q) for q in criterion.questions], RULES)
    (batch,) = build_batches(_inputs(criterion), retrieved, RULES)

    contradictory = Grading()
    interpret(batch, _grade_all(batch, contradictory=True), contradictory)
    assert (
        finalize(_inputs(criterion), retrieved, contradictory, RULES).notes[0].ambiguous
    )

    keys = list(batch.question_keys)
    dispersed = Grading()
    interpret(
        batch,
        RawGrading(
            (
                RawGrade(keys[0], 0, ("E1",), False),
                RawGrade(keys[1], 3, ("E2",), False),
            ),
            {},
        ),
        dispersed,
    )
    note = finalize(_inputs(criterion), retrieved, dispersed, RULES).notes[0]
    assert note.ambiguous and note.note == 1.5
    assert note.justification == "Moyenne des notes de 2 sous-questions documentées."


def test_without_ai_nothing_is_rated() -> None:
    criterion = _criterion("Pertinence", 2)
    retrieved = retrieve([_candidate(q) for q in criterion.questions], RULES)

    results = finalize(_inputs(criterion), retrieved, Grading(), RULES)

    (note,) = results.notes
    assert note.status == "non_conclu" and note.note is None
    assert (note.documented, note.total) == (2, 2)
    assert "indisponible" in note.justification
    assert all(answer.note is None for answer in results.answers)
    assert all(answer.evidences for answer in results.answers)


def test_a_criterion_without_questions_is_not_concluded() -> None:
    criterion = _criterion("Pertinence", 0)

    results = finalize(_inputs(criterion), {}, Grading(), RULES)

    assert results.notes[0].status == "non_conclu"
    assert results.notes[0].justification == "Aucune sous-question active."
    assert build_batches(_inputs(criterion), {}, RULES) == []


def test_payload_matches_complete_run() -> None:
    criterion = _criterion("Pertinence", 1)
    retrieved = retrieve([_candidate(q) for q in criterion.questions], RULES)
    (batch,) = build_batches(_inputs(criterion), retrieved, RULES)
    grading = Grading()
    interpret(batch, _grade_all(batch, note=1), grading)

    payload = finalize(_inputs(criterion), retrieved, grading, RULES).payload()

    assert payload["notes"][0]["note"] == 1.0
    assert payload["notes"][0]["documented_sub_questions"] == 1
    assert payload["answers"][0]["note"] == 1
    assert payload["evidences"][0]["sub_answer_id"] == payload["answers"][0]["id"]


def test_truncate_cuts_on_a_word() -> None:
    assert truncate("un deux   trois", 50) == "un deux trois"
    assert truncate("alpha beta gamma delta", 12) == "alpha beta…"
