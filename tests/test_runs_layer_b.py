import json
from dataclasses import replace
from uuid import UUID, uuid4

from app.modules.runs.layer_b import (
    Candidate,
    Criterion,
    Grading,
    LayerBInputs,
    LayerBRules,
    RawGrade,
    RawGrading,
    RawGradingTranslation,
    SubQuestion,
    build_batches,
    finalize,
    interpret,
    propose,
    retrieve,
    truncate,
)
from app.modules.runs.project import ProjectContext

RULES = LayerBRules(min_similarity=0.4)
DOCUMENTS = [uuid4(), uuid4()]
# Une justification argumentée : un paragraphe sourcé.
JUSTIFIED = (
    "Le projet aligne ses activités sur les priorités nationales [E1] et ses "
    "rapports annuels détaillent les zones ciblées, les publics visés et les "
    "moyens consacrés à chacun. Les documents donnent ainsi une lecture "
    "cohérente de son ancrage, sans encore montrer comment ces priorités "
    "orientent ses arbitrages budgétaires."
)


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


def test_queries_carry_the_expected_evidence() -> None:
    criterion = _criterion("Efficience", 1)
    ((_, text),) = _inputs(criterion).queries()
    assert text == "Efficience : Efficience question 0?\nRapport de suivi"
    bare = replace(criterion, questions=(SubQuestion(uuid4(), "Q?", None),))
    assert _inputs(bare).queries()[0][1] == "Efficience : Q?"


def test_a_segment_found_by_its_words_is_kept_under_a_lower_threshold() -> None:
    rules = LayerBRules(min_similarity=0.4, lexical_min_similarity=0.3)
    question = SubQuestion(uuid4(), "Q?")
    by_words = replace(_candidate(question, 0.33), lexical_rank=1, score=0.03)
    by_vector_only = _candidate(question, 0.33)
    too_far = replace(_candidate(question, 0.2), lexical_rank=2, score=0.02)

    kept = retrieve([by_words, by_vector_only, too_far], rules)[question.id]

    assert [candidate.segment_id for candidate in kept] == [by_words.segment_id]


def test_fused_score_orders_the_extracts() -> None:
    question = SubQuestion(uuid4(), "Q?")
    close = replace(_candidate(question, 0.7), score=0.016)
    both = replace(_candidate(question, 0.5), lexical_rank=1, score=0.032)

    kept = retrieve([close, both], RULES)[question.id]

    assert [candidate.segment_id for candidate in kept] == [
        both.segment_id,
        close.segment_id,
    ]


def test_one_document_does_not_take_every_place() -> None:
    rules = LayerBRules(min_similarity=0.4, top_k=4, max_per_document=2)
    question = SubQuestion(uuid4(), "Q?")
    report, audit = uuid4(), uuid4()
    first = [
        replace(_candidate(question, 0.9 - index / 100), document_id=report)
        for index in range(4)
    ]
    other = replace(_candidate(question, 0.5), document_id=audit)

    kept = retrieve([*first, other], rules)[question.id]

    assert len(kept) == 4
    assert other in kept
    # La dernière place revient au document principal faute d'autre source.
    assert sum(candidate.document_id == report for candidate in kept) == 3
    assert kept == sorted(kept, key=lambda candidate: -candidate.similarity)


def test_every_sub_question_is_sent_to_the_ai() -> None:
    pertinence = _criterion("Pertinence", 3)
    impact = _criterion("Impact", 3)
    candidates = [_candidate(question) for question in pertinence.questions[:2]]
    candidates.append(_candidate(impact.questions[0]))

    batches = build_batches(
        _inputs(pertinence, impact), retrieve(candidates, RULES), RULES
    )

    # Même sous 50 % (Impact) ou sans extrait : chaque sous-question reçoit
    # une réponse ; RG-1 ne joue que sur la note.
    (batch,) = batches
    assert batch.criteria == [pertinence, impact]
    assert len(batch.question_keys) == 6
    content = json.loads(batch.message())
    assert content["c"][0]["q"][2]["e"] == []


def test_closest_extracts_are_proposed_below_the_threshold() -> None:
    criterion = _criterion("Pertinence", 2)
    first, second = criterion.questions
    above = _candidate(first, 0.6)
    below = [_candidate(second, 0.3), _candidate(second, 0.2), _candidate(second, 0.1)]
    retrieved = retrieve([above, *below], RULES)

    proposed = propose([above, *below], retrieved, RULES)

    assert second.id not in retrieved
    assert proposed[first.id] == [above]
    assert proposed[second.id] == below[:2]
    assert propose(below, {}, replace(RULES, fallback_excerpts=0)) == {}


def test_an_undocumented_sub_question_keeps_its_answer_and_sources() -> None:
    criterion = _criterion("Pertinence", 3)
    first, second, third = criterion.questions
    candidates = [
        _candidate(first),
        _candidate(second),
        _candidate(third, 0.3),  # sous le seuil : proposé, pas retrouvé
    ]
    retrieved = retrieve(candidates, RULES)
    proposed = propose(candidates, retrieved, RULES)
    (batch,) = build_batches(_inputs(criterion), proposed, RULES)
    keys = list(batch.question_keys)
    closest = (
        "Les documents du projet évoquent un comité de suivi [E3] sans dire "
        "comment il consulte les communautés ni à quelle fréquence."
    )
    unsourced = (
        "Les documents du projet ne renseignent pas ce point : il faudrait y "
        "trouver un rapport de suivi."
    )
    grading = Grading()
    interpret(
        batch,
        RawGrading(
            grades=(
                RawGrade(keys[0], 3, ("E1",), False, JUSTIFIED.replace("[E1]", "[E1]")),
                RawGrade(keys[1], None, (), False, unsourced),
                RawGrade(keys[2], None, (), False, closest),
            ),
            justifications={"C1": JUSTIFIED},
        ),
        grading,
    )
    results = finalize(_inputs(criterion), retrieved, grading, RULES, proposed=proposed)

    answers = {answer.sub_question_id: answer for answer in results.answers}
    assert [answers[q.id].status for q in criterion.questions] == [
        "documentee",
        "non_documentee",
        "non_documentee",
    ]
    # Réponse sans source : publiée quand même, sans preuve.
    assert answers[second.id].interpretation.text.startswith("Les documents du projet")
    assert answers[second.id].interpretation.citations == []
    assert answers[second.id].evidences == []
    # Réponse sourcée : sa preuve est l'extrait proposé sous le seuil.
    assert "⟦1⟧" in answers[third.id].interpretation.text
    assert [c.segment_id for c in answers[third.id].evidences] == [
        candidates[2].segment_id
    ]
    assert answers[third.id].note is None
    payload = results.payload()
    assert len(payload["evidences"]) == 2
    # RG-1 : une sur trois documentée, pas de note ; justification de l'IA.
    (note,) = results.notes
    assert note.status == "non_conclu" and note.note is None
    assert note.written is not None


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
    # Interprétations et justifications, rédigées puis traduites (tr).
    assert content["tr"] == ["en"]
    assert batch.locales == ("en",)
    assert batch.max_output_tokens == 2 * (4 * 260 + 2 * 480)


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
    assert all(
        sum(len(sentence) for sentence in sentences) <= 100
        for sentences in content["x"].values()
    )


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
        justifications={"C1": JUSTIFIED},
    )

    interpret(batch, raw, grading)
    results = finalize(_inputs(criterion), retrieved, grading, RULES)

    (note,) = results.notes
    assert note.status == "conclu"
    assert note.note == 2.5
    assert (note.documented, note.total) == (2, 3)
    assert note.justification.startswith(
        "Le projet aligne ses activités sur les priorités nationales⟦1⟧ et ses"
    )
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


def test_interpretations_and_justifications_cite_their_sources() -> None:
    criterion = _criterion("Pertinence", 1)
    question = criterion.questions[0]
    text = (
        "Le projet répond aux priorités nationales. "
        "Le comité valide le plan chaque année."
    )
    segment = uuid4()
    candidates = [Candidate(question.id, segment, DOCUMENTS[0], 0.6, text)]
    retrieved = retrieve(candidates, RULES)
    (batch,) = build_batches(_inputs(criterion), retrieved, RULES)
    content = json.loads(batch.message())
    assert content["x"] == {
        "E1": [
            "Le projet répond aux priorités nationales.",
            "Le comité valide le plan chaque année.",
        ]
    }
    interpretation = (
        "Le projet soumet son plan à la validation annuelle de son comité "
        "[E1.2, E3], ce qui montre une instance de pilotage en place. Les "
        "documents ne disent pas ce que le comité modifie dans le plan."
    )
    translated = (
        "The project submits its plan to its board every year [E1.2]. The "
        "documents do not say what the board changes in the plan itself."
    )
    justification = (
        "Le plan est validé chaque année [E1.2, E5] et le projet se réfère "
        "explicitement aux priorités nationales [E1.1]. Ces éléments montrent "
        "une conception alignée, mais les documents ne disent rien de la "
        "manière dont les besoins des bénéficiaires ont été recueillis."
    )
    raw = RawGrading(
        grades=(RawGrade("Q1", 3, ("E1.2",), False, interpretation),),
        justifications={"C1": justification},
        translations={
            "en": RawGradingTranslation(interpretations={"Q1": translated}),
            "pt": RawGradingTranslation(interpretations={"Q1": translated}),
        },
    )
    grading = Grading()

    interpret(batch, raw, grading)
    results = finalize(_inputs(criterion), retrieved, grading, RULES)

    first = (0, text.index(" Le comité"))
    second = (text.index("Le comité"), len(text))
    (answer,) = results.answers
    assert answer.spans == {segment: (second,)}
    # E3 n'est pas un extrait de la sous-question : sa référence disparaît.
    reading = answer.interpretation
    assert reading is not None
    assert reading.text.startswith(
        "Le projet soumet son plan à la validation annuelle de son comité⟦1⟧, ce"
    )
    # Seule la langue demandée (tr) compte.
    assert set(reading.translations) == {"en"}
    assert reading.translations["en"].startswith(
        "The project submits its plan to its board every year⟦1⟧."
    )
    assert [source.as_dict() for source in reading.citations] == [
        {"segment_id": str(segment), "spans": [list(second)]}
    ]
    (note,) = results.notes
    # E5 n'est pas un extrait du critère : seules ses sources E1 restent.
    assert note.justification.startswith(
        "Le plan est validé chaque année⟦1⟧ et le projet se réfère "
        "explicitement aux priorités nationales⟦2⟧."
    )
    payload = results.payload()
    assert payload["answers"][0]["interpretation"] == reading.text
    assert payload["answers"][0]["interpretation_translations"] == {
        "en": reading.translations["en"]
    }
    assert payload["answers"][0]["interpretation_citations"] == [
        {"segment_id": str(segment), "spans": [list(second)]}
    ]
    # La lecture de l'extrait vient de l'étape de lecture, pas d'ici.
    assert "interpretation" not in payload["evidences"][0]
    assert payload["evidences"][0]["spans"] == [list(second)]
    assert payload["notes"][0]["citations"] == [
        {"segment_id": str(segment), "spans": [list(second)]},
        {"segment_id": str(segment), "spans": [list(first)]},
    ]
    assert payload["notes"][0]["justification_translations"] == {}


def test_a_short_or_unsourced_text_is_not_published() -> None:
    criterion = _criterion("Pertinence", 1)
    question = criterion.questions[0]
    text = "Le projet cible trois régions. Le comité valide le plan chaque année."
    retrieved = retrieve(
        [Candidate(question.id, uuid4(), DOCUMENTS[0], 0.6, text)], RULES
    )
    (batch,) = build_batches(_inputs(criterion), retrieved, RULES)
    unsourced = (
        "Le projet concentre son action sur un nombre restreint de régions, "
        "ce qui facilite son pilotage, mais les documents ne précisent pas "
        "comment ces régions ont été choisies ni quels besoins y ont été relevés."
    )
    raw = RawGrading(
        grades=(RawGrade("Q1", 2, ("E1.1",), False, "Le projet est ciblé [E1.1]."),),
        justifications={"C1": unsourced},
    )
    grading = Grading()

    interpret(batch, raw, grading)
    results = finalize(_inputs(criterion), retrieved, grading, RULES)

    # La note tient ; la phrase de circonstance n'est pas publiée.
    assert grading.grades[question.id].note == 2
    assert grading.grades[question.id].interpretation is None
    # Sans source, la justification cède la place au libellé factuel.
    (note,) = results.notes
    assert note.justification == "Moyenne des notes de 1 sous-questions documentées."
    assert note.written is None


def test_the_evaluated_project_goes_with_the_message() -> None:
    criterion = _criterion("Pertinence", 1)
    question = criterion.questions[0]
    retrieved = retrieve([_candidate(question)], RULES)
    project = ProjectContext.from_row(
        {
            "name": "Résilience Sahel",
            "kind": "project",
            "description": "  Irrigation   et stockage. ",
            "country": "Niger",
            "financiers": ["FIDA", "FIDA", ""],
            "themes": [],
            "start_year": 2021,
            "end_year": 2026,
        }
    )
    inputs = LayerBInputs("fr", "mi_parcours", (criterion,), project)

    (batch,) = build_batches(inputs, retrieved, RULES)

    assert batch.content["pj"] == {
        "n": "Résilience Sahel",
        "t": "project",
        "d": "Irrigation et stockage.",
        "pa": "Niger",
        "f": ["FIDA"],
        "an": "2021-2026",
    }
    # Sans projet connu, le message n'a pas de bloc pj.
    (bare,) = build_batches(_inputs(criterion), retrieved, RULES)
    assert "pj" not in bare.content
