from uuid import UUID, uuid4

from app.modules.runs.domain import LayerAResults, Match, PillarScore, VariableResult
from app.modules.runs.layer_b import (
    Candidate,
    CriterionNote,
    LayerBResults,
    SubAnswerResult,
)
from app.modules.runs.layer_c import RULES as ALERT_RULES
from app.modules.runs.layer_c import Alert, LayerCInputs, PillarCriterion
from app.modules.runs.restitution import (
    RawAction,
    RawItem,
    RawProjection,
    RawRestitution,
    RestitutionRules,
    build_batch,
    interpret,
    select,
)

BUDGET = UUID("d0000000-0000-4000-8000-000000000001")
SUIVI = UUID("d0000000-0000-4000-8000-000000000002")
PERTINENCE = UUID("c1000000-0000-4000-8000-000000000001")
IMPACT = UUID("c1000000-0000-4000-8000-000000000002")
V1 = UUID("90000000-0000-4000-8000-000000000001")
V2 = UUID("90000000-0000-4000-8000-000000000002")
BUDGET_SEGMENT = UUID("0a000000-0000-4000-8000-000000000001")
SUIVI_SEGMENT = UUID("0a000000-0000-4000-8000-000000000002")
PERTINENCE_SEGMENT = UUID("0b000000-0000-4000-8000-000000000001")
IMPACT_SEGMENT = UUID("0b000000-0000-4000-8000-000000000002")
RULES = RestitutionRules()


def _inputs(locale: str = "fr") -> LayerCInputs:
    return LayerCInputs(
        locale=locale,
        pillar_names={BUDGET: "Budget et absorption", SUIVI: "Suivi-évaluation"},
        links=(
            PillarCriterion(BUDGET, PERTINENCE, "pertinence", "Pertinence"),
            PillarCriterion(SUIVI, IMPACT, "impact", "Impact"),
        ),
    )


def _score(
    pillar: UUID,
    version: UUID,
    score: float | None,
    delta: float | None = None,
    label: str | None = None,
    confidence: float = 0.8,
) -> PillarScore:
    return PillarScore(
        pillar_id=pillar,
        program_version_id=version,
        score=score,
        corpus_confidence=confidence,
        documents_count=3,
        delta=delta,
        evolution_label=label,
    )


SEGMENTS = {BUDGET: BUDGET_SEGMENT, SUIVI: SUIVI_SEGMENT}


def _layer_a(*scores: PillarScore, evidences: bool = True) -> LayerAResults:
    variables = [
        VariableResult(
            id=uuid4(),
            pillar_id=score.pillar_id,
            variable_code="has_budget_execution",
            program_version_id=score.program_version_id,
            value=1.0,
            state="present",
            documents_count=2,
            corpus_confidence=score.corpus_confidence,
            evidences=(
                [Match(SEGMENTS[score.pillar_id], uuid4(), uuid4(), 0.6)]
                if evidences
                else []
            ),
        )
        for score in scores
    ]
    return LayerAResults(variables=variables, scores=list(scores))


def _note(criterion: UUID, note: float | None, documented: int = 2) -> CriterionNote:
    return CriterionNote(
        criterion_id=criterion,
        note=note,
        status="conclu" if note is not None else "non_conclu",
        documented=documented,
        total=4,
        justification="",
        ambiguous=False,
    )


def _layer_b(*notes: CriterionNote) -> LayerBResults:
    segments = {PERTINENCE: PERTINENCE_SEGMENT, IMPACT: IMPACT_SEGMENT}
    return LayerBResults(
        notes=list(notes),
        answers=[
            SubAnswerResult(
                id=uuid4(),
                sub_question_id=uuid4(),
                status="documentee",
                note=1,
                contradictory=False,
                evidences=[
                    Candidate(uuid4(), segments[note.criterion_id], uuid4(), 0.5, "B")
                ],
                criterion_id=note.criterion_id,
            )
            for note in notes
        ],
    )


def _alert(
    severity: str = "majeure",
    pillar: UUID = BUDGET,
    criterion: UUID | None = PERTINENCE,
    message: str = "Budget et absorption : écart à vérifier.",
) -> Alert:
    return Alert(
        id=uuid4(),
        rule=ALERT_RULES["ALERT-01"],
        severity=severity,  # type: ignore[arg-type]
        pillar_id=pillar,
        pillar_name="Budget et absorption",
        criterion_id=criterion,
        criterion_name="Pertinence" if criterion else None,
        program_version_id=V2,
        score=72,
        note=1.0,
        delta=None,
        confidence=0.8,
        segment_ids=[PERTINENCE_SEGMENT, BUDGET_SEGMENT],
        message=message,
    )


def test_a_reinforced_pillar_is_a_progress_with_a_factual_label() -> None:
    restitution = select(
        _inputs(),
        _layer_a(
            _score(BUDGET, V1, 50),
            _score(BUDGET, V2, 70, delta=20, label="renforce"),
        ),
        None,
        [],
        RULES,
    )

    (progress,) = restitution.items("progres")
    assert progress.text == (
        "Budget et absorption : 70/100, +20 points depuis la version précédente."
    )
    assert (progress.pillar_id, progress.program_version_id) == (BUDGET, V2)
    assert progress.segment_ids == [BUDGET_SEGMENT]
    assert restitution.qualified is False and restitution.conclusion is None
    assert restitution.actions == []


def test_a_maintained_pillar_is_not_a_progress() -> None:
    restitution = select(
        _inputs(),
        _layer_a(
            _score(BUDGET, V1, 67),
            _score(BUDGET, V2, 70, delta=3, label="maintenu"),
        ),
        None,
        [],
        RULES,
    )

    assert restitution.items("progres") == []


def test_weaknesses_put_major_alerts_first_then_the_weakest() -> None:
    alert = _alert()
    restitution = select(
        _inputs(),
        _layer_a(_score(BUDGET, V2, 85), _score(SUIVI, V2, 20)),
        _layer_b(_note(PERTINENCE, 1.0)),
        [alert],
        RULES,
    )

    first, second = restitution.items("faiblesse")
    # Pertinence porte une alerte majeure : elle passe avant le pilier
    # pourtant plus faible ; Budget (85) n'est pas une faiblesse.
    assert (first.key, first.origin, first.criterion_id) == (
        "W1",
        "criterion",
        PERTINENCE,
    )
    assert first.text == "Pertinence : note 1.0/3."
    assert (second.key, second.pillar_id) == ("W2", SUIVI)
    assert second.text == "Suivi-évaluation : 20/100 (confiance corpus 80 %)."
    (to_verify,) = restitution.items("a_verifier")
    assert to_verify.alert_id == alert.id and to_verify.text == alert.message


def test_weaknesses_are_relative_below_eighty_percent() -> None:
    restitution = select(
        _inputs(),
        _layer_a(_score(BUDGET, V2, 62.5), _score(SUIVI, V2, 80)),
        _layer_b(_note(PERTINENCE, 1.8), _note(IMPACT, 2.4)),
        [],
        RULES,
    )

    # Un programme « bon » a quand même ses points les plus faibles ; un
    # pilier à 80/100 ou un critère à 2,4/3 n'en sont pas.
    assert [
        (item.criterion_id or item.pillar_id) for item in restitution.items("faiblesse")
    ] == [PERTINENCE, BUDGET]


def test_at_most_three_items_per_section_major_alerts_first() -> None:
    minor = [_alert("mineure", message=f"Mineure {index}.") for index in range(3)]
    major = [_alert("majeure", message=f"Majeure {index}.") for index in range(2)]

    restitution = select(
        _inputs(), _layer_a(_score(BUDGET, V2, 72)), None, minor + major, RULES
    )

    assert [item.text for item in restitution.items("a_verifier")] == [
        "Majeure 0.",
        "Majeure 1.",
        "Mineure 0.",
    ]


def test_a_candidate_without_evidence_is_not_retained() -> None:
    restitution = select(
        _inputs(),
        _layer_a(_score(SUIVI, V2, 20), evidences=False),
        None,
        [],
        RULES,
    )

    assert restitution.findings == []


def test_the_synthesis_leads_with_the_corpus_limits() -> None:
    def limits(layer_a: LayerAResults, *notes: CriterionNote):
        return select(_inputs(), layer_a, _layer_b(*notes), [], RULES).limits

    rich = _layer_a(_score(BUDGET, V2, 72))
    several = limits(rich, _note(PERTINENCE, None), _note(IMPACT, None))
    assert (several.not_concluded, several.criteria, several.leads) == (2, 2, True)
    one = limits(rich, _note(PERTINENCE, None), _note(IMPACT, 2.0))
    assert (one.confidence, one.leads) == (0.8, False)
    poor = limits(
        _layer_a(_score(BUDGET, V2, 72, confidence=0.3)), _note(PERTINENCE, 2.0)
    )
    assert poor.leads is True


def test_criteria_not_concluded_are_action_origins_only() -> None:
    restitution = select(
        _inputs(),
        _layer_a(_score(BUDGET, V2, 85)),
        _layer_b(_note(IMPACT, None, documented=1)),
        [],
        RULES,
    )

    (origin,) = restitution.origins()
    assert (origin.key, origin.section, origin.criterion_id) == ("N1", None, IMPACT)
    assert restitution.payload()["items"] == []


def _full_restitution():
    alert = _alert()
    restitution = select(
        _inputs(),
        _layer_a(
            _score(BUDGET, V1, 63),
            _score(BUDGET, V2, 85, delta=22, label="renforce"),
            _score(SUIVI, V2, 20),
        ),
        _layer_b(_note(PERTINENCE, 1.0), _note(IMPACT, None)),
        [alert],
        RULES,
    )
    texts = {
        BUDGET_SEGMENT: "Le taux d'exécution budgétaire atteint 85 %. " * 20,
        SUIVI_SEGMENT: "Le système de suivi reste manuel.",
        PERTINENCE_SEGMENT: "Les besoins ne sont pas analysés.",
        IMPACT_SEGMENT: "Les effets ne sont pas mesurés.",
    }
    batch = build_batch("fr", restitution, texts, RULES)
    assert batch is not None
    return restitution, batch, alert


def test_every_candidate_goes_in_one_compact_ai_call() -> None:
    restitution, batch, _ = _full_restitution()

    content = batch.content
    assert [entry["k"] for entry in content["p"]] == ["P1"]
    assert [entry["k"] for entry in content["w"]] == ["W1", "W2"]
    assert [entry["k"] for entry in content["v"]] == ["V1"]
    assert [entry["k"] for entry in content["n"]] == ["N1"]
    assert content["o"] == ["W1", "W2", "V1", "N1"]
    assert content["l"] == {"nc": 1, "tc": 2, "cf": 0.8, "ld": False}
    # Un extrait cité par plusieurs candidats n'est envoyé qu'une fois, tronqué.
    assert len(content["x"]) == 4
    assert all(len(text) <= RULES.excerpt_chars for text in content["x"].values())
    assert content["v"][0]["e"] == ["E2", "E1"]
    assert batch.estimated_input_tokens > 0 and batch.max_output_tokens > 0


def test_nothing_to_write_means_no_ai_call() -> None:
    restitution = select(_inputs(), _layer_a(_score(BUDGET, V2, 85)), None, [], RULES)

    assert build_batch("fr", restitution, {}, RULES) is None


def _projections(reason: str = "le critère est faiblement étayé"):
    return {
        persona: RawProjection(f"Action ({persona}).", reason)
        for persona in ("decideur", "pmu", "analyste")
    }


def test_the_ai_writes_items_conclusion_and_actions() -> None:
    restitution, batch, alert = _full_restitution()
    raw = RawRestitution(
        items=(
            RawItem("P1", "L'exécution budgétaire progresse.", ("E1", "E9")),
            RawItem("W1", "  ", ("E3",)),
            RawItem("V1", "Écart de pertinence à vérifier.", ("E3",)),
        ),
        conclusion="Les résultats suggèrent une exécution mieux documentée.",
        actions=(
            RawAction(
                "V1",
                ("E2",),
                _projections("Recommandée parce que l'écart A/B est majeur"),
            ),
            RawAction("W1", ("E3",), _projections()),
            RawAction(
                "W2",
                ("E2",),
                {"decideur": RawProjection("Décider.", "faible")},
            ),
            RawAction("P1", ("E1",), _projections()),
        ),
    )

    interpret(restitution, batch, raw)

    progress = restitution.items("progres")[0]
    assert progress.text == "L'exécution budgétaire progresse."
    # E9 n'existe pas : seul E1 est retenu.
    assert progress.segment_ids == [BUDGET_SEGMENT]
    weakness = restitution.items("faiblesse")[0]
    assert weakness.text == "Pertinence : note 1.0/3."
    to_verify = restitution.items("a_verifier")[0]
    # E3 n'est pas proposé pour l'alerte : ses preuves restent.
    assert to_verify.segment_ids == [PERTINENCE_SEGMENT, BUDGET_SEGMENT]
    assert restitution.conclusion.startswith("Les résultats suggèrent")
    # W2 sans ses trois projections et P1 hors des origines : ignorées ;
    # l'ordre suit celui des origines.
    assert [action.finding.key for action in restitution.actions] == ["W1", "V1"]
    v1 = restitution.actions[1]
    assert [projection.persona for projection in v1.projections] == [
        "decideur",
        "pmu",
        "analyste",
    ]
    assert v1.projections[0].justification == "l'écart A/B est majeur"
    assert v1.segment_ids == [PERTINENCE_SEGMENT]
    assert restitution.qualified is True

    payload = restitution.payload()
    assert payload["synthesis"]["qualified"] is True
    assert [(item["section"], item["position"]) for item in payload["items"]] == [
        ("progres", 1),
        ("faiblesse", 1),
        ("faiblesse", 2),
        ("a_verifier", 1),
    ]
    first, second = payload["actions"]
    assert (first["position"], first["origin_type"], first["criterion_id"]) == (
        1,
        "criterion",
        str(PERTINENCE),
    )
    assert (second["origin_type"], second["alert_id"]) == ("alert", str(alert.id))
    assert second["pillar_id"] is None and len(second["projections"]) == 3


def test_long_actions_and_a_narrative_conclusion_are_kept_whole() -> None:
    restitution, batch, _ = _full_restitution()
    action = "Documenter la mise en œuvre effective du volet. " * 6
    reason = "les rapports décrivent le dispositif prévu, pas sa mise en œuvre. " * 4
    paragraphs = [
        f"Paragraphe {index}, {'phrase prudente ' * 30}." for index in (1, 2, 3)
    ]
    raw = RawRestitution(
        items=(),
        conclusion="\n\n".join(paragraphs).replace("\n\n", "\n \n", 1),
        actions=(
            RawAction(
                "W1",
                ("E3",),
                {
                    persona: RawProjection(action, reason)
                    for persona in ("decideur", "pmu", "analyste")
                },
            ),
        ),
    )

    interpret(restitution, batch, raw)

    # Prompt v1.1 : une action et sa raison de 300 caractères ne sont plus
    # coupées ; la conclusion garde ses paragraphes.
    (written,) = restitution.actions
    assert written.projections[0].text == action.strip()
    assert written.projections[0].justification == reason.strip()
    assert restitution.conclusion.split("\n\n") == [
        " ".join(paragraph.split()) for paragraph in paragraphs
    ]

    raw = RawRestitution(items=(), conclusion="Mot " * 1000, actions=())
    interpret(restitution, batch, raw)
    assert len(restitution.conclusion) <= 2400
    assert restitution.conclusion.endswith("…")


def test_english_labels_follow_the_approach_locale() -> None:
    restitution = select(
        _inputs("en"), _layer_a(_score(SUIVI, V2, 20)), None, [], RULES
    )

    assert restitution.items("faiblesse")[0].text == (
        "Suivi-évaluation: 20/100 (corpus confidence 80%)."
    )
