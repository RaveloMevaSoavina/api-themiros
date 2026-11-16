import json
from uuid import UUID, uuid4

from app.modules.runs.citations import written
from app.modules.runs.domain import LayerAResults, Match, PillarScore, VariableResult
from app.modules.runs.layer_b import Candidate, LayerBResults, SubAnswerResult
from app.modules.runs.project import ProjectContext
from app.modules.runs.reading import (
    RawReading,
    RawReadingTranslation,
    ReadingInputs,
    ReadingResults,
    ReadingRules,
    VariableLabel,
    build_batches,
    interpret,
    segments,
    subjects,
)

PILLAR = UUID("d0000000-0000-4000-8000-000000000001")
V1 = UUID("90000000-0000-4000-8000-000000000001")
V2 = UUID("90000000-0000-4000-8000-000000000002")
DOCUMENT = UUID("dd000000-0000-4000-8000-000000000001")
A1 = UUID("0a000000-0000-4000-8000-000000000001")
A2 = UUID("0a000000-0000-4000-8000-000000000002")
B1 = UUID("0b000000-0000-4000-8000-000000000001")
RULES = ReadingRules()
PAD = (
    " Les documents du projet décrivent ce point en détail, sans préciser "
    "encore comment il se traduit dans sa mise en œuvre effective."
)
EN_PAD = (
    " The project documents describe this point in great detail, without "
    "yet stating how it translates into its actual day-to-day implementation."
)
TEXTS = {
    A1: "Le budget est exécuté à 85 %. Les rapports sont trimestriels.",
    A2: "Le comité valide les dépenses.",
    B1: "Les besoins des ménages sont analysés.",
}


def _variable(code: str, version: UUID, *evidences: UUID, state="present"):
    return VariableResult(
        id=uuid4(),
        pillar_id=PILLAR,
        variable_code=code,
        program_version_id=version,
        value=1.0 if state == "present" else None,
        state=state,
        documents_count=len(evidences),
        corpus_confidence=0.8 if evidences else None,
        evidences=[Match(segment, DOCUMENT, uuid4(), 0.6) for segment in evidences],
    )


def _layer_a() -> LayerAResults:
    return LayerAResults(
        variables=[
            _variable("execution", V1, A2),
            _variable("execution", V2, A1, A2),
            _variable("audit", V2, state="non_renseigne"),
        ],
        scores=[
            PillarScore(PILLAR, V1, 50, 0.6, 1),
            PillarScore(PILLAR, V2, 80, 0.8, 1, delta=30, evolution_label="renforce"),
        ],
    )


def _inputs(locale: str = "fr") -> ReadingInputs:
    return ReadingInputs(
        locale=locale,
        pillar_names={PILLAR: "Budget et absorption"},
        variables={
            (PILLAR, "execution"): VariableLabel("Exécution budgétaire", "Dépenses."),
            (PILLAR, "audit"): VariableLabel("Audit externe"),
        },
        project=ProjectContext(name="Résilience Sahel"),
    )


def _layer_b() -> LayerBResults:
    return LayerBResults(
        notes=[],
        answers=[
            SubAnswerResult(
                id=uuid4(),
                sub_question_id=uuid4(),
                status="documentee",
                note=2,
                contradictory=False,
                evidences=[
                    Candidate(uuid4(), segment, DOCUMENT, 0.6, TEXTS[segment])
                    for segment in (B1, A1)
                ],
            )
        ],
    )


def test_the_latest_scored_version_of_each_pillar_is_read() -> None:
    (subject,) = subjects(_layer_a(), _inputs(), RULES)

    assert subject.score.program_version_id == V2
    assert [variable.variable_code for variable in subject.variables] == ["execution"]
    # Les volets dont les documents ne parlent pas sont nommés, pas lus.
    assert subject.missing == ["Audit externe"]
    assert subject.all_segments() == [A1, A2]


def test_each_segment_is_read_once_across_the_layers() -> None:
    pillars = subjects(_layer_a(), _inputs(), RULES)

    found = segments(pillars, _layer_b(), [[B1, A2], [uuid4()]])

    assert found[:3] == [A1, A2, B1]
    assert len(found) == 4


def test_one_call_carries_extracts_variables_and_pillars() -> None:
    pillars = subjects(_layer_a(), _inputs(), RULES)
    to_read = segments(pillars, _layer_b(), [])

    (batch,) = build_batches(_inputs(), pillars, to_read, TEXTS, RULES)

    content = json.loads(batch.message())
    assert content["lang"] == "fr" and content["tr"] == ["en"]
    assert content["pj"] == {"n": "Résilience Sahel"}
    assert content["x"] == {
        "E1": ["Le budget est exécuté à 85 %.", "Les rapports sont trimestriels."],
        "E2": ["Le comité valide les dépenses."],
        "E3": ["Les besoins des ménages sont analysés."],
    }
    assert content["r"] == ["E1", "E2", "E3"]
    assert content["v"] == [
        {
            "k": "V1",
            "p": "Budget et absorption",
            "n": "Exécution budgétaire",
            "s": "present",
            "dc": 2,
            "e": ["E1", "E2"],
            "d": "Dépenses.",
            "cf": 0.8,
        }
    ]
    assert content["p"] == [
        {
            "k": "P1",
            "n": "Budget et absorption",
            "s": 80,
            "cf": 0.8,
            "v": ["V1"],
            "e": ["E1", "E2"],
            "dl": 30,
            "m": ["Audit externe"],
        }
    ]
    # Trois lectures, une variable et un pilier, rédigés puis traduits.
    assert batch.max_output_tokens == 2 * (3 * 280 + 280 + 420)


def test_a_segment_is_read_once_even_when_the_call_is_split() -> None:
    pillars = subjects(_layer_a(), _inputs(), RULES)
    to_read = segments(pillars, _layer_b(), [])
    small = ReadingRules(max_output_tokens=500)

    batches = build_batches(_inputs(), pillars, to_read, TEXTS, small)

    assert len(batches) == 2
    read = [batch.excerpt_keys[key] for batch in batches for key in batch.to_read]
    assert sorted(read) == sorted([A1, A2, B1])
    # Le pilier part avec ses variables et leurs extraits.
    assert list(batches[0].pillar_keys) == ["P1"]
    assert batches[1].pillar_keys == {}


def test_without_text_nothing_is_sent() -> None:
    pillars = subjects(_layer_a(), _inputs(), RULES)

    assert build_batches(_inputs(), pillars, [A1, A2], {}, RULES) == []


def test_texts_cite_their_own_sources_and_are_translated() -> None:
    pillars = subjects(_layer_a(), _inputs(), RULES)
    (batch,) = build_batches(_inputs(), pillars, [A1, A2, B1], TEXTS, RULES)
    raw = RawReading(
        segments={
            "e1": "Le projet exécute son budget [E1.1, E3]." + PAD,
            "E2": "Trop court [E2].",
            "E3": "Le projet analyse les besoins des ménages." + PAD,
        },
        variables={"V1": "Le projet suit ses dépenses [E1.2, E3]." + PAD},
        pillars={"P1": "Le projet gère ses finances [E2]." + PAD},
        translations={
            "en": RawReadingTranslation(
                segments={"E1": "The project spends its budget [E1.1]." + EN_PAD},
                pillars={"P1": "The project manages its finances [E2]." + EN_PAD * 2},
            ),
            "es": RawReadingTranslation(segments={"E1": "El proyecto." + EN_PAD}),
        },
    )
    results = ReadingResults()

    interpret(batch, raw, results)

    budget = batch.excerpts["E1"]
    reading = results.readings[A1]
    # Une lecture ne cite que son extrait : E3 disparaît.
    assert reading.text == "Le projet exécute son budget⟦1⟧." + PAD
    assert reading.translations == {"en": "The project spends its budget⟦1⟧." + EN_PAD}
    assert [source.spans for source in reading.citations] == [(budget.spans[0],)]
    assert reading.spans == (budget.spans[0],)
    # Trop courte : non publiée.
    assert A2 not in results.readings
    # Sans source, la lecture renvoie à l'extrait entier.
    assert results.readings[B1].text.startswith("Le projet analyse les besoins")
    assert "⟦1⟧" in results.readings[B1].text
    variable = next(iter(results.variables.values()))
    # Une variable ne cite que ses extraits : E3 (couche B) disparaît.
    assert variable.text == "Le projet suit ses dépenses⟦1⟧." + PAD
    # Un pilier trop court n'est pas publié.
    assert results.pillars == {}


def test_results_go_into_the_complete_run_payload() -> None:
    layer_a = _layer_a()
    pillars = subjects(layer_a, _inputs(), RULES)
    (batch,) = build_batches(_inputs(), pillars, [A1], TEXTS, RULES)
    raw = RawReading(
        segments={"E1": "Le projet exécute son budget [E1.1]." + PAD},
        variables={"V1": "Le projet suit ses dépenses [E1.2]." + PAD},
        pillars={"P1": "Le projet gère ses finances [E2]." + PAD * 2},
    )
    results = ReadingResults()
    interpret(batch, raw, results)
    payload = {
        "variables": [{"id": str(variable.id)} for variable in layer_a.variables],
        "scores": [
            {
                "pillar_id": str(score.pillar_id),
                "program_version_id": str(score.program_version_id),
            }
            for score in layer_a.scores
        ],
    }

    results.apply(payload)

    (reading,) = payload["readings"]
    assert reading["segment_id"] == str(A1)
    assert reading["interpretation_citations"] == [
        {"segment_id": str(A1), "spans": [list(batch.excerpts["E1"].spans[0])]}
    ]
    interpreted = [item for item in payload["variables"] if "interpretation" in item]
    assert len(interpreted) == 1
    assert interpreted[0]["interpretation_translations"] == {}
    # Seule la dernière version scorée est interprétée.
    assert "interpretation" not in payload["scores"][0]
    assert payload["scores"][1]["interpretation"].startswith(
        "Le projet gère ses finances⟦1⟧."
    )


def test_written_needs_a_paragraph_and_a_source() -> None:
    pillars = subjects(_layer_a(), _inputs(), RULES)
    (batch,) = build_batches(_inputs(), pillars, [A1], TEXTS, RULES)

    assert written("Court [E1].", {}, batch.excerpts, None, 900, 1500, 100) is None
    assert (
        written("Sans source." + PAD, {}, batch.excerpts, None, 900, 1500, 100) is None
    )
    text = written(
        "Le projet [E1]." + PAD,
        {"en": "Too short [E1].", "pt": None},
        batch.excerpts,
        None,
        900,
        1500,
        100,
    )
    assert text is not None and text.translations == {}
