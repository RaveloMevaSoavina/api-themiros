from datetime import UTC, datetime
from itertools import count
from uuid import UUID, uuid4

import pytest

from app.modules.runs.domain import (
    Match,
    Pillar,
    ProgramVersion,
    RunInputs,
    ScoringRules,
    Variable,
    compute_layer_a,
    corpus_confidence,
    evolution_label,
    pillar_confidence,
    pillar_score,
    select_evidences,
    variable_state,
    variable_value,
)

RULES = ScoringRules(strong_similarity=0.55, min_independent_sources=2)


def _uuid(prefix: int, index: int) -> UUID:
    return UUID(f"{prefix:08x}-0000-4000-8000-{index:012x}")


def _match(document: int, segment: int, similarity: float, variable: UUID) -> Match:
    return Match(_uuid(0xD0C, segment), _uuid(0xD0C0, document), variable, similarity)


VARIABLE = _uuid(0xA, 1)


def test_corpus_confidence_reproduces_the_specification_example() -> None:
    # §7.5 : 10 documents, segments convergents 3, 2, 4, 0, 1, 3, 0, 2, 1, 0.
    detections = [
        _match(document, document * 10 + segment, 0.6, VARIABLE)
        for document, segments in enumerate([3, 2, 4, 0, 1, 3, 0, 2, 1, 0])
        for segment in range(segments)
    ]
    assert corpus_confidence(detections, total_documents=10) == 1.0


def test_corpus_confidence_combines_partial_documents() -> None:
    detections = [
        _match(1, 1, 0.6, VARIABLE),
        _match(2, 2, 0.6, VARIABLE),
        _match(2, 3, 0.6, VARIABLE),
    ]
    # 1 − (1 − 1/3)(1 − 2/3)
    assert corpus_confidence(detections, total_documents=5) == 0.7778


def test_corpus_confidence_of_a_single_document_is_its_probability() -> None:
    assert corpus_confidence([_match(1, 1, 0.6, VARIABLE)], total_documents=1) == 0.3333


def test_corpus_confidence_counts_each_segment_once() -> None:
    same_segment = [_match(1, 1, 0.6, VARIABLE), _match(1, 1, 0.7, VARIABLE)]
    assert corpus_confidence(same_segment, total_documents=1) == 0.3333


def test_corpus_confidence_is_empty_without_documents() -> None:
    assert corpus_confidence([], total_documents=0) is None


def test_a_variable_without_detection_is_not_reported() -> None:
    assert variable_value([], RULES) is None
    assert variable_state(None) == "non_renseigne"


def test_strong_detections_in_two_documents_make_a_variable_present() -> None:
    detections = [_match(1, 1, 0.58, VARIABLE), _match(2, 2, 0.61, VARIABLE)]
    assert variable_value(detections, RULES) == 1.0
    assert variable_state(1.0) == "present"


@pytest.mark.parametrize(
    "detections",
    [
        # Un seul document, même avec plusieurs extraits forts.
        [_match(1, 1, 0.7, VARIABLE), _match(1, 2, 0.8, VARIABLE)],
        # Deux documents, dont un seul au-dessus du seuil fort.
        [_match(1, 1, 0.7, VARIABLE), _match(2, 2, 0.45, VARIABLE)],
        # Mentions faibles uniquement.
        [_match(1, 1, 0.42, VARIABLE), _match(2, 2, 0.44, VARIABLE)],
    ],
)
def test_detections_without_convergence_are_partial(detections) -> None:
    assert variable_value(detections, RULES) == 0.5
    assert variable_state(0.5) == "partiel"


def test_the_engine_never_concludes_absence() -> None:
    values = {
        variable_value(detections, RULES)
        for detections in ([], [_match(1, 1, 0.41, VARIABLE)])
    }
    assert 0 not in values


def test_pillar_score_is_the_weighted_mean_of_reported_variables() -> None:
    assert pillar_score([(2, 1.0), (1, 0.5), (3, None)]) == 83.33


def test_pillar_score_is_empty_without_reported_variable() -> None:
    assert pillar_score([(1, None), (2, None)]) is None
    assert pillar_score([]) is None


def test_unreported_variables_lower_the_pillar_confidence() -> None:
    assert pillar_confidence([0.9, None]) == 0.45
    assert pillar_confidence([]) == 0.0


@pytest.mark.parametrize(
    ("delta", "label"),
    [
        (5, "renforce"),
        (14, "renforce"),
        (4.99, "maintenu"),
        (-4.99, "maintenu"),
        (-5, "affaibli"),
    ],
)
def test_evolution_labels_follow_the_five_point_threshold(delta, label) -> None:
    assert evolution_label(delta) == label


def test_evidences_show_each_document_before_repeating_one() -> None:
    detections = [
        _match(1, 1, 0.9, VARIABLE),
        _match(1, 2, 0.85, VARIABLE),
        _match(2, 3, 0.5, VARIABLE),
    ]
    selected = select_evidences(detections, limit=2)
    assert [match.segment_id for match in selected] == [
        detections[0].segment_id,
        detections[2].segment_id,
    ]


def _inputs() -> RunInputs:
    """Deux versions ; un pilier à deux variables, un pilier sans détection.

    ProDoc 2022 : documents 1 et 2. Revue 2025 : documents 3 et 4.
    """
    budget, absorption, gender = _uuid(0xB, 1), _uuid(0xB, 2), _uuid(0xB, 3)
    v2022, v2025 = _uuid(0x5, 1), _uuid(0x5, 2)
    return RunInputs(
        state="ready",
        created_at=datetime.now(UTC),
        versions=(ProgramVersion(v2025, 2), ProgramVersion(v2022, 1)),
        document_versions={
            _uuid(0xD0C0, 1): frozenset({v2022}),
            _uuid(0xD0C0, 2): frozenset({v2022}),
            _uuid(0xD0C0, 3): frozenset({v2025}),
            _uuid(0xD0C0, 4): frozenset({v2025}),
        },
        unassigned_documents=0,
        pillars=(
            Pillar(
                _uuid(0xF, 1),
                (
                    Variable(budget, "has_budget", 2),
                    Variable(absorption, "has_absorption", 1),
                ),
            ),
            Pillar(_uuid(0xF, 2), (Variable(gender, "has_gender_strategy", 1),)),
        ),
        segments={"published": 10, "classified": 6, "low_confidence": 1},
        matches=(
            # 2022 : budget fort dans un seul document → partiel ; absorption
            # non renseignée.
            _match(1, 1, 0.7, budget),
            _match(1, 2, 0.6, budget),
            # 2025 : budget fort dans deux documents → présent ; absorption
            # faible → partielle.
            _match(3, 3, 0.6, budget),
            _match(4, 4, 0.58, budget),
            _match(4, 5, 0.45, absorption),
        ),
    )


def test_layer_a_scores_each_pillar_and_version() -> None:
    ids = count(1)
    results = compute_layer_a(_inputs(), RULES, new_id=lambda: _uuid(0xE, next(ids)))

    states = {
        (variable.variable_code, variable.program_version_id.int & 0xF): variable.state
        for variable in results.variables
    }
    assert states == {
        ("has_budget", 1): "partiel",
        ("has_absorption", 1): "non_renseigne",
        ("has_budget", 2): "present",
        ("has_absorption", 2): "partiel",
        ("has_gender_strategy", 1): "non_renseigne",
        ("has_gender_strategy", 2): "non_renseigne",
    }

    budget_2022, budget_2025, gender_2022, gender_2025 = results.scores
    # 2022 : seule la variable budget (poids 2, valeur 0,5) est renseignée.
    assert budget_2022.score == 50.0
    assert budget_2022.delta is None and budget_2022.evolution_label is None
    assert budget_2022.documents_count == 1
    # 2025 : (2 × 1 + 1 × 0,5) / 3.
    assert budget_2025.score == 83.33
    assert budget_2025.delta == 33.33
    assert budget_2025.evolution_label == "renforce"
    assert budget_2025.documents_count == 2
    # Budget 2025 : 1 − (1 − 1/3)², absorption : 1/3 ; moyenne.
    assert budget_2025.corpus_confidence == round((0.5556 + 0.3333) / 2, 4)
    assert gender_2022.score is None and gender_2025.score is None
    assert gender_2025.delta is None
    assert gender_2025.corpus_confidence == 0.0


def test_payload_links_evidences_to_their_variable() -> None:
    results = compute_layer_a(_inputs(), RULES)
    payload = results.payload(RULES)

    variable_ids = {item["id"] for item in payload["variables"]}
    assert len(payload["evidences"]) == 5
    assert {item["variable_id"] for item in payload["evidences"]} <= variable_ids
    assert payload["rules"]["engine_version"] == "layer-a/1.0.0"
    assert payload["rules"]["strong_similarity"] == 0.55
    assert results.summary()["variables"] == {
        "variables_computed": 6,
        "variables_non_renseigne": 3,
        "evidences": 5,
    }


def test_run_inputs_parse_the_frozen_corpus() -> None:
    version = uuid4()
    inputs = RunInputs.from_row(
        {
            "state": "ready",
            "created_at": "2026-10-20T15:15:42.252053+00:00",
            "versions": [{"id": str(version), "label": "2025", "order_index": 1}],
            "documents": [
                {
                    "document_id": str(_uuid(0xD0C0, 1)),
                    "program_version_id": str(version),
                },
                {"document_id": str(_uuid(0xD0C0, 2)), "program_version_id": None},
            ],
            "pillars": [
                {
                    "id": str(_uuid(0xF, 1)),
                    "weight": 100,
                    "variables": [
                        {"id": str(VARIABLE), "code": "has_budget", "weight": "1.5"}
                    ],
                }
            ],
            "segments": {"published": 4, "classified": 3, "low_confidence": 0},
            "matches": [
                {
                    "segment_id": str(_uuid(0xD0C, 1)),
                    "document_id": str(_uuid(0xD0C0, 1)),
                    "pillar_variable_id": str(VARIABLE),
                    "similarity": 0.61,
                }
            ],
        }
    )
    assert inputs.document_versions == {_uuid(0xD0C0, 1): frozenset({version})}
    assert inputs.unassigned_documents == 1
    assert inputs.pillars[0].variables[0].weight == 1.5
    assert inputs.matches[0].similarity == 0.61
