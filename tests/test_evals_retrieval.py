from uuid import uuid4

import pytest

from app.modules.evals.retrieval import (
    DatasetError,
    Variant,
    annotation_sheet,
    baseline,
    evaluate,
    normalize,
    parse_dataset,
    queries,
    summarize,
)
from app.modules.runs.layer_b import LayerBRules

REPORT, AUDIT = uuid4(), uuid4()
RULES = LayerBRules(
    min_similarity=0.4,
    lexical_min_similarity=0.3,
    max_per_document=3,
    top_k=8,
    excerpts_per_question=2,
)
PILOTAGE = "Le comité de pilotage se réunit chaque trimestre et valide le plan."
BUDGET = "Les dépenses de 2025 atteignent 60 % du budget prévu."
GENERIC = "Le projet poursuit ses activités dans les trois régions."


def _raw(**overrides):
    raw = {
        "name": "Projet test",
        "documents": [str(REPORT), str(AUDIT)],
        "questions": [
            {
                "id": "G1",
                "criterion": "Gouvernance",
                "text": "Le projet a-t-il un comité de pilotage actif ?",
                "expected_evidence": "Comptes rendus du comité",
                "relevant": [
                    {"document": str(REPORT), "quote": "comité de pilotage se réunit"}
                ],
            },
            {
                "id": "N1",
                "criterion": "Durabilité",
                "text": "Une stratégie de sortie est-elle prévue ?",
                "relevant": [],
            },
        ],
    }
    raw.update(overrides)
    return raw


class FakeEmbedder:
    model = "m"

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def embed(self, texts):
        self.texts.extend(texts)
        return [[0.1, 0.2] for _ in texts]


class FakeGateway:
    """Le comité de pilotage n'est trouvé que par ses mots, sous le seuil
    vectoriel ; la question sans réponse ne trouve que du bruit."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    async def service_rpc(self, function_name, *, payload, request_timeout=30):
        assert function_name == "segment_matches"
        self.payloads.append(payload)
        rows = []
        for query in payload["p_queries"]:
            hybrid = "text" in query
            question = query["sub_question_id"]
            if "comité" in query.get("text", "") or len(rows) == 0:
                rows += [
                    self._row(question, REPORT, GENERIC, 0.55, None, 0.0164),
                    self._row(
                        question,
                        REPORT,
                        PILOTAGE,
                        0.35,
                        1 if hybrid else None,
                        0.032 if hybrid else 0.0161,
                    ),
                ]
            else:
                rows.append(self._row(question, AUDIT, BUDGET, 0.36, None, 0.0164))
        return rows

    async def service_select(self, path, *, params=None):
        return [
            {"id": str(REPORT), "original_filename": "rapport-2025.pdf"},
            {"id": str(AUDIT), "original_filename": "audit.pdf"},
        ]

    @staticmethod
    def _row(question, document, text, similarity, lexical_rank, score):
        return {
            "sub_question_id": question,
            "segment_id": str(uuid4()),
            "document_id": str(document),
            "similarity": similarity,
            "lexical_rank": lexical_rank,
            "score": score,
            "text": text,
        }


def test_dataset_is_validated() -> None:
    dataset = parse_dataset(_raw())
    assert [question.key for question in dataset.questions] == ["G1", "N1"]
    assert dataset.questions[1].relevant == ()

    outside = _raw()
    outside["questions"][0]["relevant"][0]["document"] = str(uuid4())
    with pytest.raises(DatasetError, match="hors du corpus"):
        parse_dataset(outside)
    short = _raw()
    short["questions"][0]["relevant"][0]["quote"] = "comité"
    with pytest.raises(DatasetError, match="trop courte"):
        parse_dataset(short)
    twice = _raw()
    twice["questions"][1]["id"] = "G1"
    with pytest.raises(DatasetError, match="double"):
        parse_dataset(twice)
    with pytest.raises(DatasetError, match="aucun document"):
        parse_dataset(_raw(documents=[]))


def test_quotes_match_whatever_case_spacing_or_soft_hyphens() -> None:
    assert normalize("Comité  de\npilo­tage") == "comité de pilotage"
    assert normalize("ﬁnancement") == "financement"


def test_queries_are_built_like_the_engine() -> None:
    dataset = parse_dataset(_raw())
    current = dict(queries(dataset, Variant("courant", RULES)))
    former = dict(queries(dataset, baseline(RULES)))
    assert current["G1"] == (
        "Gouvernance : Le projet a-t-il un comité de pilotage actif ?\n"
        "Comptes rendus du comité"
    )
    assert former["G1"] == (
        "Gouvernance : Le projet a-t-il un comité de pilotage actif ?"
    )


async def test_hybrid_search_is_measured_against_the_former_one() -> None:
    dataset = parse_dataset(_raw())
    gateway = FakeGateway()

    hybrid, hybrid_results = await evaluate(
        gateway, FakeEmbedder(), dataset, Variant("courant", RULES)
    )
    former, _ = await evaluate(gateway, FakeEmbedder(), dataset, baseline(RULES))

    # Le texte n'est envoyé que pour la recherche hybride.
    assert all("text" in query for query in gateway.payloads[0]["p_queries"])
    assert not any("text" in query for query in gateway.payloads[1]["p_queries"])
    assert gateway.payloads[0]["p_document_ids"] == [str(REPORT), str(AUDIT)]
    # Le comité de pilotage, sous le seuil vectoriel, n'est trouvé que par ses
    # mots, et passe alors en tête.
    assert hybrid["rappel"] == 1.0 and former["rappel"] == 0.0
    assert hybrid["mrr"] == 1.0
    assert hybrid_results[0].first_relevant_rank == 1
    # La question sans réponse reste vide : le budget est sous le seuil.
    assert hybrid["sans_reponse_vides"] == 1.0


def test_summary_of_answered_and_unanswered_questions() -> None:
    from app.modules.evals.retrieval import QuestionResult

    summary = summarize(
        [
            QuestionResult("A", 2, 1, 2, 1, 2),
            QuestionResult("B", 2, 0, 1, 0, None),
            QuestionResult("C", 0, 0, 0, 0, None),
            QuestionResult("D", 1, 0, 0, 0, None),
        ]
    )
    assert summary["rappel"] == 0.25
    assert summary["trouvees"] == 0.5
    assert summary["mrr"] == 0.25
    assert summary["precision"] == round((0.5 + 0 + 0) / 3, 3)
    assert summary["sans_reponse_vides"] == 0.5
    assert summary["extraits"] == 1.25


async def test_annotation_sheet_lists_candidates_and_marks_known_passages() -> None:
    sheet = await annotation_sheet(
        FakeGateway(),
        FakeEmbedder(),
        parse_dataset(_raw()),
        Variant("courant", RULES),
    )
    assert "## G1 — Gouvernance" in sheet
    assert "Preuves attendues : Comptes rendus du comité" in sheet
    assert f"✓ rapport-2025.pdf (`{REPORT}`)" in sheet
    assert "mots n° 1" in sheet
