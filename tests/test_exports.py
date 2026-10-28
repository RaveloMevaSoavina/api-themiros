import csv
import io
from datetime import UTC, datetime
from uuid import UUID

import pymupdf
import pytest
from docx import Document
from fastapi.testclient import TestClient
from openpyxl import load_workbook

from app.api.dependencies import CurrentUser, get_current_user
from app.core.errors import ApiError
from app.main import app
from app.modules.exports.renderers import (
    filename,
    render_csv,
    render_docx,
    render_pdf,
    render_xlsx,
)
from app.modules.exports.report import RunRows, build_report
from app.modules.exports.router import get_export_service
from app.modules.exports.schemas import ExportRequest
from app.modules.exports.service import ExportService

USER = CurrentUser(
    id=UUID("11111111-1111-4111-8111-111111111111"),
    email="analyste@example.com",
    token="user-token",
)
RUN_ID = "22222222-2222-4222-8222-222222222222"
EXPORTED_AT = datetime(2026, 10, 28, 9, 30, tzinfo=UTC)

PILLAR = {
    "name": "Gouvernance",
    "description": "Pilotage du programme.",
    "source_locale": "fr",
    "pillar_translations": [
        {"locale": "en", "name": "Governance", "description": "Steering."}
    ],
}
VARIABLES = {
    "pillar_variables": [
        {
            "code": "V1",
            "label": "Comité de pilotage",
            "description": None,
            "source_locale": "fr",
            "pillar_variable_translations": [],
        },
        {
            "code": "V2",
            "label": "Rapports de suivi",
            "description": None,
            "source_locale": "fr",
            "pillar_variable_translations": [],
        },
    ]
}


def run_row(**overrides) -> dict:
    return {
        "id": RUN_ID,
        "workspace_id": "33333333-3333-4333-8333-333333333333",
        "status": "completed",
        "engine_version": "2.4.0",
        "corpus_size": 7,
        "corpus_seuil": "exploratoire",
        "created_at": "2026-10-27T08:00:00+00:00",
        "completed_at": "2026-10-27T08:05:00+00:00",
        "validated_at": None,
        "validated_by_name": None,
        "workspaces": {"name": "Programme Eau Sahel"},
        **overrides,
    }


def rows(**overrides) -> RunRows:
    values = {
        "run": run_row(),
        "steps": [
            {"step_name": "layer_b", "status": "completed"},
            {"step_name": "layer_c", "status": "completed"},
        ],
        "scores": [
            {
                "id": "s1",
                "pillar_id": "p1",
                "score": "62.50",
                "delta": None,
                "evolution_label": None,
                "corpus_confidence": "0.58",
                "documents_count": 5,
                "pillars": PILLAR,
                "program_versions": {"label": "Phase 1", "order_index": 1},
            },
            {
                "id": "s2",
                "pillar_id": "p2",
                "score": None,
                "delta": None,
                "evolution_label": None,
                "corpus_confidence": "0.20",
                "documents_count": 0,
                "pillars": {**PILLAR, "name": "Équité", "pillar_translations": []},
                "program_versions": {"label": "Phase 1", "order_index": 1},
            },
        ],
        "variables": [
            {
                "id": "iv1",
                "pillar_id": "p1",
                "variable_code": "V1",
                "value": "1.00",
                "state": "present",
                "documents_count": 3,
                "corpus_confidence": "0.70",
                "program_versions": {"label": "Phase 1", "order_index": 1},
                "pillars": VARIABLES,
            },
            {
                "id": "iv2",
                "pillar_id": "p1",
                "variable_code": "V2",
                "value": None,
                "state": "non_renseigne",
                "documents_count": 0,
                "corpus_confidence": None,
                "program_versions": {"label": "Phase 1", "order_index": 1},
                "pillars": VARIABLES,
            },
        ],
        "notes": [
            {
                "id": "n1",
                "criterion_id": "c1",
                "note": "2.50",
                "status": "conclu",
                "documented_sub_questions": 3,
                "total_sub_questions": 4,
                "justification": "Le comité se réunit chaque trimestre.",
                "ambiguity_flag": False,
                "criteria": {"code": "EFF", "name": "Efficacité"},
            },
            {
                "id": "n2",
                "criterion_id": "c2",
                "note": None,
                "status": "non_conclu",
                "documented_sub_questions": 1,
                "total_sub_questions": 4,
                "justification": None,
                "ambiguity_flag": False,
                "criteria": {"code": "DUR", "name": "Durabilité"},
            },
        ],
        "alerts": [
            {
                "id": "a1",
                "rule_code": "ALERT-01",
                "alert_type": "documentaire_sans_credibilite",
                "severity": "majeure",
                "message": "Score élevé, note faible.",
                "qualified": False,
                "status": "a_instruire",
                "instruction_comment": None,
                "pillar_id": "p1",
                "criterion_id": "c1",
                "pillars": PILLAR,
                "criteria": {"name": "Efficacité"},
                "alert_evidences": [{"evidence_id": "e2"}],
            }
        ],
        "evidences": [
            {
                "id": "e1",
                "document_id": "d1",
                "page_number": 4,
                "section_title": "Gouvernance",
                "excerpt": "Le comité de pilotage s'est réuni   quatre fois.",
                "pillar_id": "p1",
                "criterion_id": None,
                "variable_code": "V1",
                "documents": {"original_filename": "rapport-annuel.pdf"},
            },
            {
                "id": "e2",
                "document_id": "d2",
                "page_number": None,
                "section_title": None,
                "excerpt": "=SOMME(A1) les résultats restent partiels.",
                "pillar_id": None,
                "criterion_id": "c1",
                "variable_code": None,
                "documents": {"original_filename": "evaluation; mi-parcours.docx"},
            },
        ],
        "synthesis": {
            "conclusion": "Premier paragraphe.\n\nSecond paragraphe.",
            "limits": {"not_concluded": 1, "criteria": 2, "confidence": 0.39},
            "status": "completed",
        },
        "items": [
            {
                "id": "i1",
                "section": "a_verifier",
                "position": 1,
                "text": "Vérifier l'écart sur l'efficacité.",
                "pillar_id": None,
                "criterion_id": None,
                "alert_id": "a1",
                "synthesis_item_evidences": [{"evidence_id": "e2"}],
            }
        ],
        "actions": [
            {
                "id": "ac1",
                "position": 1,
                "origin_type": "alert",
                "pillar_id": None,
                "criterion_id": None,
                "alert_id": "a1",
                "run_action_projections": [
                    {
                        "persona": "pmu",
                        "text": "Documenter le suivi du comité.",
                        "justification": "l'alerte reste à instruire",
                    },
                    {
                        "persona": "decideur",
                        "text": "Exiger un rapport de suivi.",
                        "justification": "le suivi n'est pas démontré",
                    },
                ],
                "run_action_evidences": [{"evidence_id": "e1"}],
            }
        ],
        "corpus_changed": True,
    }
    return RunRows(**{**values, **overrides})


def report(locale: str = "fr", persona: str = "pmu", **overrides):
    return build_report(
        rows(**overrides),
        locale=locale,
        persona=persona,
        exported_by=USER.email,
        exported_at=EXPORTED_AT,
    )


def test_limits_always_lead_and_list_every_gap() -> None:
    limits = report().limits

    assert limits[0].startswith("Niveau de masse critique : exploratoire")
    assert "Critères non conclus : 1 sur 2." in limits
    assert "Confiance corpus moyenne : 39 %." in limits
    assert any("Équité" in line for line in limits)
    assert "Variables non renseignées : 1 sur 2." in limits
    assert any("non qualifiées par l'IA : 1" in line for line in limits)
    assert any("Le corpus a changé" in line for line in limits)
    assert "Ce run n'a pas été validé humainement." in limits


def test_missing_layers_and_restitution_are_stated() -> None:
    limits = report(
        steps=[{"step_name": "layer_c", "status": "skipped"}], synthesis=None
    ).limits

    assert any("couche C" in line for line in limits)
    assert any("synthèse et le plan d'action n'ont pas" in line for line in limits)


def test_evidences_become_numbered_notes_in_citation_order() -> None:
    built = report()

    assert [note.number for note in built.notes] == [1, 2]
    assert built.pillars[0].variables[0].notes == [1]
    assert [criterion.code for criterion in built.criteria] == ["DUR", "EFF"]
    assert built.criteria[1].notes == [2]
    assert built.alerts[0].notes == [2]
    assert built.actions[0].notes == [1]
    assert built.notes[0].excerpt == "Le comité de pilotage s'est réuni quatre fois."
    assert built.notes[0].subject == "Gouvernance · Comité de pilotage"


def test_report_follows_locale_and_persona() -> None:
    built = report(locale="en", persona="decideur")

    assert built.pillars[0].name == "Governance"
    assert [action.text for action in built.actions] == ["Exiger un rapport de suivi."]
    assert built.validation == "Not validated"


def test_pdf_opens_with_exploratory_banner_first() -> None:
    document = pymupdf.open("pdf", render_pdf(report()))
    first_page = document[0].get_text()

    assert "Run exploratoire : résultats indicatifs" in first_page
    assert "Limites du corpus" in first_page
    text = "".join(page.get_text() for page in document)
    for heading in (
        "Comparatif général",
        "couche A",
        "couche B",
        "couche C",
        "Synthèse",
        "Notes : références des preuves",
        "rapport-annuel.pdf",
        "2.4.0",
    ):
        assert heading in text


def test_validated_run_without_banner() -> None:
    built = report(
        run=run_row(
            corpus_seuil="standard",
            validated_at="2026-10-27T10:00:00+00:00",
            validated_by_name="Awa Diop",
        )
    )
    text = pymupdf.open("pdf", render_pdf(built))[0].get_text()

    assert "Run exploratoire" not in text
    assert "Validé par Awa Diop le 27/10/2026" in text


def test_docx_keeps_sections_and_notes() -> None:
    document = Document(io.BytesIO(render_docx(report())))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    tables = "\n".join(
        cell.text
        for table in document.tables
        for row in table.rows
        for cell in row.cells
    )

    assert "Run exploratoire : résultats indicatifs" in tables
    assert "Critères : notes évaluatives (couche B)" in text
    assert "Plan d'action : lecture Chef de projet / PMU" in text
    assert "[1] rapport-annuel.pdf · p. 4 · Gouvernance" in text


def test_xlsx_has_one_sheet_per_screen() -> None:
    workbook = load_workbook(io.BytesIO(render_xlsx(report())))

    assert workbook.sheetnames == [
        "Résumé",
        "Comparatif",
        "Variables",
        "Critères",
        "Alertes",
        "Synthèse",
        "Plan d'action",
        "Preuves",
    ]
    assert workbook["Résumé"]["A2"].value == "Run exploratoire : résultats indicatifs"
    assert workbook["Comparatif"]["C2"].value == 62.5
    assert workbook["Critères"]["D2"].value == "Non conclu"


def test_csv_neutralises_formulas_and_keeps_layers() -> None:
    content = render_csv(report())

    assert content.startswith("﻿".encode())
    lines = list(csv.reader(io.StringIO(content.decode("utf-8-sig")), delimiter=";"))
    assert lines[0][0] == "Couche"
    assert {line[0] for line in lines[1:]} >= {"Run", "Limites", "A", "B", "C"}
    excerpts = [line[6] for line in lines if line[0] == "Preuves"]
    assert "'=SOMME(A1) les résultats restent partiels." in excerpts
    assert ["A", "Pilier", "Gouvernance", "62,5"] == next(
        line[:4] for line in lines if line[0] == "A"
    )


def test_filename_is_ascii() -> None:
    assert filename(report(), "pdf") == "themiros-programme-eau-sahel-27-10-2026.pdf"


class FakeRepository:
    def __init__(self, run=None, failing: bool = False) -> None:
        self._run = run
        self._failing = failing
        self.recorded: list[tuple[str, dict]] = []

    async def run(self, run_id, *, token):
        assert token == USER.token
        return self._run

    async def rows(self, run, *, token):
        if self._failing:
            raise ApiError(status_code=503, code="DATABASE_UNAVAILABLE", detail="x")
        return rows(run=run)

    async def record(self, run, *, token, user_id, operation, metadata):
        assert user_id == USER.id
        self.recorded.append((operation, metadata))


async def test_export_is_journaled_with_format_and_run() -> None:
    repository = FakeRepository(run_row())
    exported = await ExportService(repository).export(
        UUID(RUN_ID), "xlsx", ExportRequest(persona="analyste"), USER
    )

    assert exported.filename.endswith(".xlsx")
    [(operation, metadata)] = repository.recorded
    assert operation == "report_exported"
    assert metadata["format"] == "xlsx"
    assert metadata["persona"] == "analyste"
    assert metadata["exploratory"] is True
    assert metadata["notes"] == 2


async def test_failed_export_is_journaled_too() -> None:
    repository = FakeRepository(run_row(), failing=True)
    with pytest.raises(ApiError):
        await ExportService(repository).export(
            UUID(RUN_ID), "pdf", ExportRequest(), USER
        )

    [(operation, metadata)] = repository.recorded
    assert operation == "report_export_failed"
    assert metadata["error"] == "DATABASE_UNAVAILABLE"


@pytest.mark.parametrize(
    ("run", "status", "code"),
    [
        (None, 404, "RUN_NOT_FOUND"),
        (run_row(status="running"), 409, "RUN_NOT_COMPLETED"),
    ],
)
def test_only_visible_completed_runs_are_exported(run, status, code) -> None:
    repository = FakeRepository(run)
    app.dependency_overrides[get_current_user] = lambda: USER
    app.dependency_overrides[get_export_service] = lambda: ExportService(repository)
    try:
        with TestClient(app) as client:
            response = client.post(f"/api/v1/exports/{RUN_ID}/pdf", json={})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == status
    assert response.json()["code"] == code
    assert repository.recorded == []


def test_export_endpoint_returns_the_file() -> None:
    repository = FakeRepository(run_row())
    app.dependency_overrides[get_current_user] = lambda: USER
    app.dependency_overrides[get_export_service] = lambda: ExportService(repository)
    try:
        with TestClient(app) as client:
            response = client.post(
                f"/api/v1/exports/{RUN_ID}/csv", json={"locale": "en"}
            )
            unknown = client.post(f"/api/v1/exports/{RUN_ID}/odt", json={})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert (
        'filename="themiros-programme-eau-sahel-2026-10-27.csv"'
        in (response.headers["content-disposition"])
    )
    assert response.content.decode("utf-8-sig").startswith("Layer;")
    assert unknown.status_code == 422
