from uuid import UUID

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import CurrentUser, get_current_user
from app.core.errors import ApiError
from app.main import app
from app.modules.evidences.highlight import highlight
from app.modules.evidences.router import get_evidence_service
from app.modules.evidences.service import EvidenceService

USER = CurrentUser(
    id=UUID("11111111-1111-4111-8111-111111111111"),
    email="analyste@example.com",
    token="user-token",
)
EVIDENCE_ID = "44444444-4444-4444-8444-444444444444"

FILLER = (
    "Le rapport annuel présente le contexte du programme et les "
    "partenaires mobilisés dans les trois régions d'intervention."
)
PASSAGE = (
    "Le comité de pilotage s'est réuni quatre fois en 2025 et a validé "
    "le plan de travail révisé, y compris le budget des activités de "
    "terrain et le calendrier des missions de suivi."
)


def pdf(*pages: list[str]) -> bytes:
    """Un PDF dont chaque page empile des paragraphes, repliés à la ligne."""
    document = pymupdf.open()
    for paragraphs in pages:
        page = document.new_page()
        page.insert_textbox(
            pymupdf.Rect(72, 72, 360, 770), "\n\n".join(paragraphs), fontsize=11
        )
    return document.tobytes()


def highlights(content: bytes) -> dict[int, str]:
    """Texte surligné, page par page (numérotées à partir de 1)."""
    document = pymupdf.open("pdf", content)
    found = {}
    for page in document:
        words = page.get_text("words")
        for annotation in page.annots():
            assert annotation.type[1] == "Highlight"
            area = annotation.rect
            found[page.number + 1] = " ".join(
                word[4]
                for word in words
                if pymupdf.Rect(word[:4]).intersects(area)
                and pymupdf.Rect(word[:4]).get_area() > 0
            )
    return found


def test_the_cited_passage_is_highlighted_on_its_page() -> None:
    content = pdf([FILLER], [FILLER, PASSAGE, FILLER])

    result = highlight(content, PASSAGE, 2)

    assert result.highlighted
    assert result.page == 2
    [(page, text)] = highlights(result.content).items()
    assert page == 2
    assert "comité de pilotage" in text
    assert "missions de suivi" in text
    assert "partenaires" not in text
    catalog = pymupdf.open("pdf", result.content)
    kind, value = catalog.xref_get_key(catalog.pdf_catalog(), "OpenAction")
    assert kind == "array" and "/FitH" in value


def test_line_breaks_and_spacing_do_not_prevent_the_match() -> None:
    content = pdf([FILLER, PASSAGE])
    excerpt = "  ".join(PASSAGE.upper().split())

    assert highlight(content, excerpt, 1).highlighted


def test_an_excerpt_overflowing_from_the_previous_page_opens_there() -> None:
    head, tail = PASSAGE[:70], PASSAGE[70:]
    content = pdf([FILLER, head], [tail, FILLER])

    result = highlight(content, PASSAGE, 2)

    assert result.highlighted
    assert result.page == 1
    assert set(highlights(result.content)) == {1, 2}


def test_a_page_without_text_layer_is_returned_untouched() -> None:
    content = pdf([], [FILLER])

    result = highlight(content, PASSAGE, 1)

    assert not result.highlighted
    assert result.content == content
    assert result.page == 1


@pytest.mark.parametrize("page", [None, 9])
def test_an_unknown_page_is_not_searched(page) -> None:
    content = pdf([PASSAGE])

    result = highlight(content, PASSAGE, page)

    assert not result.highlighted
    assert result.content == content


class FakeRepository:
    def __init__(self, evidence, content: bytes = b"") -> None:
        self._evidence = evidence
        self._content = content
        self.downloaded: list[str] = []

    async def evidence(self, evidence_id, *, token):
        assert token == USER.token
        return self._evidence

    async def download(self, storage_path):
        self.downloaded.append(storage_path)
        return self._content


def evidence_row(**document) -> dict:
    return {
        "id": EVIDENCE_ID,
        "page_number": 1,
        "excerpt": PASSAGE,
        "documents": {
            "original_filename": "Rapport annuel été 2025.pdf",
            "storage_path": "ws/doc/rapport.pdf",
            "mime_type": "application/pdf",
            **document,
        },
    }


@pytest.mark.parametrize(
    ("evidence", "status", "code"),
    [
        (None, 404, "EVIDENCE_NOT_FOUND"),
        ({**evidence_row(), "documents": None}, 404, "EVIDENCE_DOCUMENT_NOT_FOUND"),
        (
            evidence_row(mime_type="application/msword"),
            409,
            "EVIDENCE_DOCUMENT_NOT_PDF",
        ),
    ],
)
async def test_only_a_visible_pdf_is_downloaded(evidence, status, code) -> None:
    repository = FakeRepository(evidence)

    with pytest.raises(ApiError) as error:
        await EvidenceService(repository).document(UUID(EVIDENCE_ID), USER)

    assert (error.value.status_code, error.value.code) == (status, code)
    assert repository.downloaded == []


async def test_an_unreadable_pdf_is_served_as_is() -> None:
    repository = FakeRepository(evidence_row(), content=b"%PDF-1.7 broken")

    document = await EvidenceService(repository).document(UUID(EVIDENCE_ID), USER)

    assert document.content == b"%PDF-1.7 broken"
    assert not document.highlighted


def test_endpoint_serves_the_highlighted_pdf_inline() -> None:
    repository = FakeRepository(evidence_row(), content=pdf([FILLER, PASSAGE]))
    app.dependency_overrides[get_current_user] = lambda: USER
    app.dependency_overrides[get_evidence_service] = lambda: EvidenceService(repository)
    try:
        with TestClient(app) as client:
            response = client.get(f"/api/v1/evidences/{EVIDENCE_ID}/document")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.headers["x-evidence-highlighted"] == "true"
    assert response.headers["x-evidence-page"] == "1"
    assert response.headers["content-disposition"] == (
        "inline; filename*=UTF-8''Rapport%20annuel%20%C3%A9t%C3%A9%202025.pdf"
    )
    assert repository.downloaded == ["ws/doc/rapport.pdf"]
    assert highlights(response.content)
