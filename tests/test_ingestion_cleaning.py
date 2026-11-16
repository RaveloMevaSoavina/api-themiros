from app.modules.ingestion.analysis.cleaning import (
    clean_pages,
    dehyphenate,
    margin_key,
)
from app.modules.ingestion.analysis.segmenter import PageText

BODY = "Le projet a construit {n} retenues collinaires dans la région."


def _report(pages: int) -> list[PageText]:
    return [
        PageText(
            number,
            "\n".join(
                [
                    "PROJET PAPRIKA – RAPPORT ANNUEL 2025",
                    BODY.format(n=number * 3),
                    f"Les comités de gestion se réunissent {number} fois par mois.",
                    f"Page {number} / {pages}",
                ]
            ),
            number * 1000,
        )
        for number in range(1, pages + 1)
    ]


def test_page_numbers_do_not_prevent_matching_margins() -> None:
    assert margin_key("Page 3 / 40") == margin_key("Page 12 / 40") == "#"
    assert margin_key("- 7 -") == "#"
    assert margin_key("Rapport annuel 2025 | 14") == margin_key(
        "Rapport annuel 2025 | 15"
    )


def test_repeated_headers_and_footers_are_removed_from_segments_only() -> None:
    pages = _report(4)
    cleaned = clean_pages(pages)
    for original, page in zip(pages, cleaned, strict=True):
        assert "RAPPORT ANNUEL" not in page.text
        assert "Page" not in page.text
        assert BODY.format(n=original.page_number * 3) in page.text
        assert f"réunissent {original.page_number} fois" in page.text
        # Le texte conservé en base n'est pas touché.
        assert "RAPPORT ANNUEL" in original.text


def test_short_documents_and_sheets_keep_their_margins() -> None:
    assert clean_pages(_report(2)) == _report(2)
    sheets = [
        PageText(number, "A1=Activité | B1=Montant\nA2=Irrigation", 0, f"S{number}")
        for number in range(1, 5)
    ]
    assert clean_pages(sheets) == sheets


def test_line_break_hyphens_are_joined() -> None:
    vocabulary = {"suivi-évaluation"}
    text = "la mise en œu-\nvre du suivi-\névaluation et des 2020-\n2025"
    assert dehyphenate(text, vocabulary) == (
        "la mise en œuvre du suivi-évaluation et des 2020-\n2025"
    )
    # Une suite en majuscule n'est pas une coupure de mot.
    assert dehyphenate("Nord-\nKivu", set()) == "Nord-\nKivu"
