from uuid import UUID

from app.modules.evidences.passages import clean, passage, sentence_spans
from app.modules.runs.citations import (
    Citations,
    Source,
    compact,
    cut,
    excerpt,
    fit,
    plain,
    references,
)

FIRST = UUID("0a000000-0000-4000-8000-000000000001")
SECOND = UUID("0a000000-0000-4000-8000-000000000002")

# Texte tel que l'extraction PDF le rend : retours à la ligne au milieu des
# phrases, césure, abréviation, puce.
PDF_TEXT = (
    "Le taux de décaissement atteint 60 % par an, selon M. Rakoto,\n"
    "chef de projet. Les responsables des institutions sont rempla-\n"
    "cés fréquemment.\n"
    "• Trois missions de suivi ont eu lieu\n"
    "• Le budget révisé est validé"
)


def _sentences(text: str) -> list[str]:
    return [clean(text[start:end]) for start, end in sentence_spans(text)]


def test_sentences_follow_punctuation_bullets_and_abbreviations() -> None:
    assert _sentences(PDF_TEXT) == [
        "Le taux de décaissement atteint 60 % par an, selon M. Rakoto, chef de projet.",
        "Les responsables des institutions sont remplacés fréquemment.",
        "• Trois missions de suivi ont eu lieu",
        "• Le budget révisé est validé",
    ]


def test_spans_point_into_the_raw_segment_text() -> None:
    spans = sentence_spans(PDF_TEXT)

    assert PDF_TEXT[slice(*spans[1])].startswith("Les responsables")
    assert passage(PDF_TEXT, [spans[1], spans[3]]) == (
        "Les responsables des institutions sont remplacés fréquemment. … "
        "• Le budget révisé est validé"
    )
    # Deux phrases qui se suivent forment un seul passage.
    assert " … " not in passage(PDF_TEXT, [spans[0], spans[1]])


def test_an_excerpt_sends_whole_sentences_within_the_limit() -> None:
    item = excerpt(FIRST, PDF_TEXT, 140)

    assert item.sentences == tuple(_sentences(PDF_TEXT)[:2])
    assert len(item.spans) == 2
    # Une première phrase plus longue que la limite est coupée.
    assert excerpt(FIRST, "mot " * 100, 40).sentences[0].endswith("…")


def _excerpts():
    return {
        "E1": excerpt(FIRST, PDF_TEXT, 600),
        "E2": excerpt(SECOND, "Le contexte politique est instable. Les élections ont "
                      "été reportées.", 600),
    }  # fmt: skip


def test_markers_become_numbered_sources() -> None:
    excerpts = _excerpts()
    citations = Citations(excerpts)

    text = citations.cite(
        "Le décaissement de 60 % par an [E1.1, E2] est élevé, malgré des "
        "remplacements fréquents [E1.2; e2.1] et [E9] (E1.7).",
    )

    assert text == (
        "Le décaissement de 60 % par an⟦1,2⟧ est élevé, malgré des "
        "remplacements fréquents⟦3,4⟧ et⟦5⟧."
    )
    spans = excerpts["E1"].spans
    assert citations.sources == [
        Source(FIRST, (spans[0],)),
        # L'extrait entier : ses phrases, d'un seul tenant.
        Source(SECOND, ((excerpts["E2"].spans[0][0], excerpts["E2"].spans[-1][1]),)),
        Source(FIRST, (spans[1],)),
        Source(SECOND, (excerpts["E2"].spans[0],)),
        # Une phrase inconnue vaut l'extrait entier.
        Source(FIRST, ((spans[0][0], spans[-1][1]),)),
    ]  # fmt: skip
    assert plain(text).endswith("fréquents et.")


def test_only_the_allowed_excerpts_are_cited() -> None:
    citations = Citations(_excerpts())

    text = citations.cite("Constat [E1, E2].", allowed={"E2"})

    assert text == "Constat⟦1⟧."
    assert [source.segment_id for source in citations.sources] == [SECOND]
    assert references("Constat [E1.2, E2] puis (E3).") == ["E1.2", "E2", "E3"]


def test_sentences_of_one_excerpt_in_one_marker_form_one_source() -> None:
    excerpts = _excerpts()
    citations = Citations(excerpts)

    assert citations.cite("Constat [E1.1, E1.2, E1.1].") == "Constat⟦1⟧."
    (source,) = citations.sources
    assert source.spans == ((excerpts["E1"].spans[0][0], excerpts["E1"].spans[1][1]),)


def test_sources_are_renumbered_by_first_citation_across_translations() -> None:
    sources = [
        Source(FIRST, ((0, 5),)),
        Source(SECOND, ((0, 5),)),
        Source(FIRST, ((6, 9),)),
    ]

    texts, kept = compact(["B⟦2⟧ puis A⟦2,1⟧.", "B⟦2⟧ then A⟦1⟧."], sources)

    assert texts == ["B⟦1⟧ puis A⟦1,2⟧.", "B⟦1⟧ then A⟦2⟧."]
    assert kept == [sources[1], sources[0]]


def test_a_cut_never_splits_a_marker_and_ignores_its_length() -> None:
    text = "Le budget est exécuté⟦1,2⟧ et le suivi reste manuel⟦3⟧."

    assert cut(text, 100) == text
    shortened = cut(text, 30)
    assert shortened == "Le budget est exécuté⟦1,2⟧ et le…"
    assert fit("x" * 20 + "⟦1⟧", 21) == "x" * 20
