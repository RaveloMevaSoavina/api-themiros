"""Surlignage d'une preuve dans son PDF source.

L'extrait d'une preuve est le texte de son segment, tiré de la page citée mais
qui peut déborder sur la précédente (chevauchement entre segments) ou la
suivante. On aligne ses mots, normalisés, sur ceux de ces pages : l'alignement
tolère césures, ligatures et retours à la ligne, que la recherche exacte d'une
chaîne ne passe pas. Une page scannée n'a pas de couche texte : rien ne s'y
aligne, le PDF est rendu tel quel.

Phrases citées (`spans`, positions dans l'extrait) : l'extrait entier est
aligné, ce qui ancre une phrase courte au bon endroit de la page, puis
seuls ses mots dans les phrases citées sont surlignés. Des phrases mal
retrouvées laissent surligné l'extrait entier.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher

import pymupdf

from app.modules.evidences.passages import merge
from app.modules.ingestion.analysis.text import normalize

# Plus court passage aligné retenu : en deçà, des mots isolés communs
# (« de la mise en ») seraient surlignés au hasard de la page.
MIN_RUN = 4
# Part de l'extrait qui doit se retrouver sur la page pour la surligner.
MIN_COVERAGE = 0.5
# Pages voisines de la page citée sur lesquelles l'extrait peut déborder.
NEIGHBOURS = 1


@dataclass(frozen=True, slots=True)
class HighlightedPdf:
    content: bytes
    page: int | None
    highlighted: bool


@dataclass(frozen=True, slots=True)
class _Token:
    text: str
    page: int
    rect: pymupdf.Rect
    line: tuple[int, int]


def _tokens(page: pymupdf.Page) -> list[_Token]:
    tokens = []
    for x0, y0, x1, y1, word, block, line, _ in page.get_text("words"):
        rect = pymupdf.Rect(x0, y0, x1, y1)
        for text in normalize(word).split():
            tokens.append(_Token(text, page.number, rect, (block, line)))
    return tokens


_WORD = re.compile(r"\S+")


def _excerpt_tokens(excerpt: str) -> list[tuple[str, int]]:
    """Mots normalisés de l'extrait, avec leur position dans le texte."""
    return [
        (text, word.start())
        for word in _WORD.finditer(excerpt)
        for text in normalize(word.group()).split()
    ]


def _matches(
    tokens: list[_Token], excerpt: list[tuple[str, int]]
) -> list[tuple[int, _Token]]:
    """Mots de la page alignés sur l'extrait, avec la position du mot de
    l'extrait auquel chacun répond."""
    matcher = SequenceMatcher(
        None,
        [token.text for token in tokens],
        [text for text, _ in excerpt],
        autojunk=False,
    )
    run = min(MIN_RUN, len(excerpt))
    matched = [
        (excerpt[block.b + offset][1], tokens[block.a + offset])
        for block in matcher.get_matching_blocks()
        if block.size >= run
        for offset in range(block.size)
    ]
    return matched if len(matched) >= MIN_COVERAGE * len(excerpt) else []


def _within(
    matched: list[tuple[int, _Token]],
    excerpt: list[tuple[str, int]],
    spans: list[tuple[int, int]],
) -> list[_Token]:
    """Mots alignés qui tombent dans les phrases citées, ou `[]` si une trop
    faible part de ces phrases a été retrouvée."""

    def inside(position: int) -> bool:
        return any(start <= position < end for start, end in spans)

    expected = sum(inside(position) for _, position in excerpt)
    kept = [token for position, token in matched if inside(position)]
    return kept if expected and len(kept) >= MIN_COVERAGE * expected else []


def _lines(tokens: list[_Token]) -> dict[int, list[pymupdf.Rect]]:
    """Un rectangle par ligne de texte surlignée, groupés par page."""
    lines: dict[tuple[int, tuple[int, int]], pymupdf.Rect] = {}
    for token in tokens:
        key = (token.page, token.line)
        lines[key] = lines[key] | token.rect if key in lines else +token.rect
    pages: dict[int, list[pymupdf.Rect]] = {}
    for (page, _), rect in lines.items():
        pages.setdefault(page, []).append(rect)
    return pages


def highlight(
    content: bytes,
    excerpt: str,
    page_number: int | None,
    spans: Sequence[Sequence[int]] | None = None,
) -> HighlightedPdf:
    """Le PDF avec l'extrait surligné (ses phrases citées seules, quand
    `spans` les désigne), ouvert sur la page où le passage commence."""
    document = pymupdf.open("pdf", content)
    if document.needs_pass or not page_number or page_number > document.page_count:
        return HighlightedPdf(content, page_number, False)

    cited = page_number - 1
    first = max(0, cited - NEIGHBOURS)
    last = min(document.page_count - 1, cited + NEIGHBOURS)
    tokens = [
        token for index in range(first, last + 1) for token in _tokens(document[index])
    ]
    words = _excerpt_tokens(excerpt)
    aligned = _matches(tokens, words)
    if not aligned:
        return HighlightedPdf(content, page_number, False)
    matched = [token for _, token in aligned]
    if spans:
        matched = _within(aligned, words, merge(spans)) or matched

    for index, rects in _lines(matched).items():
        # La page doit vivre autant que son annotation.
        page = document[index]
        annotation = page.add_highlight_annot(quads=rects)
        annotation.set_info(title="Themiros")
        annotation.update()

    start = matched[0]
    page = document[start.page]
    # Ouverture sur le haut du passage, dans les lecteurs qui honorent
    # /OpenAction ; le dashboard passe aussi `#page=`.
    top = (pymupdf.Point(0, max(0, start.rect.y0 - 24)) * ~page.transformation_matrix).y
    document.xref_set_key(
        document.pdf_catalog(), "OpenAction", f"[{page.xref} 0 R /FitH {top:g}]"
    )
    return HighlightedPdf(
        document.tobytes(garbage=1, deflate=True), start.page + 1, True
    )
