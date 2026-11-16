"""Nettoyage du texte avant découpage (spec §4.4, post-traitement).

Le texte des pages reste conservé intact (RG-4.6) : seuls les segments
indexés sont débarrassés des en-têtes et pieds de page répétés, et des mots
coupés en fin de ligne.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace

from app.modules.ingestion.analysis.segmenter import PageText

# En-têtes et pieds de page : cherchés dans les trois premières et les trois
# dernières lignes, retenus s'ils reviennent sur au moins la moitié des pages
# (et au moins trois).
MARGIN_LINES = 3
MARGIN_MIN_PAGES = 3
MARGIN_PAGE_SHARE = 0.5
MARGIN_MAX_CHARS = 150

_SPACES = re.compile(r"\s+")
# Numéro de page en tête ou en fin de ligne : « 12 », « - 12 - »,
# « Page 3/40 », « p. 3 sur 40 ».
_PAGE_NUMBER = r"[-–—|·•\s]*(?:page|p\.)?\s*\d+(?:\s*(?:/|sur|of|de)\s*\d+)?[-–—|·•\s]*"
_LEADING_PAGE_NUMBER = re.compile(rf"^{_PAGE_NUMBER}(?=\D|$)", re.IGNORECASE)
_TRAILING_PAGE_NUMBER = re.compile(
    rf"(?<=\D){_PAGE_NUMBER}$|^{_PAGE_NUMBER}$", re.IGNORECASE
)
# « œu-\nvre » : mot coupé en fin de ligne, suite en minuscule.
_HYPHEN_BREAK = re.compile(r"(\w+)-[ \t]*\n[ \t]*(\w+)")
_WORD = re.compile(r"\w+(?:-\w+)*")


def clean_pages(pages: Sequence[PageText]) -> list[PageText]:
    top, bottom = repeated_margins(pages)
    vocabulary = {word.lower() for page in pages for word in _WORD.findall(page.text)}
    return [
        replace(
            page,
            text=dehyphenate(
                strip_margins(page.text, top, bottom)
                if page.section_title is None
                else page.text,
                vocabulary,
            ),
        )
        for page in pages
    ]


def margin_key(line: str) -> str:
    """Forme comparable d'une ligne de marge : le numéro de page varie d'une
    page à l'autre, le reste de l'en-tête non."""
    value = _SPACES.sub(" ", line.strip().lower())
    value = _TRAILING_PAGE_NUMBER.sub("", value)
    value = _LEADING_PAGE_NUMBER.sub("", value).strip()
    return value or "#"


Margins = tuple[frozenset[str], frozenset[str]]


def repeated_margins(pages: Sequence[PageText]) -> Margins:
    """Lignes répétées en haut et en bas des pages.

    Seules les pages sans titre propre sont comparées (pages PDF) : une
    feuille de classeur ou un document Word ne se répète pas ainsi.
    """
    paginated = [
        lines
        for page in pages
        if page.section_title is None
        and (lines := [line for line in page.text.split("\n") if line.strip()])
    ]
    if len(paginated) < MARGIN_MIN_PAGES:
        return frozenset(), frozenset()
    top: Counter[str] = Counter()
    bottom: Counter[str] = Counter()
    for lines in paginated:
        top.update({margin_key(line) for line in lines[:MARGIN_LINES]})
        bottom.update({margin_key(line) for line in lines[-MARGIN_LINES:]})
    threshold = max(MARGIN_MIN_PAGES, math.ceil(len(paginated) * MARGIN_PAGE_SHARE))

    def kept(counter: Counter[str]) -> frozenset[str]:
        return frozenset(
            key
            for key, count in counter.items()
            if count >= threshold and len(key) <= MARGIN_MAX_CHARS
        )

    return kept(top), kept(bottom)


def strip_margins(text: str, top: frozenset[str], bottom: frozenset[str]) -> str:
    if not top and not bottom:
        return text
    lines = text.split("\n")
    filled = [index for index, line in enumerate(lines) if line.strip()]
    removed = {
        index for index in filled[:MARGIN_LINES] if margin_key(lines[index]) in top
    } | {
        index for index in filled[-MARGIN_LINES:] if margin_key(lines[index]) in bottom
    }
    if not removed:
        return text
    return "\n".join(
        line for index, line in enumerate(lines) if index not in removed
    ).strip()


def dehyphenate(text: str, vocabulary: set[str]) -> str:
    """Recolle les mots coupés en fin de ligne. Un mot composé
    (« suivi-évaluation ») garde son trait d'union s'il est écrit ainsi
    ailleurs dans le document et jamais d'un seul tenant."""

    def join(match: re.Match[str]) -> str:
        left, right = match.group(1), match.group(2)
        if not right[0].isalpha() or not right[0].islower() or left[-1].isdigit():
            return match.group(0)
        joined = f"{left}{right}"
        compound = f"{left}-{right}"
        if compound.lower() in vocabulary and joined.lower() not in vocabulary:
            return compound
        return joined

    return _HYPHEN_BREAK.sub(join, text)
