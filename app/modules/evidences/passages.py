"""Phrases d'un extrait, repérées par leur position dans le texte du segment.

L'IA cite des phrases (E1.2) ; le moteur enregistre leurs positions
`[début, fin)` dans le texte du segment, recopié tel quel dans la preuve :
le surlignage et le dashboard retrouvent le passage sans redécouper, même si
ce découpage évolue. Le texte des PDF garde ses retours à la ligne et ses
césures : une phrase est nettoyée pour la lecture, jamais ses positions.
"""

import re
from collections.abc import Iterable, Sequence

Span = tuple[int, int]

# Fin de phrase : ponctuation forte, guillemet ou parenthèse fermante
# facultatif, puis espace.
_END = re.compile(r"[.!?…]+[\"»”’)\]]*(?=\s)")
# Rupture sans ponctuation : ligne vide, ou puce en début de ligne.
_BREAK = re.compile(r"\n\s*\n|\n(?=\s*(?:[•▪◦●\-–*]|\d{1,2}[.)])\s)")
_STARTS_SENTENCE = re.compile(r"[\"«“(\[]?[A-ZÀ-ÖØ-Þ0-9•▪◦●\-–]")
_HYPHENATED = re.compile(r"(\w)-\s*\n\s*(\w)")
_LAST_WORD = re.compile(r"(\S+)$")
# Abréviations courantes qu'un point ne termine pas.
_ABBREVIATIONS = frozenset(
    {
        "m", "mm", "mme", "mmes", "mlle", "dr", "pr", "me", "st", "ste",
        "p", "pp", "art", "al", "cf", "ex", "fig", "vol", "no", "n°", "nº",
        "env", "min", "max", "approx", "réf", "ref", "chap", "sect",
        "mr", "mrs", "ms", "vs", "e.g", "i.e", "jan", "fév", "feb",
        "av", "apr", "sept", "oct", "nov", "déc", "dec", "inc", "ltd", "co",
    }
)  # fmt: skip
# Phrase trop longue (liste sans ponctuation, tableau) : coupée au
# point-virgule, faute de mieux.
MAX_SENTENCE_CHARS = 500
_SOFT_END = re.compile(r";(?=\s)")


def _is_abbreviation(text: str, end: int) -> bool:
    match = _LAST_WORD.search(text, 0, end)
    if match is None:
        return False
    word = match.group(1).rstrip(".…!?").lower()
    if word in _ABBREVIATIONS:
        return True
    # Initiale (« J. Dupont ») ou numéro de liste (« 1. », « II. »).
    return (len(word) == 1 and word.isalpha()) or (word.isdigit() and len(word) <= 2)


def _ends(text: str) -> Iterable[int]:
    """Positions où une phrase se termine (exclues)."""
    for match in _BREAK.finditer(text):
        yield match.start()
    for match in _END.finditer(text):
        after = text[match.end() :].lstrip()
        if not after or not _STARTS_SENTENCE.match(after):
            continue
        if match.group().startswith(".") and _is_abbreviation(text, match.start()):
            continue
        yield match.end()


def _trim(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _split_long(text: str, span: Span) -> list[Span]:
    start, end = span
    if end - start <= MAX_SENTENCE_CHARS:
        return [span]
    spans: list[Span] = []
    cursor = start
    for match in _SOFT_END.finditer(text, start, end):
        if match.end() - cursor >= 80:
            trimmed = _trim(text, cursor, match.end())
            if trimmed:
                spans.append(trimmed)
            cursor = match.end()
    trimmed = _trim(text, cursor, end)
    if trimmed:
        spans.append(trimmed)
    return spans


def sentence_spans(text: str) -> list[Span]:
    """Positions `[début, fin)` des phrases du texte, dans l'ordre."""
    cuts = sorted({0, *_ends(text), len(text)})
    spans: list[Span] = []
    for start, end in zip(cuts, cuts[1:], strict=False):
        trimmed = _trim(text, start, end)
        if trimmed:
            spans.extend(_split_long(text, trimmed))
    return spans


def clean(text: str) -> str:
    """Texte lisible : césures de fin de ligne recollées, retours à la ligne
    et espaces multiples ramenés à un espace."""
    return " ".join(_HYPHENATED.sub(r"\1\2", text).split())


def merge(spans: Iterable[Sequence[int]]) -> list[Span]:
    """Positions triées, celles qui se touchent ou se chevauchent fusionnées."""
    merged: list[Span] = []
    for start, end in sorted((int(start), int(end)) for start, end in spans):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def passage(text: str, spans: Iterable[Sequence[int]]) -> str:
    """Phrases citées, nettoyées ; « … » entre deux passages disjoints."""
    parts = [
        clean(text[start:end])
        for start, end in merge(spans)
        if 0 <= start < end <= len(text)
    ]
    return " … ".join(part for part in parts if part)
