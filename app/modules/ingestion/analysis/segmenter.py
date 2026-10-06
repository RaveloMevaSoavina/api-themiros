"""Découpage en segments (spec §8, spec RAG §3.2).

Section, puis paragraphe, puis phrase si un paragraphe dépasse 1 024 tokens.
Cible 512 tokens, minimum 128, chevauchement de 64 tokens entre segments
consécutifs d'une même section.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

TARGET_TOKENS = 512
MAX_TOKENS = 1024
MIN_TOKENS = 128
OVERLAP_TOKENS = 64

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?;:])\s+(?=\S)")
# Titres : « 3.2 Mise en œuvre », « CHAPITRE II », « Annexe 1 - Budget ».
_NUMBERED_HEADING = re.compile(
    r"^((\d+(\.\d+)*\.?)|([IVXLC]+\.)|((chapitre|chapter|section|annexe|annex|partie|part)\s+\S+))\s*[-–:]?\s*\S",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PageText:
    page_number: int
    text: str
    char_start: int
    section_title: str | None = None


@dataclass(frozen=True, slots=True)
class Segment:
    page_number: int
    section_title: str | None
    paragraph_index: int
    text: str
    char_start: int
    char_end: int
    token_count: int


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    page_number: int
    paragraph_index: int
    char_start: int
    char_end: int
    section_title: str | None
    tokens: int


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    # Tokenizer du modèle d'embedding text-embedding-3-small.
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def is_heading(line: str) -> bool:
    value = line.strip()
    if not value or len(value) > 120 or "\n" in value:
        return False
    if value.endswith((".", ",", ";")) and not _NUMBERED_HEADING.match(value):
        return False
    words = value.split()
    if len(words) > 14:
        return False
    if _NUMBERED_HEADING.match(value) and len(words) >= 2:
        return True
    letters = [character for character in value if character.isalpha()]
    return len(letters) >= 4 and all(not c.islower() for c in letters)


def _paragraphs(page: PageText) -> Iterable[tuple[str, int, int]]:
    """Paragraphes d'une page avec leur position dans le texte complet."""
    position = 0
    for block in _PARAGRAPH_BREAK.split(page.text):
        for line_group in _split_lines(block):
            value = line_group.strip()
            if not value:
                continue
            start = page.text.find(value, position)
            if start < 0:
                start = position
            position = start + len(value)
            yield value, page.char_start + start, page.char_start + position


def _split_lines(block: str) -> list[str]:
    """Dans un bloc sans ligne vide (texte PDF), un titre seul sur sa ligne
    ouvre un nouveau paragraphe ; les autres lignes sont recollées."""
    groups: list[str] = []
    current: list[str] = []
    for line in block.split("\n"):
        if is_heading(line):
            if current:
                groups.append("\n".join(current))
                current = []
            groups.append(line)
        else:
            current.append(line)
    if current:
        groups.append("\n".join(current))
    return groups


def _units(pages: Iterable[PageText]) -> list[_Unit]:
    units: list[_Unit] = []
    paragraph_index = 0
    for page in pages:
        section = page.section_title
        for text, start, end in _paragraphs(page):
            if is_heading(text):
                section = text.strip()
            tokens = count_tokens(text)
            if tokens <= MAX_TOKENS:
                units.append(
                    _Unit(
                        text,
                        page.page_number,
                        paragraph_index,
                        start,
                        end,
                        section,
                        tokens,
                    )
                )
            else:
                units.extend(
                    _split_long(text, page.page_number, paragraph_index, start, section)
                )
            paragraph_index += 1
    return units


def _split_long(
    text: str, page_number: int, paragraph_index: int, start: int, section: str | None
) -> list[_Unit]:
    """Paragraphe trop long : découpe par phrase, puis par fenêtre de tokens
    pour une phrase qui dépasse encore la taille maximale."""
    encoding = _encoding()
    units: list[_Unit] = []
    cursor = 0
    for sentence in _SENTENCE_BREAK.split(text):
        offset = text.find(sentence, cursor)
        offset = cursor if offset < 0 else offset
        cursor = offset + len(sentence)
        tokens = encoding.encode(sentence, disallowed_special=())
        if len(tokens) <= MAX_TOKENS:
            units.append(
                _Unit(
                    sentence,
                    page_number,
                    paragraph_index,
                    start + offset,
                    start + cursor,
                    section,
                    len(tokens),
                )
            )
            continue
        step = TARGET_TOKENS - OVERLAP_TOKENS
        for window_start in range(0, len(tokens), step):
            window = encoding.decode(
                tokens[window_start : window_start + TARGET_TOKENS]
            )
            units.append(
                _Unit(
                    window,
                    page_number,
                    paragraph_index,
                    start + offset,
                    start + cursor,
                    section,
                    min(TARGET_TOKENS, len(tokens) - window_start),
                )
            )
    return units


def segment_pages(pages: Iterable[PageText]) -> list[Segment]:
    encoding = _encoding()
    segments: list[Segment] = []
    chunk: list[_Unit] = []
    overlap = ""
    overlap_tokens = 0

    def flush(keep_overlap: bool) -> str:
        if not chunk:
            return ""
        body = "\n".join(unit.text for unit in chunk)
        text = f"{overlap}\n{body}".strip() if overlap else body
        segments.append(
            Segment(
                page_number=chunk[0].page_number,
                section_title=chunk[0].section_title,
                paragraph_index=chunk[0].paragraph_index,
                text=text,
                char_start=chunk[0].char_start,
                char_end=chunk[-1].char_end,
                token_count=count_tokens(text),
            )
        )
        if not keep_overlap:
            return ""
        tokens = encoding.encode(body, disallowed_special=())
        return (
            encoding.decode(tokens[-OVERLAP_TOKENS:])
            if len(tokens) > OVERLAP_TOKENS
            else ""
        )

    for unit in _units(pages):
        if chunk and unit.section_title != chunk[-1].section_title:
            # Nouvelle section : pas de chevauchement entre sections.
            flush(False)
            _merge_small_tail(segments, chunk)
            chunk, overlap, overlap_tokens = [], "", 0
        current = sum(item.tokens for item in chunk) + overlap_tokens
        if chunk and current + unit.tokens > TARGET_TOKENS:
            overlap = flush(True)
            overlap_tokens = count_tokens(overlap)
            chunk = []
        chunk.append(unit)
    if chunk:
        flush(False)
        _merge_small_tail(segments, chunk)
    return segments


def _merge_small_tail(segments: list[Segment], chunk: list[_Unit]) -> None:
    """Un dernier segment de section sous 128 tokens rejoint le précédent de la
    même section, tant que la taille maximale est respectée."""
    if len(segments) < 2:
        return
    tail, previous = segments[-1], segments[-2]
    if (
        tail.token_count >= MIN_TOKENS
        or previous.section_title != tail.section_title
        or previous.token_count + tail.token_count > MAX_TOKENS
    ):
        return
    body = "\n".join(unit.text for unit in chunk)
    text = f"{previous.text}\n{body}"
    segments[-2:] = [
        Segment(
            page_number=previous.page_number,
            section_title=previous.section_title,
            paragraph_index=previous.paragraph_index,
            text=text,
            char_start=previous.char_start,
            char_end=tail.char_end,
            token_count=count_tokens(text),
        )
    ]
