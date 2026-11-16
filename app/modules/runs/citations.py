"""Citations dans les textes rédigés par l'IA (constats, conclusion, actions,
justifications des critères).

Chaque extrait part vers l'IA découpé en phrases (`x: {"E1": ["…", "…"]}`).
L'IA cite dans son texte, après l'affirmation qu'elles fondent, la phrase
ou l'extrait entier : « … 60 % par an [E1.2, E3] ». Le moteur :

- ne garde que les références valables (extrait proposé pour ce texte ;
  une phrase hors de celles envoyées vaut l'extrait entier) ;
- regroupe les phrases d'un même extrait en une source, repérée par le
  segment et les positions des phrases dans son texte (`passages`) ;
- réécrit le marqueur en numéros de sources : « ⟦1,2⟧ ». Les crochets
  mathématiques n'apparaissent pas dans un texte rédigé : un marqueur ne se
  confond pas avec une référence bibliographique « [1] ».

Les sources d'un texte sont numérotées dans l'ordre de leur première
citation, traductions comprises : « Source 2 » est le même passage dans
toutes les langues. `complete_run` rattache chaque source à une preuve du run.

Fonctions pures : aucune dépendance au stockage ni au réseau.
"""

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from app.modules.evidences.passages import Span, clean, merge, sentence_spans

OPEN, CLOSE = "⟦", "⟧"
MARKER = re.compile(r"⟦(\d+(?:,\d+)*)⟧")
_RAW_MARKER = re.compile(
    r"\s*[\[(]\s*(E\d+(?:\.\d+)?(?:\s*[,;]\s*E\d+(?:\.\d+)?)*)\s*[\])]",
    re.IGNORECASE,
)
_REFERENCE = re.compile(r"^E(\d+)(?:\.(\d+))?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class Excerpt:
    """Extrait proposé à l'IA : ses phrases nettoyées et leurs positions
    dans le texte du segment."""

    segment_id: UUID
    sentences: tuple[str, ...]
    spans: tuple[Span, ...]


@dataclass(frozen=True, slots=True)
class Source:
    """Passage cité : un segment et les positions des phrases citées."""

    segment_id: UUID
    spans: tuple[Span, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_id": str(self.segment_id),
            "spans": [list(span) for span in self.spans],
        }


def shorten(text: str, limit: int) -> str:
    """Coupe au dernier mot avant `limit` caractères, suivie de « … »."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut + "…"


def excerpt(segment_id: UUID, text: str, limit: int) -> Excerpt:
    """Les premières phrases du segment, dans la limite de `limit`
    caractères ; une première phrase plus longue est coupée."""
    sentences: list[str] = []
    spans: list[Span] = []
    used = 0
    for span in sentence_spans(text):
        sentence = clean(text[span[0] : span[1]])
        if not sentence:
            continue
        if sentences and used + len(sentence) > limit:
            break
        sentences.append(shorten(sentence, limit))
        spans.append(span)
        used += len(sentence) + 1
    return Excerpt(segment_id, tuple(sentences), tuple(spans))


def resolve(
    reference: str,
    excerpts: Mapping[str, Excerpt],
    allowed: Collection[str] | None = None,
) -> tuple[str, tuple[Span, ...]] | None:
    """Extrait et phrases désignés par « E1.2 » ou « E1 », ou `None` si
    l'extrait n'est pas proposé. Une phrase inconnue vaut l'extrait entier."""
    match = _REFERENCE.match(reference.strip())
    if match is None:
        return None
    key = f"E{int(match.group(1))}"
    item = excerpts.get(key)
    if item is None or (allowed is not None and key not in allowed):
        return None
    if match.group(2) is not None:
        index = int(match.group(2)) - 1
        if 0 <= index < len(item.spans):
            return key, (item.spans[index],)
    return key, item.spans


def cited(
    references: Iterable[str],
    excerpts: Mapping[str, Excerpt],
    allowed: Collection[str] | None = None,
) -> dict[str, list[Span]]:
    """Extraits valables parmi les références, dans l'ordre de citation,
    avec les positions de leurs phrases citées."""
    found: dict[str, list[Span]] = {}
    for reference in references:
        resolved = resolve(reference, excerpts, allowed)
        if resolved is not None:
            key, spans = resolved
            found.setdefault(key, []).extend(spans)
    return found


def references(text: str | None) -> list[str]:
    """Références citées dans un texte de l'IA (« [E1.2, E3] »)."""
    return [
        reference.strip()
        for match in _RAW_MARKER.finditer(text or "")
        for reference in re.split(r"[,;]", match.group(1))
    ]


@dataclass(slots=True)
class Citations:
    """Sources d'un texte et de ses traductions."""

    excerpts: Mapping[str, Excerpt]
    sources: list[Source] = field(default_factory=list)

    def cite(self, text: str, allowed: Collection[str] | None = None) -> str:
        """Le texte, ses marqueurs réécrits en numéros de sources ; un
        marqueur sans référence valable disparaît."""
        text = text.replace(OPEN, "").replace(CLOSE, "")

        def replace(match: re.Match[str]) -> str:
            found = cited(re.split(r"[,;]", match.group(1)), self.excerpts, allowed)
            numbers = list(
                dict.fromkeys(self._number(key, spans) for key, spans in found.items())
            )
            if not numbers:
                return ""
            return OPEN + ",".join(str(number) for number in numbers) + CLOSE

        return _RAW_MARKER.sub(replace, text)

    def _number(self, key: str, spans: Sequence[Span]) -> int:
        source = Source(self.excerpts[key].segment_id, tuple(merge(spans)))
        if source not in self.sources:
            self.sources.append(source)
        return self.sources.index(source) + 1


def compact(
    texts: Sequence[str | None], sources: Sequence[Source]
) -> tuple[list[str | None], list[Source]]:
    """Sources renumérotées dans l'ordre de leur première citation, celles
    qu'aucun texte ne cite (coupées avec la fin d'un texte) retirées."""
    order: dict[int, int] = {}
    for text in texts:
        for match in MARKER.finditer(text or ""):
            for number in match.group(1).split(","):
                index = int(number)
                if 1 <= index <= len(sources) and index not in order:
                    order[index] = len(order) + 1

    def replace(match: re.Match[str]) -> str:
        numbers = list(
            dict.fromkeys(
                order[int(number)]
                for number in match.group(1).split(",")
                if int(number) in order
            )
        )
        if not numbers:
            return ""
        return OPEN + ",".join(str(number) for number in numbers) + CLOSE

    renumbered = [MARKER.sub(replace, text) if text else text for text in texts]
    kept = [sources[index - 1] for index in sorted(order, key=order.__getitem__)]
    return renumbered, kept


@dataclass(slots=True)
class Reading:
    """Interprétation d'un extrait : ce qu'il établit sur le projet, ses
    marqueurs « ⟦1⟧ » renvoyant à ses propres phrases (`citations`), ses
    traductions (mêmes numéros), et toutes ses phrases citées (`spans`)."""

    text: str
    spans: tuple[Span, ...] = ()
    translations: dict[str, str] = field(default_factory=dict)
    citations: list[Source] = field(default_factory=list)


def read(
    key: str,
    text: str | None,
    translated: Mapping[str, str | None],
    excerpts: Mapping[str, Excerpt],
    limit: int,
    max_chars: int,
    fallback: Sequence[Span] = (),
    min_chars: int = 0,
) -> Reading | None:
    """Interprétation de l'extrait `key` : ses sources ne peuvent être que
    ses propres phrases (« [E1.2] ») ; une référence à un autre extrait
    disparaît. Sans source valable, elle renvoie aux phrases `fallback`
    (celles que les autres textes citent), à défaut à l'extrait entier :
    une interprétation mène toujours au passage surligné. `None` si le
    texte est vide ou plus court que `min_chars` hors marqueurs."""
    if not text or not text.strip() or key not in excerpts:
        return None
    allowed = (key,)
    citations = Citations(excerpts)
    original = truncate(citations.cite(text, allowed), limit)
    if not long_enough(original, min_chars):
        return None
    versions = {
        locale: truncate(citations.cite(value, allowed), limit)
        for locale, value in translated.items()
        if value and value.strip()
    }
    locales = list(versions)
    texts, sources = compact(
        [original, *(versions[locale] for locale in locales)], citations.sources
    )
    if not sources:
        item = excerpts[key]
        sources = [Source(item.segment_id, tuple(merge(fallback or item.spans)))]
        texts = [_marked(value) if value else value for value in texts]
    return Reading(
        text=fit(texts[0] or "", max_chars),
        translations={
            locale: fit(value or "", max_chars)
            for locale, value in zip(locales, texts[1:], strict=True)
            if value
        },
        citations=sources,
    )


def long_enough(text: str | None, min_chars: int) -> bool:
    """Au moins `min_chars` caractères hors marqueurs : un paragraphe, pas
    une phrase de circonstance."""
    return len(plain(text).strip()) >= min_chars


def written(
    text: str | None,
    translated: Mapping[str, str | None],
    excerpts: Mapping[str, Excerpt],
    allowed: Collection[str] | None,
    limit: int,
    max_chars: int,
    min_chars: int = 0,
) -> Reading | None:
    """Interprétation rédigée par l'IA (variable, pilier, sous-question,
    écart…) et ses traductions : sources numérotées, les mêmes dans toutes
    les langues, prises parmi les extraits `allowed`. `None` si le texte est
    vide, plus court que `min_chars` hors marqueurs, ou sans aucune source
    valable : une interprétation sans source n'est pas publiée. Une
    traduction trop courte est ignorée (le texte s'affiche dans la langue du
    run)."""
    if not text or not text.strip():
        return None
    citations = Citations(excerpts)
    original = truncate(citations.cite(text, allowed), limit)
    if not long_enough(original, min_chars):
        return None
    versions = {
        locale: truncate(citations.cite(value, allowed), limit)
        for locale, value in translated.items()
        if value and value.strip()
    }
    locales = [
        locale for locale, value in versions.items() if long_enough(value, min_chars)
    ]
    texts, sources = compact(
        [original, *(versions[locale] for locale in locales)], citations.sources
    )
    if not sources:
        return None
    return Reading(
        text=fit(texts[0] or "", max_chars),
        translations={
            locale: fit(value or "", max_chars)
            for locale, value in zip(locales, texts[1:], strict=True)
            if value
        },
        citations=sources,
    )


def reading_payload(reading: Reading | None) -> dict[str, Any]:
    """Champs `interpretation*` du corps de `complete_run`."""
    if reading is None:
        return {
            "interpretation": None,
            "interpretation_translations": {},
            "interpretation_citations": [],
        }
    return {
        "interpretation": reading.text,
        "interpretation_translations": reading.translations,
        "interpretation_citations": [source.as_dict() for source in reading.citations],
    }


_END = re.compile(r"([.!?…]*)\s*$")


def _marked(text: str) -> str:
    """Le texte suivi du marqueur « ⟦1⟧ », avant sa ponctuation finale."""
    match = _END.search(text)
    assert match is not None
    return text[: match.start()].rstrip() + OPEN + "1" + CLOSE + match.group(1)


def plain(text: str | None) -> str:
    """Le texte sans ses marqueurs (exports, longueur affichée)."""
    return MARKER.sub("", text or "")


def cut(text: str, limit: int) -> str:
    """Comme `shorten`, la longueur comptée sans les marqueurs, qui ne sont
    jamais coupés : ceux d'avant la coupe restent."""
    if len(plain(text)) <= limit:
        return text
    count = 0
    index = 0
    while index < len(text) and count < limit - 1:
        marker = MARKER.match(text, index)
        if marker:
            index = marker.end()
            continue
        count += 1
        index += 1
    head = text[:index]
    space = max(head.rfind(" "), head.rfind("\n"))
    if space > 0 and len(plain(head[:space])) > limit // 2:
        head = head[:space]
    return head.rstrip() + "…"


def truncate(text: str, limit: int) -> str:
    """Espaces normalisés, puis `cut`."""
    return cut(" ".join(text.split()), limit)


def fit(text: str, limit: int) -> str:
    """Garde-fou du plafond de la base : un texte trop long avec ses
    marqueurs les perd plutôt que d'être coupé au milieu de l'un d'eux."""
    return text if len(text) <= limit else shorten(plain(text), limit)
