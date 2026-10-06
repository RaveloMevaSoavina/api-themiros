"""Détection du pays mentionné (spec §6) : entités nommées spaCy filtrées par
un dictionnaire de pays, puis pays le plus cité.

Le dictionnaire couvre les noms de pays en anglais, français, espagnol et
portugais (P-3). Sans modèle spaCy disponible, le texte est parcouru
directement avec le dictionnaire.
"""

import gettext
import logging
from collections import Counter
from functools import lru_cache

import pycountry

from app.modules.ingestion.analysis.text import normalize
from app.modules.ingestion.domain import Detection

logger = logging.getLogger(__name__)

NER_MAX_CHARS = 200_000
GAZETTEER_LOCALES = ("fr", "es", "pt")
NER_LABELS = {"GPE", "LOC"}
EXTRA_ALIASES = {
    "rdc": "CD",
    "drc": "CD",
    "congo kinshasa": "CD",
    "congo brazzaville": "CG",
    "usa": "US",
    "etats unis": "US",
    "royaume uni": "GB",
    "uk": "GB",
    "cote d ivoire": "CI",
    "ivory coast": "CI",
    "cabo verde": "CV",
    "cap vert": "CV",
    "eswatini": "SZ",
    "swaziland": "SZ",
    "burma": "MM",
    "birmanie": "MM",
    "laos": "LA",
    "syrie": "SY",
    "tanzanie": "TZ",
    "bolivie": "BO",
    "venezuela": "VE",
    "vietnam": "VN",
    "viet nam": "VN",
    "russie": "RU",
    "coree du sud": "KR",
    "coree du nord": "KP",
    "south korea": "KR",
    "north korea": "KP",
}


@lru_cache(maxsize=1)
def country_gazetteer() -> dict[str, str]:
    """Nom normalisé → code ISO 3166-1 alpha-2."""
    translations = []
    for locale in GAZETTEER_LOCALES:
        try:
            translations.append(
                gettext.translation(
                    "iso3166-1", pycountry.LOCALES_DIR, languages=[locale]
                )
            )
        except OSError:
            logger.warning("Missing pycountry translations for %s", locale)

    gazetteer: dict[str, str] = {}
    for country in pycountry.countries:
        names = {
            country.name,
            getattr(country, "official_name", None),
            getattr(country, "common_name", None),
        }
        names.discard(None)
        for name in list(names):
            names.update(translation.gettext(name) for translation in translations)
        for name in names:
            key = normalize(name)
            # Les variantes « X, République de » restent trouvables par « X ».
            for variant in {
                key,
                key.split(" republic of")[0],
                key.split(" republique")[0],
            }:
                variant = variant.strip()
                if len(variant) >= 4:
                    gazetteer.setdefault(variant, country.alpha_2)
    gazetteer.update(EXTRA_ALIASES)
    return gazetteer


class CountryDetector:
    def __init__(self, ner_model: str | None = "xx_ent_wiki_sm") -> None:
        self._nlp = None
        if ner_model:
            try:
                import spacy

                self._nlp = spacy.load(ner_model, disable=["parser", "tagger"])
                self._nlp.max_length = NER_MAX_CHARS + 1_000
            except (ImportError, OSError):
                logger.warning(
                    "spaCy model %s unavailable: gazetteer-only country detection",
                    ner_model,
                )
        self._gazetteer = country_gazetteer()
        self._max_words = max(len(name.split()) for name in self._gazetteer)

    @property
    def uses_ner(self) -> bool:
        return self._nlp is not None

    def detect(self, text: str) -> Detection:
        candidates = self._ner_candidates(text) if self._nlp else []
        if not candidates:
            candidates = self._gazetteer_candidates(text)
        if not candidates:
            return Detection(None, 0.0)
        code, count = Counter(candidates).most_common(1)[0]
        return Detection(code, round(count / len(candidates), 4))

    def _ner_candidates(self, text: str) -> list[str]:
        assert self._nlp is not None
        document = self._nlp(text[:NER_MAX_CHARS])
        candidates: list[str] = []
        for entity in document.ents:
            if entity.label_ not in NER_LABELS:
                continue
            code = self._gazetteer.get(normalize(entity.text))
            if code:
                candidates.append(code)
        return candidates

    def _gazetteer_candidates(self, text: str) -> list[str]:
        """Parcours du texte, correspondance la plus longue d'abord
        (« Papouasie-Nouvelle-Guinée » avant « Guinée »)."""
        words = normalize(text).split()
        candidates: list[str] = []
        index = 0
        while index < len(words):
            for size in range(min(self._max_words, len(words) - index), 0, -1):
                code = self._gazetteer.get(" ".join(words[index : index + size]))
                if code:
                    candidates.append(code)
                    index += size
                    break
            else:
                index += 1
        return candidates


@lru_cache(maxsize=2)
def get_country_detector(ner_model: str | None) -> CountryDetector:
    return CountryDetector(ner_model)
