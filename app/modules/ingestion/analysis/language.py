"""Détection de la langue (spec §5).

La spec recommande fastText et accepte langdetect ; langdetect est retenu
car il s'installe sans compilation et suffit sur des échantillons de 5 000
caractères.
"""

from collections import Counter

from langdetect import DetectorFactory, LangDetectException, detect_langs

from app.modules.ingestion.domain import Detection

DetectorFactory.seed = 0

SAMPLE_CHARS = 10_000
WINDOW_CHARS = 5_000
MIN_SAMPLE_CHARS = 20


def _predict(sample: str) -> tuple[str, float] | None:
    if len(sample.strip()) < MIN_SAMPLE_CHARS:
        return None
    try:
        best = detect_langs(sample)[0]
    except LangDetectException:
        return None
    return best.lang, float(best.prob)


def detect_language(text: str, confidence_threshold: float = 0.7) -> Detection:
    """Algorithme de la spec §5.3 : un échantillon de 10 000 caractères, puis
    un vote sur des fenêtres de 5 000 caractères si la confiance est faible."""
    first = _predict(text[:SAMPLE_CHARS])
    if first and first[1] >= confidence_threshold:
        return Detection(first[0], round(first[1], 4))

    windows = [
        text[start : start + WINDOW_CHARS]
        for start in range(0, len(text), WINDOW_CHARS)
    ]
    votes = [prediction[0] for window in windows if (prediction := _predict(window))]
    if not votes:
        if first:
            return Detection(first[0], round(first[1], 4))
        return Detection(None, 0.0)
    language, count = Counter(votes).most_common(1)[0]
    return Detection(language, round(count / len(votes), 4))
