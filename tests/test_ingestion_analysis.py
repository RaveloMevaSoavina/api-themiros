from app.modules.ingestion.analysis.country import CountryDetector
from app.modules.ingestion.analysis.embeddings import l2_normalize, semantic_score
from app.modules.ingestion.analysis.language import detect_language
from app.modules.ingestion.analysis.segmenter import (
    MAX_TOKENS,
    TARGET_TOKENS,
    PageText,
    is_heading,
    segment_pages,
)
from app.modules.ingestion.analysis.signals import detect_signals

FRENCH = (
    "Le programme d'adaptation agricole soutient les communautés rurales. "
    "Les activités concernent la gestion de l'eau, l'irrigation et la "
    "résilience des exploitations familiales face aux sécheresses répétées. "
) * 20
ENGLISH = (
    "The programme supports rural communities through climate adaptation, "
    "water management and resilient agricultural practices for smallholders. "
) * 20


def test_detects_french_and_english() -> None:
    assert detect_language(FRENCH).value == "fr"
    english = detect_language(ENGLISH)
    assert english.value == "en"
    assert english.confidence >= 0.7


def test_language_votes_on_windows_for_mixed_documents() -> None:
    detection = detect_language(FRENCH * 3 + ENGLISH)
    assert detection.value == "fr"


def test_language_unknown_for_empty_text() -> None:
    detection = detect_language("   ")
    assert detection.value is None
    assert detection.confidence == 0


def test_country_detection_with_multilingual_gazetteer() -> None:
    detector = CountryDetector(ner_model=None)
    text = (
        "Le programme couvre l'Éthiopie. En Éthiopie, les régions Amhara et "
        "Oromia sont ciblées. Une mission régionale a visité le Kenya."
    )
    detection = detector.detect(text)
    assert detection.value == "ET"
    assert detection.confidence == round(2 / 3, 4)


def test_country_detection_prefers_longest_names() -> None:
    detector = CountryDetector(ner_model=None)
    assert detector.detect("Projet en Papouasie-Nouvelle-Guinée").value == "PG"
    assert detector.detect("Projeto em Moçambique e no Moçambique").value == "MZ"
    assert detector.detect("Programa en Etiopía").value == "ET"


def test_country_detection_returns_none_without_mentions() -> None:
    detection = CountryDetector(ner_model=None).detect("Rapport annuel des activités")
    assert detection.value is None
    assert detection.confidence == 0


def test_heading_detection() -> None:
    assert is_heading("3.2 Mise en œuvre")
    assert is_heading("CHAPITRE II")
    assert is_heading("Annexe 1 - Budget")
    assert not is_heading("Le programme couvre plusieurs régions.")
    assert not is_heading("")


def test_segments_respect_target_size_overlap_and_offsets() -> None:
    paragraphs = "\n\n".join(
        f"Paragraphe {index}. " + "Les activités agricoles progressent. " * 30
        for index in range(30)
    )
    pages = [PageText(page_number=1, text=paragraphs, char_start=0)]
    segments = segment_pages(pages)

    assert len(segments) > 5
    assert all(segment.token_count <= MAX_TOKENS for segment in segments)
    assert all(segment.token_count <= TARGET_TOKENS + 64 for segment in segments[:-1])
    # Chevauchement : le segment suivant reprend la fin du précédent.
    tail = segments[0].text.split()[-5:]
    assert " ".join(tail) in segments[1].text
    assert segments[0].char_start == 0
    assert segments[1].char_start > segments[0].char_start


def test_segments_track_sections_and_pages() -> None:
    pages = [
        PageText(1, "1. Contexte\n\n" + "Le contexte national. " * 80, 0),
        PageText(2, "2. Budget\n\n" + "Le budget prévisionnel. " * 80, 5000),
    ]
    segments = segment_pages(pages)
    sections = {segment.section_title for segment in segments}
    assert {"1. Contexte", "2. Budget"} <= sections
    budget = next(
        segment for segment in segments if segment.section_title == "2. Budget"
    )
    assert budget.page_number == 2
    assert budget.char_start >= 5000


def test_very_long_paragraph_is_split_by_sentence() -> None:
    long_paragraph = "Une phrase sur le suivi des indicateurs du programme. " * 400
    segments = segment_pages([PageText(1, long_paragraph, 0)])
    assert len(segments) > 1
    assert all(segment.token_count <= MAX_TOKENS for segment in segments)


def test_small_final_segment_is_merged_into_previous_one() -> None:
    text = "\n\n".join(["Les activités agricoles progressent. " * 60, "Fin."])
    segments = segment_pages([PageText(1, text, 0)])
    assert segments[-1].text.endswith("Fin.")
    assert segments[-1].token_count >= 128


def test_signals_detect_financiers_themes_and_off_topic() -> None:
    text = (
        "TECHNICAL OFFER for road construction. Financement du FIDA et de "
        "l'Union européenne pour l'irrigation."
    )
    signals = detect_signals(
        text,
        financier_terms=["FIDA", "IFAD", "UE", "Union européenne"],
        themes=["water", "gender"],
        off_topic_keywords=["technical offer", "appel d offres"],
    )
    assert signals.financiers_found == ("FIDA", "Union européenne")
    assert signals.themes_found == ("water",)
    assert signals.off_topic_keywords_found == ("technical offer",)


def test_acronyms_are_matched_case_sensitively() -> None:
    signals = detect_signals(
        "eu sou do Brasil, o projeto ue",
        financier_terms=["UE", "EU"],
        themes=[],
        off_topic_keywords=[],
    )
    assert signals.financiers_found == ()


def test_off_topic_keyword_far_in_document_needs_repetition() -> None:
    body = "Programme agricole. " * 600 + " appel d'offres pour les semences."
    signals = detect_signals(
        body, financier_terms=[], themes=[], off_topic_keywords=["appel d offres"]
    )
    assert signals.off_topic_keywords_found == ()


def test_semantic_score_is_weighted_by_segment_length() -> None:
    fingerprint = l2_normalize([1.0, 0.0])
    aligned = l2_normalize([1.0, 0.0])
    orthogonal = l2_normalize([0.0, 1.0])
    assert semantic_score([aligned, orthogonal], fingerprint, [3, 1]) == 75
    assert semantic_score([], fingerprint, []) == 0
