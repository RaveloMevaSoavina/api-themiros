from app.modules.ingestion.domain import (
    COUNTRY_UNDETECTED_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    Detection,
    IngestionRules,
    RelevanceSignals,
    WorkspaceProfile,
    country_name,
    decide,
    fingerprint_text,
    score_relevance,
)

RULES = IngestionRules()
PROFILE = WorkspaceProfile(
    id="11111111-1111-4111-8111-111111111111",
    kind="programme",
    target_country="ET",
    financiers=("FIDA",),
    themes=("agriculture", "water"),
    expected_languages=("fr", "en"),
    stage="implementation",
    start_year=2022,
    end_year=2028,
)


def _decide(
    score: int, country: str | None = "ET", exploitable: int = 10, total: int = 10
):
    return decide(
        score=score,
        profile=PROFILE,
        country=Detection(country, 0.9),
        exploitable_pages=exploitable,
        total_pages=total,
        rules=RULES,
    )


def test_score_applies_bonuses_and_clamps_to_100() -> None:
    score = score_relevance(
        semantic=90,
        profile=PROFILE,
        language=Detection("en", 0.99),
        country=Detection("ET", 0.8),
        signals=RelevanceSignals(financiers_found=("FIDA",), themes_found=("water",)),
        rules=RULES,
    )
    assert score.adjustments == {
        "country": 10,
        "financier": 5,
        "theme": 5,
        "language": 5,
    }
    assert score.final == 100


def test_score_applies_maluses_for_other_country_and_off_topic() -> None:
    score = score_relevance(
        semantic=45,
        profile=PROFILE,
        language=Detection("en", 0.99),
        country=Detection("KE", 0.8),
        signals=RelevanceSignals(off_topic_keywords_found=("technical offer",)),
        rules=RULES,
    )
    assert score.adjustments["other_country"] == -30
    assert score.adjustments["off_topic"] == -20
    assert score.final == 0


def test_conforme_at_70_and_above() -> None:
    decision = _decide(70)
    assert decision.status == "conforme"
    assert decision.reason_code is None


def test_ambiguous_between_40_and_69() -> None:
    decision = _decide(55)
    assert decision.status == "a_verifier"
    assert decision.reason_code == "out_of_scope"
    assert decision.message == OUT_OF_SCOPE_MESSAGE


def test_rejected_below_40_with_explicit_reason() -> None:
    decision = _decide(28)
    assert decision.status == "rejete"
    assert decision.message == OUT_OF_SCOPE_MESSAGE


def test_other_country_requires_review_with_imposed_message() -> None:
    decision = _decide(85, country="KE")
    assert decision.status == "a_verifier"
    assert decision.reason_code == "country_mismatch"
    assert decision.reason_params == {"detected": "KE", "target": "ET"}
    assert decision.message is not None
    assert decision.message.startswith("⚠ Ce document semble concerner Kenya")
    assert "configuré pour Éthiopie" in decision.message


def test_other_country_below_40_is_rejected_with_country_reason() -> None:
    decision = _decide(10, country="KE")
    assert decision.status == "rejete"
    assert decision.reason_code == "country_mismatch"


def test_undetected_country_requires_review() -> None:
    decision = _decide(90, country=None)
    assert decision.status == "a_verifier"
    assert decision.message == COUNTRY_UNDETECTED_MESSAGE


def test_partial_coverage_message_uses_page_counts() -> None:
    decision = _decide(85, exploitable=12, total=40)
    assert decision.status == "conforme"
    assert decision.reason_code == "partial_coverage"
    assert decision.reason_params == {"n": 12, "total": 40}
    assert decision.message == (
        "ℹ Ce document a été accepté, mais seules 12 pages sur 40 contiennent "
        "du contenu exploitable pour l'analyse. Les scores refléteront cette "
        "couverture partielle."
    )


def test_thresholds_come_from_versioned_settings() -> None:
    strict = IngestionRules(version=2, conformity_threshold=80, ambiguous_threshold=50)
    decision = decide(
        score=75,
        profile=PROFILE,
        country=Detection("ET", 1),
        exploitable_pages=1,
        total_pages=1,
        rules=strict,
    )
    assert decision.status == "a_verifier"


def test_fingerprint_text_follows_the_specification_format() -> None:
    text = fingerprint_text(PROFILE)
    assert text.startswith("programme en Éthiopie financé par FIDA")
    assert "thématiques agriculture, water" in text
    assert "années 2022-2028" in text
    assert text.endswith("langues FR, EN")


def test_country_name_is_localized() -> None:
    assert country_name("CI", "fr") == "Côte d'Ivoire"
    assert country_name("ET", "en") == "Ethiopia"
    assert country_name("ZZ") is None
