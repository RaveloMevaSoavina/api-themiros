import pytest

from app.modules.approaches.engine import compute_complexity
from app.modules.approaches.schemas import ComplexityCalculationRequest


def request(**overrides) -> ComplexityCalculationRequest:
    values = {
        "scale": "national",
        "actors": "2-3",
        "themes": ["agriculture", "eau"],
        "object_type": "program",
        "budget": "inconnu",
    }
    values.update(overrides)
    return ComplexityCalculationRequest(**values)


def test_documented_complexity_example_is_complicated() -> None:
    result = compute_complexity(request())

    assert result.score == 5
    assert result.value == "complique"
    assert result.calculated_value == "complique"
    assert result.factors.model_dump() == {
        "scale": 1,
        "actors": 1,
        "themes": 1,
        "object_type": 1,
        "budget": 1,
    }
    assert result.overridden is False


@pytest.mark.parametrize(
    ("overrides", "expected_score", "expected_class"),
    [
        (
            {
                "scale": "local",
                "actors": "1",
                "themes": ["eau"],
                "object_type": "project",
                "budget": "<5M",
            },
            0,
            "simple",
        ),
        ({}, 5, "complique"),
        (
            {
                "scale": "multi-pays",
                "actors": "4+",
                "themes": ["a", "b", "c", "d"],
                "object_type": "policy",
                "budget": ">50M",
            },
            10,
            "complexe",
        ),
    ],
)
def test_complexity_boundaries(overrides, expected_score, expected_class) -> None:
    result = compute_complexity(request(**overrides))

    assert result.score == expected_score
    assert result.value == expected_class


def test_duplicate_themes_do_not_increase_complexity() -> None:
    result = compute_complexity(request(themes=["eau", "eau"]))

    assert result.factors.themes == 0
    assert result.score == 4


def test_manual_class_override_keeps_calculated_score() -> None:
    result = compute_complexity(request(class_override="complexe"))

    assert result.score == 5
    assert result.calculated_value == "complique"
    assert result.value == "complexe"
    assert result.overridden is True


def test_same_inputs_always_return_same_result() -> None:
    payload = request()

    assert compute_complexity(payload) == compute_complexity(payload)
