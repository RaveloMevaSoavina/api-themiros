from app.modules.approaches.schemas import (
    ComplexityCalculation,
    ComplexityCalculationRequest,
    ComplexityClass,
    ComplexityFactors,
)

SCALE_POINTS = {"local": 0, "national": 1, "multi-pays": 2}
ACTOR_POINTS = {"1": 0, "2-3": 1, "4+": 2}
OBJECT_TYPE_POINTS = {"project": 0, "program": 1, "policy": 2}
BUDGET_POINTS = {"<5M": 0, "5-50M": 1, "inconnu": 1, ">50M": 2}


def _theme_points(theme_count: int) -> int:
    if theme_count == 1:
        return 0
    if theme_count <= 3:
        return 1
    return 2


def _complexity_class(score: int) -> ComplexityClass:
    if score <= 3:
        return "simple"
    if score <= 6:
        return "complique"
    return "complexe"


def compute_complexity(
    request: ComplexityCalculationRequest,
) -> ComplexityCalculation:
    """Apply RG-4.4 without any model call or non-deterministic input."""
    factors = ComplexityFactors(
        scale=SCALE_POINTS[request.scale],
        actors=ACTOR_POINTS[request.actors],
        themes=_theme_points(len(set(request.themes))),
        object_type=OBJECT_TYPE_POINTS[request.object_type],
        budget=BUDGET_POINTS[request.budget],
    )
    score = sum(factors.model_dump().values())
    calculated_value = _complexity_class(score)
    value = request.class_override or calculated_value

    return ComplexityCalculation(
        value=value,
        calculated_value=calculated_value,
        score=score,
        factors=factors,
        overridden=request.class_override is not None,
    )
