from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import CurrentUser, get_current_user
from app.modules.approaches.engine import compute_complexity
from app.modules.approaches.schemas import (
    ComplexityCalculation,
    ComplexityCalculationRequest,
)

router = APIRouter(prefix="/approach", tags=["approach"])


@router.post("/complexity", response_model=ComplexityCalculation)
async def calculate_complexity(
    payload: ComplexityCalculationRequest,
    _: Annotated[CurrentUser, Depends(get_current_user)],
) -> ComplexityCalculation:
    return compute_complexity(payload)
