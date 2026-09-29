from typing import Literal

from pydantic import BaseModel, Field

Scale = Literal["local", "national", "multi-pays"]
ActorCount = Literal["1", "2-3", "4+"]
BudgetRange = Literal["<5M", "5-50M", ">50M", "inconnu"]
ObjectType = Literal["project", "program", "policy"]
ComplexityClass = Literal["simple", "complique", "complexe"]


class ComplexityCalculationRequest(BaseModel):
    scale: Scale
    actors: ActorCount
    themes: list[str] = Field(min_length=1)
    object_type: ObjectType
    budget: BudgetRange
    class_override: ComplexityClass | None = None


class ComplexityFactors(BaseModel):
    scale: int = Field(ge=0, le=2)
    actors: int = Field(ge=0, le=2)
    themes: int = Field(ge=0, le=2)
    object_type: int = Field(ge=0, le=2)
    budget: int = Field(ge=0, le=2)


class ComplexityCalculation(BaseModel):
    value: ComplexityClass
    calculated_value: ComplexityClass
    score: int = Field(ge=0, le=10)
    factors: ComplexityFactors
    rule: Literal["RG-4.4"] = "RG-4.4"
    overridden: bool
