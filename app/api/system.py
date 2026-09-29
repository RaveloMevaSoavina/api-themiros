from datetime import UTC, datetime
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.infrastructure.supabase import SupabaseGateway

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    timestamp: datetime


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    return HealthResponse(version=settings.app_version, timestamp=datetime.now(UTC))


@router.get("/ready", response_model=ReadyResponse)
async def ready(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReadyResponse | JSONResponse:
    client: httpx.AsyncClient = request.app.state.http_client
    gateway = SupabaseGateway(settings, client)
    try:
        db_ok, db_status = await gateway.ping()
    except httpx.HTTPError as error:
        db_ok, db_status = False, type(error).__name__

    payload = ReadyResponse(
        status="ready" if db_ok else "not_ready",
        checks={"db": db_status},
    )
    if db_ok:
        return payload
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=payload.model_dump(),
    )
