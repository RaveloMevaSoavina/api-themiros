from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.infrastructure.supabase import SupabaseGateway
from app.modules.demo_requests.guard import (
    AbuseGuard,
    client_ip,
    client_key,
    too_many,
)
from app.modules.demo_requests.notifier import ResendNotifier
from app.modules.demo_requests.repository import DemoRequestRepository
from app.modules.demo_requests.schemas import DemoRequestPayload, DemoRequestReceipt
from app.modules.demo_requests.service import DemoRequestService

router = APIRouter(prefix="/demo-requests", tags=["demo-requests"])


def get_demo_request_service(
    request: Request, settings: Annotated[Settings, Depends(get_settings)]
) -> DemoRequestService:
    client: httpx.AsyncClient = request.app.state.http_client
    return DemoRequestService(
        DemoRequestRepository(SupabaseGateway(settings, client)),
        ResendNotifier(client, settings.resend_api_key)
        if settings.resend_api_key
        else None,
        sender=settings.demo_request_mail_from,
        recipients=settings.demo_request_recipients,
        backoffice_url=settings.backoffice_url,
        rate_limit=settings.demo_request_rate_limit,
        rate_window_seconds=settings.demo_request_rate_window_seconds,
    )


def get_abuse_guard(request: Request) -> AbuseGuard:
    return request.app.state.demo_request_guard


@router.post(
    "",
    response_model=DemoRequestReceipt,
    status_code=201,
    responses={
        413: {"description": "Body larger than 16 KiB"},
        422: {"description": "Missing or invalid field"},
        429: {"description": "Too many requests, or address blocked"},
    },
)
async def submit_demo_request(
    payload: DemoRequestPayload,
    request: Request,
    service: Annotated[DemoRequestService, Depends(get_demo_request_service)],
    guard: Annotated[AbuseGuard, Depends(get_abuse_guard)],
) -> DemoRequestReceipt | JSONResponse:
    """Formulaire public de la landing page : aucune authentification.
    Tentatives par adresse et corps trop gros : voir guard.py (middleware)."""
    ip = client_ip(request)

    if payload.website:
        # Robot : banni, mais il reçoit un succès et n'apprend rien.
        guard.ban(client_key(ip), reason="honeypot")
        return DemoRequestReceipt()

    retry_after = guard.global_retry_after()
    if retry_after is not None:
        return too_many("DEMO_REQUEST_GLOBAL_LIMIT", retry_after)

    if await service.submit(
        payload, ip=ip, user_agent=request.headers.get("user-agent")
    ):
        guard.record_stored()
    return DemoRequestReceipt()
