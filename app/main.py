from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.system import router as system_router
from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.errors import install_error_handlers
from app.core.middleware import install_middleware
from app.modules.demo_requests.guard import AbuseGuard, install_demo_request_guard


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient() as client:
        app.state.http_client = client
        yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.app_debug,
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    # Avant le CORS : le dernier middleware ajouté est le plus externe, et les
    # refus du garde doivent porter les en-têtes CORS.
    install_demo_request_guard(
        app,
        AbuseGuard(
            attempt_limit=settings.demo_request_attempt_limit,
            attempt_window_seconds=settings.demo_request_attempt_window_seconds,
            ban_seconds=settings.demo_request_ban_seconds,
            global_limit=settings.demo_request_global_limit,
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        # Nom du fichier d'un export (E14) et page d'une preuve surlignée,
        # lus par le dashboard.
        expose_headers=[
            "Content-Disposition",
            "X-Evidence-Page",
            "X-Evidence-Highlighted",
        ],
    )
    install_middleware(app)
    install_error_handlers(app)
    app.include_router(system_router)
    app.include_router(v1_router)
    return app


app = create_app()
