from fastapi import APIRouter

from app.modules.approaches.router import router as approaches_router
from app.modules.frameworks.router import router as frameworks_router

router = APIRouter(prefix="/api/v1")
router.include_router(approaches_router)
router.include_router(frameworks_router)
