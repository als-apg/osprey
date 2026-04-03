"""Web Terminal route package.

Splits the monolithic routes module into domain-specific sub-modules.
The composite ``router`` re-exported here is drop-in compatible with the
previous single-file layout.
"""

from __future__ import annotations

from fastapi import APIRouter

from osprey.interfaces.web_terminal.routes.config import router as config_router
from osprey.interfaces.web_terminal.routes.files import router as files_router
from osprey.interfaces.web_terminal.routes.memory import router as memory_router
from osprey.interfaces.web_terminal.routes.panels import router as panels_router
from osprey.interfaces.web_terminal.routes.prompts import router as prompts_router
from osprey.interfaces.web_terminal.routes.session import router as session_router
from osprey.interfaces.web_terminal.routes.chat import router as chat_router
from osprey.interfaces.web_terminal.routes.websocket import router as websocket_router

router = APIRouter()
router.include_router(panels_router)
router.include_router(session_router)
router.include_router(config_router)
router.include_router(files_router)
router.include_router(memory_router)
router.include_router(prompts_router)
router.include_router(websocket_router)
router.include_router(chat_router)

__all__ = ["router"]
