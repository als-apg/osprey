"""Shared middleware for OSPREY FastAPI applications."""

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("osprey.interfaces.middleware")


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """Prevent browsers from caching static assets and API responses.

    Avoids stale code/config after updates by setting Cache-Control headers
    on all responses for paths under /static/ and /api/.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/static/") or path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


class ExceptionLoggingMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions — log traceback + return structured JSON 500."""

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            logger.error("Unhandled exception on %s %s", request.method, request.url.path, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc), "path": request.url.path},
            )
