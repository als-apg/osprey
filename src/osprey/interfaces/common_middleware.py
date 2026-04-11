"""Shared middleware for OSPREY FastAPI applications."""

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger("osprey.interfaces.middleware")


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """Control browser caching for static assets and API responses.

    Vendor assets (versioned filenames like plotly-3.3.1.min.js) are cached
    aggressively — they never change without a filename bump.  All other
    static/API paths are uncached to avoid stale code after updates.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if "/vendor/" in path and path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif path.startswith("/static/") or path.startswith("/api/"):
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
