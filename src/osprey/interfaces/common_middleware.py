"""Shared middleware for OSPREY FastAPI applications."""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


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
