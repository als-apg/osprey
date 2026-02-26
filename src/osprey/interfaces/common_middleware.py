"""Shared middleware for OSPREY FastAPI applications."""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """Prevent browsers from caching static assets (JS/CSS).

    Avoids stale code after updates by setting Cache-Control headers
    on all responses for paths under /static/.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response
