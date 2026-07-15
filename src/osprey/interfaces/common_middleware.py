"""Shared middleware for OSPREY FastAPI applications."""

from __future__ import annotations

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger("osprey.interfaces.middleware")


def normalize_base_path(value: str | None) -> str:
    """Normalize a URL base path to ``""`` or ``/segment[/segment...]``.

    Ensures a single leading slash, strips any trailing slash, and treats
    ``None``, ``""`` and ``"/"`` as "no base path" (root). Examples::

        None        -> ""
        ""          -> ""
        "/"         -> ""
        "user/a"    -> "/user/a"
        "/user/a/"  -> "/user/a"
    """
    if not value:
        return ""
    trimmed = value.strip().strip("/")
    return f"/{trimmed}" if trimmed else ""


class BasePathMiddleware:
    """Serve the app under a URL prefix (base path) behind a reverse proxy.

    Strips the prefix from the incoming request path when it is present, so the
    app's routes and static mounts (declared at ``/``, ``/api``, ``/static``,
    …) match whether or not the upstream proxy already stripped it. This keeps a
    single ``osprey web --base-path /user/a`` working across proxy setups that
    strip the location prefix (``proxy_pass http://backend/;``), those that
    forward the full path, and direct local access.

    When no ``base_path`` is configured, an ``X-Forwarded-Prefix`` request
    header is honored as a fallback source of the prefix. A no-op when neither
    is present, so root-served deployments are unaffected.

    Note: this deliberately does *not* set ``scope['root_path']`` — doing so
    breaks ``StaticFiles`` mount resolution in the pinned Starlette. The prefix
    the frontend needs for generating URLs is exposed separately via
    ``app.state.base_path`` (see the web-terminal app factory).
    """

    def __init__(self, app: ASGIApp, base_path: str = "") -> None:
        self.app = app
        self.base_path = normalize_base_path(base_path)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        prefix = self.base_path
        if not prefix:
            for name, value in scope.get("headers") or []:
                if name == b"x-forwarded-prefix":
                    prefix = normalize_base_path(value.decode("latin-1"))
                    break

        if prefix:
            path = scope.get("path", "")
            if path == prefix or path.startswith(prefix + "/"):
                scope = dict(scope)
                scope["path"] = path[len(prefix) :] or "/"
                raw = scope.get("raw_path")
                if raw:
                    praw = prefix.encode("latin-1")
                    if raw.startswith(praw):
                        scope["raw_path"] = raw[len(praw) :] or b"/"

        await self.app(scope, receive, send)


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
            logger.error(
                "Unhandled exception on %s %s", request.method, request.url.path, exc_info=True
            )
            return JSONResponse(
                status_code=500,
                content={"error": str(exc), "path": request.url.path},
            )
