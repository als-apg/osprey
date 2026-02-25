"""OSPREY Tuning Panel — FastAPI Application.

A lightweight proxy service that serves the OSPREY-styled tuning frontend
and reverse-proxies API calls to the Tuning Scripts backend, avoiding CORS
issues since all requests go through the same origin.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = __import__("logging").getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def _resolve_tuning_api_url(explicit: str | None = None) -> str | None:
    """Resolve the Tuning Scripts backend URL from arg, env, or config."""
    if explicit:
        return explicit

    env_url = os.environ.get("TUNING_API_URL")
    if env_url:
        return env_url

    try:
        from osprey.mcp_server.common import load_osprey_config

        config = load_osprey_config()
        return config.get("tuning", {}).get("api_url")
    except Exception:
        return None


def _create_lifespan(tuning_api_url: str | None = None):
    """Create a lifespan context manager that initializes the httpx client."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        import httpx

        app.state.tuning_api_url = _resolve_tuning_api_url(tuning_api_url)
        app.state.http_client = httpx.AsyncClient(timeout=30.0)

        if app.state.tuning_api_url:
            logger.info("Tuning API backend: %s", app.state.tuning_api_url)
        else:
            logger.warning("No tuning API URL configured — proxy will return 503")

        yield

        await app.state.http_client.aclose()

    return lifespan


def create_app(tuning_api_url: str | None = None) -> FastAPI:
    """Create the Tuning Panel FastAPI application.

    Args:
        tuning_api_url: URL of the Tuning Scripts backend to proxy to.
            Falls back to TUNING_API_URL env var or config.tuning.api_url.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(
        title="OSPREY Tuning Panel",
        description="Bayesian optimization interface for accelerator tuning",
        version="1.0.0",
        lifespan=_create_lifespan(tuning_api_url),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"status": "healthy", "service": "tuning"}

    @app.get("/")
    async def root():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/config")
    async def api_config(request: Request):
        """Return the tuning API URL so the JS frontend knows where calls go."""
        api_url = getattr(request.app.state, "tuning_api_url", None)
        return {"api_url": api_url, "proxy_enabled": api_url is not None}

    @app.api_route("/api/proxy/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy(path: str, request: Request):
        """Reverse-proxy requests to the Tuning Scripts backend."""
        api_url = getattr(request.app.state, "tuning_api_url", None)
        if not api_url:
            return Response(
                content='{"error": "Tuning API URL not configured"}',
                status_code=503,
                media_type="application/json",
            )

        client = request.app.state.http_client
        target_url = f"{api_url.rstrip('/')}/{path}"

        # Forward query params
        if request.url.query:
            target_url = f"{target_url}?{request.url.query}"

        # Forward headers (skip hop-by-hop)
        skip_headers = {"host", "connection", "transfer-encoding"}
        headers = {k: v for k, v in request.headers.items() if k.lower() not in skip_headers}

        body = await request.body()

        try:
            resp = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body if body else None,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                media_type=resp.headers.get("content-type", "application/json"),
            )
        except Exception as exc:
            logger.warning("Proxy error for %s: %s", target_url, exc)
            return Response(
                content=f'{{"error": "Proxy error: {exc}"}}',
                status_code=502,
                media_type="application/json",
            )

    # Prevent browsers from caching JS/CSS (avoids stale code after updates)
    from starlette.middleware.base import BaseHTTPMiddleware

    class NoCacheStaticMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            if request.url.path.startswith("/static/"):
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return response

    app.add_middleware(NoCacheStaticMiddleware)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
