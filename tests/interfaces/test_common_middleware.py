"""Tests for shared FastAPI middleware (cache-control + exception logging)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.common_middleware import (
    ExceptionLoggingMiddleware,
    NoCacheStaticMiddleware,
)


@pytest.fixture
def cache_client():
    app = FastAPI()
    app.add_middleware(NoCacheStaticMiddleware)

    @app.get("/static/vendor/plotly-3.3.1.min.js")
    async def vendored():
        return {"ok": True}

    @app.get("/static/app.js")
    async def static_asset():
        return {"ok": True}

    @app.get("/api/thing")
    async def api_thing():
        return {"ok": True}

    @app.get("/api/vendor/lib.js")
    async def vendor_outside_static():
        return {"ok": True}

    @app.get("/other")
    async def other():
        return {"ok": True}

    return TestClient(app)


class TestNoCacheStaticMiddleware:
    def test_vendored_asset_is_immutable(self, cache_client):
        resp = cache_client.get("/static/vendor/plotly-3.3.1.min.js")
        assert resp.headers["Cache-Control"] == "public, max-age=31536000, immutable"

    def test_non_vendor_static_is_uncached(self, cache_client):
        resp = cache_client.get("/static/app.js")
        assert resp.headers["Cache-Control"] == "no-cache, no-store, must-revalidate"

    def test_api_path_is_uncached(self, cache_client):
        resp = cache_client.get("/api/thing")
        assert resp.headers["Cache-Control"] == "no-cache, no-store, must-revalidate"

    def test_unrelated_path_gets_no_cache_header(self, cache_client):
        resp = cache_client.get("/other")
        assert "Cache-Control" not in resp.headers

    def test_vendor_outside_static_is_not_immutable(self, cache_client):
        """The immutable rule requires BOTH /vendor/ and a /static/ prefix; a
        /vendor/ path served from /api falls through to the no-cache branch."""
        resp = cache_client.get("/api/vendor/lib.js")
        assert resp.headers["Cache-Control"] == "no-cache, no-store, must-revalidate"


@pytest.fixture
def exc_client():
    app = FastAPI()
    app.add_middleware(ExceptionLoggingMiddleware)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("kaboom")

    @app.get("/fine")
    async def fine():
        return {"ok": True}

    # raise_server_exceptions=False so the middleware's own 500 response is
    # observed rather than TestClient re-raising before we can inspect it.
    return TestClient(app, raise_server_exceptions=False)


class TestExceptionLoggingMiddleware:
    def test_unhandled_exception_becomes_structured_500(self, exc_client):
        resp = exc_client.get("/boom")
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"] == "kaboom"
        assert body["path"] == "/boom"

    def test_successful_request_passes_through(self, exc_client):
        resp = exc_client.get("/fine")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}
