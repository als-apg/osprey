"""Tests for shared FastAPI middleware (cache-control, exception logging, owner stamp)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.common_middleware import (
    ExceptionLoggingMiddleware,
    NoCacheStaticMiddleware,
    _stamp_owner,
)
from osprey.utils.owner_header import OWNER_HEADER

#: The owner header's wire name as an ASGI scope carries it. Derived from the
#: shared spelling rather than retyped, so a rename cannot leave the
#: assertions below watching a header nothing mints.
OWNER_WIRE_NAME = OWNER_HEADER.lower().encode("latin-1")


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


class TestStampOwner:
    """Unit contract for ``_stamp_owner``, the gate's one owner-minting site.

    The middleware's admit paths are driven over raw ASGI scopes in
    ``test_auth_middleware.py``; what is pinned here is the scope surgery
    itself, whose two halves — drop every inbound value, append only a minted
    one — are what make the gate a trust boundary for attribution rather than a
    relay of whatever a browser claimed.
    """

    @staticmethod
    def _owners(scope) -> list[bytes]:
        """Every owner-header value left on ``scope``, in wire order."""
        return [value for name, value in scope["headers"] if name.lower() == OWNER_WIRE_NAME]

    def test_a_named_account_replaces_every_inbound_claim(self):
        scope = {
            "type": "http",
            "headers": [
                (b"x-osprey-owner", b"bob"),
                (b"X-Osprey-Owner", b"carol"),
                (b"host", b"localhost:8080"),
            ],
        }

        _stamp_owner(scope, "op1")

        # Exactly one value, whatever the request carried and however it spelled
        # it: a reader taking the first occurrence would otherwise be steerable
        # by a forged header placed ahead of the minted one.
        assert self._owners(scope) == [b"op1"]
        assert (b"host", b"localhost:8080") in scope["headers"]
        assert scope["state"]["osprey_owner"] == "op1"

    def test_nobody_drops_the_claim_and_leaves_the_state_unset(self):
        """The sidecar's own door: authorised, attributable to no one."""
        scope = {
            "type": "http",
            "headers": [(b"x-osprey-owner", b"bob")],
            "state": {"osprey_owner": "stale"},
        }

        _stamp_owner(scope, None)

        assert self._owners(scope) == []
        # Unset rather than blank: a reader tells "nobody" from a name by the
        # key's absence, so a value left behind would be read as an attribution.
        assert "osprey_owner" not in scope["state"]

    def test_an_empty_account_names_nobody(self):
        scope = {"type": "http", "headers": [(b"x-osprey-owner", b"bob")]}

        _stamp_owner(scope, "")

        assert self._owners(scope) == []
        assert "osprey_owner" not in scope["state"]

    def test_a_scope_without_state_gets_one(self):
        """ASGI leaves ``state`` optional; Starlette's ``request.state`` reads it."""
        scope = {"type": "http", "headers": []}

        _stamp_owner(scope, "op1")

        assert scope["state"] == {"osprey_owner": "op1"}

    def test_a_websocket_scope_is_stamped_too(self):
        scope = {"type": "websocket", "headers": [(b"x-osprey-owner", b"bob")]}

        _stamp_owner(scope, "alice")

        assert self._owners(scope) == [b"alice"]
        assert scope["state"]["osprey_owner"] == "alice"

    def test_an_account_outside_latin_1_is_mangled_rather_than_raised(self):
        """Encoding may not fail: a 500 here would refuse a valid credential.

        The mangled value fails the reader's charset guard, so such a request
        is recorded owner-less — the one failure this path is allowed to have.
        """
        scope = {"type": "http", "headers": []}

        _stamp_owner(scope, "小明")

        assert len(self._owners(scope)) == 1
        assert scope["state"]["osprey_owner"] == "小明"
