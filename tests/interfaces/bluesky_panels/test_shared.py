"""Unit tests for :mod:`osprey.interfaces.bluesky_panels._shared`.

Covers the leaf helpers the three bridge-facing routers share: the fixed 502
unreachable body and ``safe_json``'s defensive parse (valid JSON passes
through; a non-JSON body returns the caller-supplied default, which differs per
route -- ``None`` for the relays, ``{}`` for the launch route).
"""

from __future__ import annotations

import httpx

from osprey.interfaces.bluesky_panels._shared import UNREACHABLE_BODY, safe_json


class TestUnreachableBody:
    def test_shape(self):
        assert UNREACHABLE_BODY == {"detail": "bluesky bridge unreachable"}


class TestSafeJson:
    def test_valid_json_object(self):
        resp = httpx.Response(200, json={"ok": True, "n": 3})
        assert safe_json(resp) == {"ok": True, "n": 3}

    def test_valid_json_list(self):
        resp = httpx.Response(200, json=[1, 2, 3])
        assert safe_json(resp) == [1, 2, 3]

    def test_non_json_returns_default_none(self):
        resp = httpx.Response(200, content=b"not json at all")
        assert safe_json(resp) is None

    def test_non_json_returns_explicit_default(self):
        # The launch route passes {} so it can index the result safely.
        resp = httpx.Response(200, content=b"<html>oops</html>")
        assert safe_json(resp, default={}) == {}

    def test_empty_body_returns_default(self):
        resp = httpx.Response(200, content=b"")
        assert safe_json(resp, default={"fallback": 1}) == {"fallback": 1}

    def test_default_is_returned_verbatim(self):
        sentinel = object()
        resp = httpx.Response(200, content=b"nope")
        assert safe_json(resp, default=sentinel) is sentinel
