"""Tests for the shared hop-by-hop header set used by OSPREY's reverse proxies.

``HOP_BY_HOP`` is imported by both the web-terminal panel proxy and the Bluesky
panels SSE relay so the two never drift. The contract those callers rely on:
the set is a frozenset of lowercased header names, so a caller can test
``header.lower() in HOP_BY_HOP`` and strip it before re-emitting an upstream
response downstream.
"""

from __future__ import annotations

from osprey.utils.http_proxy import HOP_BY_HOP


class TestHopByHop:
    def test_is_frozenset(self):
        """Immutable so neither caller can mutate the shared set."""
        assert isinstance(HOP_BY_HOP, frozenset)

    def test_all_entries_lowercase(self):
        """Callers match with ``header.lower() in HOP_BY_HOP``."""
        assert all(h == h.lower() for h in HOP_BY_HOP)

    def test_contains_rfc7230_hop_by_hop_headers(self):
        rfc_headers = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailers",
            "transfer-encoding",
            "upgrade",
        }
        assert rfc_headers <= HOP_BY_HOP

    def test_contains_content_framing_headers(self):
        """A reverse proxy recomputes these two, so they must be stripped."""
        assert "content-encoding" in HOP_BY_HOP
        assert "content-length" in HOP_BY_HOP

    def test_excludes_end_to_end_headers(self):
        """End-to-end headers must pass through untouched."""
        for header in ("content-type", "authorization", "cache-control", "etag", "date"):
            assert header not in HOP_BY_HOP

    def test_exact_membership(self):
        """Pin the full set so a drift in either caller's copy is caught."""
        assert HOP_BY_HOP == frozenset(
            {
                "connection",
                "keep-alive",
                "proxy-authenticate",
                "proxy-authorization",
                "te",
                "trailers",
                "transfer-encoding",
                "upgrade",
                "content-encoding",
                "content-length",
            }
        )
