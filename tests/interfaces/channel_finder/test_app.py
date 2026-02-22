"""Tests for Channel Finder FastAPI app factory."""

from __future__ import annotations

import pytest


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "channel-finder"

    def test_health_includes_pipeline_type(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["pipeline_type"] == "in_context"


class TestRootEndpoint:
    """Tests for the root / endpoint."""

    def test_root_serves_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
