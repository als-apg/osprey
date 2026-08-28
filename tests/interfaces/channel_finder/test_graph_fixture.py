"""The graph-mode launcher boots a real server over the shared demo corpus.

The browser and visual lanes reach the graph UI through
``launch_graph_channel_finder``, and when that launcher is wrong they fail as a
photograph that does not match or a page that renders the unconfigured shell —
symptoms several layers away from the cause. This test asserts the launcher's
own contract instead, without a browser: the app it serves really is in graph
mode, and the ontology route really is answering from the demo corpus.
"""

from __future__ import annotations

import httpx

from tests.interfaces.channel_finder.graph_fixture import (
    DEMO_CLASS_COUNT,
    DEMO_RELATIONSHIP_TYPES,
)
from tests.interfaces.conftest import launch_graph_channel_finder


class TestGraphLauncher:
    """What a lane gets when it boots the Channel Finder through the launcher."""

    def test_served_app_reports_the_graph_paradigm(self, tmp_path, monkeypatch):
        with launch_graph_channel_finder(tmp_path, monkeypatch) as base_url:
            info = httpx.get(f"{base_url}/api/info", timeout=10.0)

        assert info.status_code == 200
        payload = info.json()
        assert payload["pipeline_type"] == "graph"
        # Store-backed rather than file-backed: the paradigm is the answer, and
        # there is no database path to name.
        assert payload["graph_backed"] is True
        assert payload["db_path"] is None

    def test_ontology_route_answers_from_the_demo_corpus(self, tmp_path, monkeypatch):
        with launch_graph_channel_finder(tmp_path, monkeypatch) as base_url:
            resp = httpx.get(f"{base_url}/api/graph/ontology", timeout=10.0)

        assert resp.status_code == 200
        data = resp.json()
        # The fake store is installed and reachable: the seeded taxonomy comes
        # back whole rather than the empty-store or unavailable answer.
        assert len(data["classes"]) == DEMO_CLASS_COUNT
        assert data["relationship_types"] == DEMO_RELATIONSHIP_TYPES
        assert data["empty"] is False
