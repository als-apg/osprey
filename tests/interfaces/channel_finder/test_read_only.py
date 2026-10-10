"""The Channel Finder web API is read-only.

No route mutates a channel database or previews the impact of a mutation. A
request to a path that once carried one is answered by the router itself: 405
when the path still serves another method, 404 otherwise. Validation and the
pipeline switch are not database writes and keep answering.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from osprey.services.channel_finder.databases import FlatChannelDatabase

_DB_PATCH = "osprey.interfaces.channel_finder.database_api._get_database"

#: (method, path, paradigm that would own the route, expected status).
_ABSENT_ROUTES = [
    ("post", "/api/channels", "in_context", 405),
    ("put", "/api/channels/SR:BPM:01:X", "in_context", 404),
    ("delete", "/api/channels/SR:BPM:01:X", "in_context", 404),
    ("post", "/api/tree/node", "hierarchical", 404),
    ("put", "/api/tree/node", "hierarchical", 404),
    ("delete", "/api/tree/node", "hierarchical", 404),
    ("post", "/api/tree/impact", "hierarchical", 404),
    ("get", "/api/tree/expansion", "hierarchical", 404),
    ("put", "/api/tree/expansion", "hierarchical", 404),
    ("post", "/api/structure/family", "middle_layer", 404),
    ("delete", "/api/structure/family", "middle_layer", 404),
    ("post", "/api/structure/channel", "middle_layer", 404),
    ("delete", "/api/structure/channel", "middle_layer", 404),
    ("post", "/api/structure/impact", "middle_layer", 404),
]


@pytest.mark.parametrize(("method", "path", "paradigm", "expected"), _ABSENT_ROUTES)
def test_no_write_or_impact_route_answers(client, method, path, paradigm, expected):
    # The paradigm the route would serve is active, so a paradigm gate cannot be
    # what answers; an empty database roster would answer 503 from any live route.
    client.app.state.pipeline_type = paradigm
    client.app.state.databases = {}
    resp = client.request(method.upper(), path, json={})
    assert resp.status_code == expected
    assert resp.json()["detail"] in ("Not Found", "Method Not Allowed")


def test_validate_still_answers(client):
    client.app.state.pipeline_type = "in_context"
    mock_db = MagicMock(spec=FlatChannelDatabase)
    mock_db.validate_channels.return_value = [{"channel": "SR:BPM:01:X", "valid": True}]
    mock_db.get_valid_channels.return_value = ["SR:BPM:01:X"]
    mock_db.get_invalid_channels.return_value = []
    with patch(_DB_PATCH, return_value=mock_db):
        resp = client.post("/api/validate", json={"channels": ["SR:BPM:01:X"]})
    assert resp.status_code == 200
    assert resp.json()["valid_count"] == 1


def test_pipeline_switch_still_answers(client):
    client.app.state.pipeline_type = "in_context"
    client.app.state.available_pipelines = ["in_context", "hierarchical"]
    resp = client.put("/api/pipeline", json={"pipeline_type": "hierarchical"})
    assert resp.status_code == 200
    assert resp.json() == {"pipeline_type": "hierarchical"}
