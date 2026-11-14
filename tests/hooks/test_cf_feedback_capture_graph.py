"""Feedback capture skips the graph channel-finder pipeline.

The graph pipeline answers from the facility graph and keeps no navigation
hints, so its answers — ``read_cypher``'s ``QueryResult`` envelope and the
``search_channels`` payload — have nothing to be reviewed into.
``osprey_cf_feedback_capture.py`` leaves the pending-review store untouched
for both, while ``run_sql``'s ``rows``/``row_count`` envelope is still captured.
"""

import json

import pytest

from tests.hooks.conftest import pending_review_items

HOOK_NAME = "osprey_cf_feedback_capture.py"
READ_CYPHER_TOOL = "mcp__channel-finder__read_cypher"
SEARCH_CHANNELS_TOOL = "mcp__channel-finder__search_channels"
RUN_SQL_TOOL = "mcp__channel-finder__run_sql"

_CYPHER_ENVELOPE = {
    "columns": ["fullPv", "confidence"],
    "rows": [
        {"fullPv": "GTL:BC1:Setpoint", "confidence": 0.95},
        {"fullPv": "GTL:BC1:Readback", "confidence": 0.95},
    ],
    "row_count": 2,
    "truncated": False,
}

_SEARCH_PAYLOAD = {
    "total": 1,
    "devices": 1,
    "page": 1,
    "pages": 1,
    "rows": [{"fullPv": "GTL:BC1:Setpoint", "device": "BC1", "section": "GTL"}],
    "facets": {"section": [{"value": "GTL", "count": 1}]},
}


def _run(hook_runner, tmp_path, tool, response, query="bend current"):
    hook_runner(
        HOOK_NAME,
        tool,
        {"query": query},
        cwd=tmp_path,
        tool_response=response,
        hook_input_extra={
            "cwd": str(tmp_path),
            "session_id": "graph-session",
            "transcript_path": "/tmp/transcript.jsonl",
        },
    )
    return pending_review_items(tmp_path)


@pytest.mark.parametrize(
    ("tool", "response"),
    [
        (READ_CYPHER_TOOL, _CYPHER_ENVELOPE),
        (READ_CYPHER_TOOL, {"result": json.dumps(_CYPHER_ENVELOPE)}),
        (SEARCH_CHANNELS_TOOL, _SEARCH_PAYLOAD),
        (SEARCH_CHANNELS_TOOL, {"result": json.dumps(_SEARCH_PAYLOAD)}),
    ],
)
def test_graph_answer_is_not_captured(tmp_path, hook_runner, tool, response):
    """A graph answer with results still leaves the store empty."""
    assert not _run(hook_runner, tmp_path, tool, response)


def test_run_sql_envelope_is_still_captured(tmp_path, hook_runner):
    """The same rows/row_count shape from the middle-layer pipeline is reviewed."""
    items = _run(hook_runner, tmp_path, RUN_SQL_TOOL, _CYPHER_ENVELOPE, query="SELECT 1")

    assert len(items) == 1
    item = next(iter(items.values()))
    assert item["tool_name"] == RUN_SQL_TOOL
    assert item["channel_count"] == 2
    assert json.loads(item["tool_response"])["channels"] == [
        "GTL:BC1:Setpoint",
        "GTL:BC1:Readback",
    ]
