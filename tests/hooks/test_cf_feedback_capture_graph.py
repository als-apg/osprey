"""Feedback capture for the graph channel-finder pipeline.

The graph pipeline answers with ``read_cypher``'s ``QueryResult`` envelope —
``columns``, ``rows``, ``row_count``, ``truncated`` — or with the
``search_channels`` payload, which counts in ``total`` and lists its hits in
``rows``. The table pipelines answer with ``channels``/``total``. These tests
feed recorded payloads of both graph shapes to
``osprey_cf_feedback_capture.py`` and assert what lands in the pending-review
store, so an operator reviewing a graph answer sees the same card a table
answer produces.
"""

import ast
import json
import re
from pathlib import Path

import pytest
import yaml

from osprey.utils.workspace import DEFAULT_AGENT_DATA_BASE_DIR

HOOK_NAME = "osprey_cf_feedback_capture.py"
READ_CYPHER_TOOL = "mcp__channel-finder__read_cypher"
SEARCH_CHANNELS_TOOL = "mcp__channel-finder__search_channels"

HOOKS_DIR = (
    Path(__file__).parents[2] / "src" / "osprey" / "templates" / "claude_code" / "claude" / "hooks"
)

# -- Recorded payloads --------------------------------------------------------
#
# Shaped exactly as ``read_cypher`` returns them: the payload built in
# ``src/osprey/mcp_server/graph/tools/read_cypher.py`` from a ``QueryResult``.
# The Cypher and the addresses come from the shipped example queries, so the
# column spellings are the ones a real answer carries.

_BINDINGS_CYPHER = (
    "MATCH (d:Resource {sourceName: $name, sectionCode: $section})-[:HASBINDING]->"
    "(b:ChannelBinding)\nRETURN b.fullPv AS fullPv, b.confidence AS confidence\nLIMIT 200"
)

_FULLPV_ENVELOPE = {
    "columns": ["fullPv", "confidence"],
    "rows": [
        {"fullPv": "GTL:BC1:Setpoint", "confidence": 0.95},
        {"fullPv": "GTL:BC1:Readback", "confidence": 0.95},
        {"fullPv": "GTL:BC1:Status", "confidence": 0.8},
    ],
    "row_count": 3,
    "truncated": False,
}

# The example catalogue aliases the address column to ``pv`` (``RETURN
# b.fullPv AS pv``), so an answer to a shipped query arrives spelled that way.
_ALIASED_ENVELOPE = {
    "columns": ["pv", "signal", "direction", "confidence"],
    "rows": [
        {"pv": "GTL:BC1:Setpoint", "signal": "Current", "direction": "write", "confidence": 0.95},
        {"pv": "GTL:BC1:Readback", "signal": "Current", "direction": "read", "confidence": 0.95},
    ],
    "row_count": 2,
    "truncated": False,
}

# A perfectly ordinary graph answer that names no addresses at all.
_NO_ADDRESS_ENVELOPE = {
    "columns": ["sectionCode", "devices"],
    "rows": [{"sectionCode": "GTL", "devices": 42}, {"sectionCode": "SR", "devices": 311}],
    "row_count": 2,
    "truncated": False,
}


# Shaped as ``search_channels`` returns it: a keyword lookup in the search
# index, counted by ``total`` rather than ``row_count``, with one row per
# binding and the facet rails the agent narrows on.
_SEARCH_PAYLOAD = {
    "total": 2,
    "devices": 1,
    "page": 1,
    "pages": 1,
    "rows": [
        {
            "fullPv": "GTL:BC1:Setpoint",
            "description": "Bend current setpoint",
            "device": "BC1",
            "section": "GTL",
            "system": "magnet",
            "direction": "W",
            "signals": [{"uri": "http://example.org/Current", "name": "Current"}],
        },
        {
            "fullPv": "GTL:BC1:Readback",
            "description": "Bend current readback",
            "device": "BC1",
            "section": "GTL",
            "system": "magnet",
            "direction": "R",
            "signals": [{"uri": "http://example.org/Current", "name": "Current"}],
        },
    ],
    "facets": {
        "section": [{"value": "GTL", "count": 2}],
        "system": [{"value": "magnet", "count": 2}],
        "class": [],
        "signal": [{"value": "Current", "count": 2}],
        "dir": [{"value": "R", "count": 1}, {"value": "W", "count": 1}],
    },
}


def _hook_extra(tmp_path, transcript="/tmp/transcript.jsonl"):
    """Standard hook_input_extra with cwd set for store path resolution."""
    return {
        "cwd": str(tmp_path),
        "session_id": "graph-session",
        "transcript_path": transcript,
    }


def _pending_items(tmp_path):
    """Read items from pending_reviews.json, return dict or empty.

    The store lives under the agent-data root, resolved from the same constant
    the hook uses so a relocated ``agent_data.base_dir`` moves both together.
    """
    store = tmp_path / DEFAULT_AGENT_DATA_BASE_DIR / "feedback" / "pending_reviews.json"
    if not store.exists():
        return {}
    return json.loads(store.read_text()).get("items", {})


def _capture(hook_runner, tmp_path, response, tool_input=None, tool=READ_CYPHER_TOOL):
    """Run the hook over one recorded response and return the stored item.

    Returns ``None`` when nothing was captured.
    """
    hook_runner(
        HOOK_NAME,
        tool,
        tool_input if tool_input is not None else {"query": _BINDINGS_CYPHER, "params": {}},
        cwd=tmp_path,
        tool_response=response,
        hook_input_extra=_hook_extra(tmp_path),
    )
    items = _pending_items(tmp_path)
    if not items:
        return None
    assert len(items) == 1, f"Expected exactly one captured item, got: {items}"
    return next(iter(items.values()))


def _stored_response(item):
    """Parse the recorded tool_response back into a dict."""
    raw = item["tool_response"]
    return json.loads(raw) if isinstance(raw, str) else raw


@pytest.mark.unit
class TestReadCypherCapture:
    """A ``QueryResult`` envelope reaches the store with its counts intact."""

    def test_envelope_is_captured_with_row_count(self, tmp_path, hook_runner):
        """``row_count`` drives the capture the way ``total`` does elsewhere."""
        item = _capture(hook_runner, tmp_path, _FULLPV_ENVELOPE)

        assert item is not None, "read_cypher result was not captured"
        assert item["tool_name"] == READ_CYPHER_TOOL
        assert item["channel_count"] == 3
        assert item["query"] == _BINDINGS_CYPHER

    def test_envelope_survives_the_round_trip(self, tmp_path, hook_runner):
        """The rows an operator has to judge are stored, not just their count."""
        stored = _stored_response(_capture(hook_runner, tmp_path, _FULLPV_ENVELOPE))

        assert stored["rows"] == _FULLPV_ENVELOPE["rows"]
        assert stored["row_count"] == 3
        assert stored["truncated"] is False
        assert stored["columns"] == ["fullPv", "confidence"]

    def test_addresses_come_from_the_fullpv_column(self, tmp_path, hook_runner):
        """The review UI reads ``channels``; graph answers fill it from ``fullPv``."""
        stored = _stored_response(_capture(hook_runner, tmp_path, _FULLPV_ENVELOPE))

        assert stored["channels"] == [
            "GTL:BC1:Setpoint",
            "GTL:BC1:Readback",
            "GTL:BC1:Status",
        ]

    def test_addresses_come_from_the_aliased_pv_column(self, tmp_path, hook_runner):
        """The shipped example queries alias the address column to ``pv``."""
        stored = _stored_response(_capture(hook_runner, tmp_path, _ALIASED_ENVELOPE))

        assert stored["channels"] == ["GTL:BC1:Setpoint", "GTL:BC1:Readback"]

    def test_repeated_address_is_listed_once(self, tmp_path, hook_runner):
        """One PV bound to several devices is one address, on as many rows."""
        response = {
            "columns": ["fullPv", "sourceName"],
            "rows": [
                {"fullPv": "SR:MAG:BEND:CURRENT", "sourceName": "BEND01"},
                {"fullPv": "SR:MAG:BEND:CURRENT", "sourceName": "BEND02"},
            ],
            "row_count": 2,
            "truncated": False,
        }
        item = _capture(hook_runner, tmp_path, response)

        assert item["channel_count"] == 2, "the row count is the row count"
        assert _stored_response(item)["channels"] == ["SR:MAG:BEND:CURRENT"]

    def test_answer_without_an_address_column_is_still_captured(self, tmp_path, hook_runner):
        """A count-by-section answer names no PVs and is still worth reviewing."""
        item = _capture(hook_runner, tmp_path, _NO_ADDRESS_ENVELOPE)

        assert item is not None
        assert item["channel_count"] == 2
        stored = _stored_response(item)
        assert stored["rows"] == _NO_ADDRESS_ENVELOPE["rows"]
        assert stored.get("channels", []) == []

    def test_truncated_answer_keeps_its_flag_and_guidance(self, tmp_path, hook_runner):
        """A clipped answer must not read as a complete one in review."""
        response = {
            "columns": ["fullPv"],
            "rows": [{"fullPv": f"SR:BPM{i:02d}:X"} for i in range(200)],
            "row_count": 200,
            "truncated": True,
            "guidance": [
                "Result was cut at services.graphdb.query_max_rows = 200 rows; "
                "add a LIMIT, aggregate, or narrow the MATCH."
            ],
        }
        stored = _stored_response(_capture(hook_runner, tmp_path, response))

        assert stored["truncated"] is True
        assert stored["row_count"] == 200
        assert stored["guidance"]
        assert len(stored["channels"]) == 200

    def test_post_tool_use_result_wrapper_is_unwrapped(self, tmp_path, hook_runner):
        """PostToolUse delivers ``{"result": "<json>"}`` around the envelope."""
        wrapped = {"result": json.dumps(_FULLPV_ENVELOPE)}
        item = _capture(hook_runner, tmp_path, wrapped)

        assert item is not None
        assert item["channel_count"] == 3
        assert _stored_response(item)["channels"][0] == "GTL:BC1:Setpoint"


@pytest.mark.unit
class TestReadCypherNotCaptured:
    """Nothing to review means nothing in the store."""

    def test_empty_result_set_is_not_captured(self, tmp_path, hook_runner):
        """``row_count: 0`` is an answer with nothing for an operator to judge."""
        response = {"columns": [], "rows": [], "row_count": 0, "truncated": False}

        assert _capture(hook_runner, tmp_path, response) is None

    def test_error_envelope_is_not_captured(self, tmp_path, hook_runner):
        """A refused or failed query carries neither ``row_count`` nor rows."""
        response = {
            "error": True,
            "error_type": "validation_error",
            "error_message": "Only read queries are allowed.",
            "suggestions": ["Rewrite the query as a MATCH ... RETURN."],
        }

        assert _capture(hook_runner, tmp_path, response) is None


@pytest.mark.unit
class TestSearchChannelsCapture:
    """The index answer is reviewed on the card the Cypher answer produces."""

    def _capture_search(self, hook_runner, tmp_path, response, tool_input=None):
        return _capture(
            hook_runner,
            tmp_path,
            response,
            tool_input=tool_input if tool_input is not None else {"query": "bend current"},
            tool=SEARCH_CHANNELS_TOOL,
        )

    def test_payload_is_captured_with_its_total(self, tmp_path, hook_runner):
        """``search_channels`` counts in ``total``, the way table answers do."""
        item = self._capture_search(hook_runner, tmp_path, _SEARCH_PAYLOAD)

        assert item is not None, "search_channels result was not captured"
        assert item["tool_name"] == SEARCH_CHANNELS_TOOL
        assert item["channel_count"] == 2
        assert item["query"] == "bend current"

    def test_rows_and_facets_survive_the_round_trip(self, tmp_path, hook_runner):
        """An operator judges the hits and the rails, not just the count."""
        stored = _stored_response(self._capture_search(hook_runner, tmp_path, _SEARCH_PAYLOAD))

        assert stored["rows"] == _SEARCH_PAYLOAD["rows"]
        assert stored["facets"] == _SEARCH_PAYLOAD["facets"]
        assert stored["total"] == 2

    def test_addresses_come_from_the_fullpv_column(self, tmp_path, hook_runner):
        """The review UI reads ``channels``; index rows fill it from ``fullPv``."""
        stored = _stored_response(self._capture_search(hook_runner, tmp_path, _SEARCH_PAYLOAD))

        assert stored["channels"] == ["GTL:BC1:Setpoint", "GTL:BC1:Readback"]

    def test_post_tool_use_result_wrapper_is_unwrapped(self, tmp_path, hook_runner):
        """The tool returns a JSON string, wrapped the way PostToolUse wraps it."""
        wrapped = {"result": json.dumps(_SEARCH_PAYLOAD)}
        item = self._capture_search(hook_runner, tmp_path, wrapped)

        assert item is not None
        assert item["channel_count"] == 2
        assert _stored_response(item)["channels"][0] == "GTL:BC1:Setpoint"

    def test_no_match_is_not_captured(self, tmp_path, hook_runner):
        """A phrase that matched nothing gives an operator nothing to judge."""
        response = dict(_SEARCH_PAYLOAD, total=0, devices=0, rows=[], pages=0)

        assert self._capture_search(hook_runner, tmp_path, response) is None


@pytest.mark.unit
class TestHookDeclaresItsGraphTools:
    """The gate and the front matter have to name a tool, or nothing fires."""

    def _hook_source(self):
        return (HOOKS_DIR / HOOK_NAME).read_text()

    @pytest.mark.parametrize("tool", [READ_CYPHER_TOOL, SEARCH_CHANNELS_TOOL])
    def test_front_matter_lists_the_tool(self, tool):
        """The declared tool list is what a reader consults to know the scope."""
        docstring = ast.get_docstring(ast.parse(self._hook_source()))
        match = re.match(r"^---\n(.*?)\n---\n", docstring, re.DOTALL)
        assert match, "hook docstring has no YAML front matter"
        fields = yaml.safe_load(match.group(1))

        declared = [entry.strip() for entry in str(fields["tools"]).split(",")]
        assert tool in declared, f"front matter tools: {declared}"

    @pytest.mark.parametrize("tool", [READ_CYPHER_TOOL, SEARCH_CHANNELS_TOOL])
    def test_capture_tool_set_lists_the_tool(self, tool):
        """The runtime gate, not just the documentation, has to know the tool."""
        assert f'"{tool}"' in self._hook_source()
