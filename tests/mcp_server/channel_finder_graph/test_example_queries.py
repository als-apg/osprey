"""Unit tests for the graph channel finder's ``example_queries`` tool.

The tool serves this paradigm's own catalogue and nothing else, so the payload is
asserted directly through the unwrapped function. Two properties carry the weight:

* **Every example is runnable as shipped.** Each arrives with a parameter set
  for the shipped corpus that covers every placeholder its Cypher names, and no
  set is ever empty — an empty set would read as "runs there, takes no values",
  which is the one reading that produces a silent zero-row answer.
* **The tool answers with no store in reach.** It is a prompt provider, not a
  health check: it must serve while the graph is down, and a fresh interpreter
  plus a poisoned ``server_context`` module prove it never dials.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from osprey.mcp_server.channel_finder_graph.tools.examples_data import EXAMPLE_QUERIES

pytestmark = pytest.mark.unit

_SERVER_CONTEXT_MODULE = "osprey.mcp_server.graph.server_context"

_EXAMPLE_FIELDS = {"key", "title", "description", "cypher", "parameters"}

#: The curated keys in the order the tool must serve them. A literal, so a
#: reordered or dropped example fails here rather than quietly changing the
#: reading order the descriptions cross-reference.
_EXPECTED_KEYS = (
    "by_description",
    "by_field_meaning",
    "by_class_and_signal",
    "by_family_role",
    "by_system",
    "by_synonym",
    "census",
    "in_section",
    "device_addresses",
    "by_address",
)


def _tool_names(mcp) -> set[str]:
    """Tool names registered on a FastMCP server, across FastMCP versions."""
    tools = mcp.get_tools() if hasattr(mcp, "get_tools") else mcp.list_tools()
    if asyncio.iscoroutine(tools):
        tools = asyncio.run(tools)
    if isinstance(tools, dict):
        return set(tools)
    return {t.name for t in tools}


def _run_probe(source: str) -> subprocess.CompletedProcess:
    """Run a probe in a fresh interpreter with this worktree's ``src`` on the path."""
    from osprey.mcp_server.channel_finder_graph import server as cf_graph_server

    src = Path(cf_graph_server.__file__).parents[3]
    env = {**os.environ, "PYTHONPATH": str(src) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture()
def payload() -> dict:
    """The served catalogue, parsed."""
    from osprey.mcp_server.channel_finder_graph.tools import example_queries as mod

    fn = getattr(mod.example_queries, "fn", mod.example_queries)
    return json.loads(fn())


class TestRegistration:
    """The tool must be reachable under its documented name, on this server."""

    def test_example_queries_lands_on_the_channel_finder_server(self):
        from osprey.mcp_server.channel_finder_graph import server as cf_graph_server
        from osprey.mcp_server.channel_finder_graph.tools import example_queries  # noqa: F401

        assert "example_queries" in _tool_names(cf_graph_server.mcp)

    def test_the_tool_is_this_paradigms_own_not_the_main_agents(self):
        """A re-registration of the main-agent tool would serve the wrong catalogue."""
        from osprey.mcp_server.channel_finder_graph.tools import example_queries as cf_mod
        from osprey.mcp_server.graph.tools import example_queries as graph_mod

        cf_fn = getattr(cf_mod.example_queries, "fn", cf_mod.example_queries)
        graph_fn = getattr(graph_mod.example_queries, "fn", graph_mod.example_queries)

        assert cf_fn is not graph_fn


class TestPayload:
    """The catalogue the agent reads before writing Cypher."""

    def test_payload_has_the_documented_top_level_keys(self, payload):
        assert set(payload) == {"count", "examples", "notes"}
        assert payload["notes"]

    def test_payload_serves_the_whole_curated_set(self, payload):
        assert payload["count"] == len(EXAMPLE_QUERIES) == len(payload["examples"])

    def test_payload_keeps_the_curated_order(self, payload):
        assert tuple(example["key"] for example in payload["examples"]) == _EXPECTED_KEYS
        assert tuple(example.key for example in EXAMPLE_QUERIES) == _EXPECTED_KEYS

    def test_entries_have_the_documented_shape(self, payload):
        for example in payload["examples"]:
            assert set(example) == _EXAMPLE_FIELDS, f"{example.get('key')} has unexpected fields"
            for field in ("key", "title", "description", "cypher"):
                assert isinstance(example[field], str) and example[field].strip(), (
                    f"{example['key']}.{field} is empty"
                )
            assert isinstance(example["parameters"], dict)

    def test_every_parameter_set_is_rendered_verbatim(self, payload):
        """The tool serialises the catalogue; it does not paraphrase it."""
        by_key = {example.key: example for example in EXAMPLE_QUERIES}

        for rendered in payload["examples"]:
            source = by_key[rendered["key"]]
            assert rendered["parameters"] == source.parameters
            assert rendered["cypher"] == source.cypher
            assert rendered["title"] == source.title
            assert rendered["description"] == source.description


class TestParameterContract:
    """Every example carries a real parameter set that runs it."""

    def test_no_rendered_parameter_set_is_empty(self, payload):
        """An empty set reads as "runs here, takes no values" — the wrong answer."""
        for example in payload["examples"]:
            assert example["parameters"], f"{example['key']} renders an empty parameter set"

    def test_the_parameter_set_supplies_every_placeholder(self, payload):
        """Rendering must not drop a value the query needs."""
        for example in payload["examples"]:
            placeholders = set(re.findall(r"\$(\w+)", example["cypher"]))
            assert placeholders == set(example["parameters"]), (
                f"{example['key']}: placeholders {sorted(placeholders)} "
                f"vs parameters {sorted(example['parameters'])}"
            )

    def test_every_query_is_bounded(self, payload):
        for example in payload["examples"]:
            assert "LIMIT" in example["cypher"].upper(), f"{example['key']} has no LIMIT"


class TestNotes:
    """The notes are how the agent knows what to do with a parameter set."""

    def test_notes_explain_how_to_run_an_example(self, payload):
        notes = " ".join(payload["notes"])

        assert "params" in notes
        assert "demo" in notes
        assert "LIMIT" in notes
        assert "services.graphdb.query_max_rows" in notes

    def test_notes_warn_that_a_prose_less_corpus_returns_no_rows(self, payload):
        """Zero rows from a search-by-meaning example must not read as an empty machine."""
        notes = " ".join(payload["notes"])

        assert "no rows" in notes
        assert "get_schema" in notes

    def test_payload_round_trips_json(self, payload):
        assert json.loads(json.dumps(payload)) == payload


class TestAnswersWithNoStore:
    """Pure data serving — the tool is a prompt provider, not a health check."""

    def test_the_call_never_imports_the_server_context(self):
        """A fresh interpreter: calling the tool must not pull in server_context."""
        probe = (
            "import sys\n"
            "from osprey.mcp_server.channel_finder_graph.tools import example_queries as mod\n"
            "fn = getattr(mod.example_queries, 'fn', mod.example_queries)\n"
            "payload = fn()\n"
            "assert payload, 'tool returned an empty payload'\n"
            f"assert {_SERVER_CONTEXT_MODULE!r} not in sys.modules, (\n"
            "    'server_context leaked into sys.modules'\n"
            ")\n"
            "print('OK')\n"
        )

        result = _run_probe(probe)

        assert result.returncode == 0, (
            f"calling example_queries reached the server context:\n{result.stderr}"
        )

    def test_the_call_never_touches_a_loaded_server_context(self, monkeypatch):
        """With server_context already imported, the call must not touch it either."""

        class _Poison(types.ModuleType):
            def __getattr__(self, name: str):
                raise AssertionError(
                    f"example_queries touched {_SERVER_CONTEXT_MODULE}.{name}; "
                    "it must serve curated data without the store"
                )

        monkeypatch.setitem(sys.modules, _SERVER_CONTEXT_MODULE, _Poison(_SERVER_CONTEXT_MODULE))

        from osprey.mcp_server.channel_finder_graph.tools import example_queries as mod

        fn = getattr(mod.example_queries, "fn", mod.example_queries)
        payload = json.loads(fn())

        assert payload["count"] == len(EXAMPLE_QUERIES)

    def test_an_unconfigured_context_still_answers(self, tmp_path, monkeypatch):
        """A deployment with no ``services.graphdb`` block still gets the catalogue."""
        from osprey.mcp_server.graph.server_context import (
            get_server_context,
            initialize_server_context,
        )

        config = tmp_path / "config.yml"
        config.write_text("facility:\n  name: Test Facility\n")
        monkeypatch.setenv("OSPREY_CONFIG", str(config))
        initialize_server_context()
        assert get_server_context().configured is False

        from osprey.mcp_server.channel_finder_graph.tools import example_queries as mod

        fn = getattr(mod.example_queries, "fn", mod.example_queries)
        payload = json.loads(fn())

        assert payload["count"] == len(EXAMPLE_QUERIES)
        assert payload["examples"][0]["key"] == _EXPECTED_KEYS[0]
