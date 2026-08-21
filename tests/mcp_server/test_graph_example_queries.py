"""Unit tests for the graph MCP server's ``example_queries`` tool.

The tool serves the curated set from :mod:`~osprey.mcp_server.graph.tools.examples_data`
and nothing else, so the payload is asserted directly through the unwrapped
function. The store-down guarantee — the tool answers while the graph is
unreachable — is proved two ways: a fresh subprocess shows the call never imports
the server-context module, and a poison module in ``sys.modules`` shows the call
never touches it even when it is already loaded.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from tests.mcp_server.conftest import get_tool_fn, registered_tool_names

pytestmark = pytest.mark.unit

_SERVER_CONTEXT_MODULE = "osprey.mcp_server.graph.server_context"

#: The curated keys, in the order the tool must serve them. Kept as a literal so
#: a reordered or dropped example fails here rather than silently changing the
#: reading order the descriptions cross-reference.
_EXPECTED_KEYS = ("q1a", "q1b", "q1c", "q2", "q3", "q4b", "q4c", "q5", "q6")

_EXAMPLE_FIELDS = {"key", "title", "description", "cypher", "parameters"}

_CORPORA = {"als", "demo"}


@pytest.fixture()
def payload() -> dict:
    from osprey.mcp_server.graph.tools import example_queries as mod

    return json.loads(get_tool_fn(mod.example_queries)())


class TestExampleQueriesRegistration:
    """The tool must be reachable under its documented name."""

    def test_example_queries_registered_on_the_graph_server(self):
        from osprey.mcp_server.graph.server import mcp
        from osprey.mcp_server.graph.tools import example_queries  # noqa: F401

        names = registered_tool_names(mcp)

        assert "example_queries" in names, f"example_queries not registered; got {sorted(names)}"


class TestExampleQueriesPayload:
    """The catalogue the agent reads before writing Cypher."""

    def test_example_queries_payload_has_required_keys(self, payload):
        assert set(payload) == {"count", "examples", "notes"}
        assert payload["notes"]

    def test_example_queries_payload_serves_the_whole_curated_set(self, payload):
        from osprey.mcp_server.graph.tools.examples_data import EXAMPLE_QUERIES

        assert payload["count"] == 9
        assert payload["count"] == len(EXAMPLE_QUERIES) == len(payload["examples"])

    def test_example_queries_payload_keeps_the_curated_order(self, payload):
        assert tuple(example["key"] for example in payload["examples"]) == _EXPECTED_KEYS

    def test_example_queries_payload_entries_have_the_documented_shape(self, payload):
        for example in payload["examples"]:
            assert set(example) == _EXAMPLE_FIELDS, f"{example.get('key')} has unexpected fields"
            for field in ("key", "title", "description", "cypher"):
                assert isinstance(example[field], str) and example[field].strip(), (
                    f"{example['key']}.{field} is empty"
                )

    def test_example_queries_payload_carries_both_corpora(self, payload):
        for example in payload["examples"]:
            assert set(example["parameters"]) == _CORPORA, (
                f"{example['key']} does not carry one parameter set per corpus"
            )
            for corpus, values in example["parameters"].items():
                assert isinstance(values, dict), f"{example['key']}.{corpus} is not a mapping"

    def test_example_queries_payload_parameters_cover_every_placeholder(self, payload):
        """Every ``$name`` in the Cypher must have a value in both corpora."""
        import re

        for example in payload["examples"]:
            placeholders = set(re.findall(r"\$(\w+)", example["cypher"]))
            for corpus, values in example["parameters"].items():
                assert placeholders == set(values), (
                    f"{example['key']} ({corpus}): placeholders {sorted(placeholders)} "
                    f"vs parameters {sorted(values)}"
                )

    def test_example_queries_payload_every_query_is_bounded(self, payload):
        for example in payload["examples"]:
            assert "LIMIT" in example["cypher"].upper(), f"{example['key']} has no LIMIT"

    def test_example_queries_notes_explain_how_to_run_an_example(self, payload):
        notes = " ".join(payload["notes"])

        assert "params" in notes
        assert "als" in notes and "demo" in notes
        assert "LIMIT" in notes
        assert "services.graphdb.query_max_rows" in notes

    def test_example_queries_payload_round_trips_json(self, payload):
        assert json.loads(json.dumps(payload)) == payload


class TestExampleQueriesWorksStoreDown:
    """Pure data serving — the tool must answer with no graph store in reach."""

    def test_example_queries_call_never_imports_the_server_context(self):
        """A fresh interpreter: calling the tool must not pull in server_context."""
        code = (
            "import sys\n"
            "from osprey.mcp_server.graph.tools import example_queries as mod\n"
            "tool = mod.example_queries\n"
            "fn = getattr(tool, 'fn', tool)\n"
            "payload = fn()\n"
            "assert payload, 'tool returned an empty payload'\n"
            f"leaked = [m for m in sys.modules if m == {_SERVER_CONTEXT_MODULE!r}]\n"
            "assert not leaked, f'server_context leaked into sys.modules: {leaked}'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, (
            f"calling example_queries reached the server context:\n{result.stderr}"
        )

    def test_example_queries_call_never_touches_a_loaded_server_context(self, monkeypatch):
        """With server_context already loaded, the call must not touch it either."""

        class _Poison(types.ModuleType):
            def __getattr__(self, name: str):
                raise AssertionError(
                    f"example_queries touched {_SERVER_CONTEXT_MODULE}.{name}; "
                    "it must serve curated data without the store"
                )

        monkeypatch.setitem(sys.modules, _SERVER_CONTEXT_MODULE, _Poison(_SERVER_CONTEXT_MODULE))

        from osprey.mcp_server.graph.tools import example_queries as mod

        payload = json.loads(get_tool_fn(mod.example_queries)())

        assert payload["count"] == 9
