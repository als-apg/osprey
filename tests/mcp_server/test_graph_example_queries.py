"""Unit tests for the graph MCP server's ``example_queries`` tool.

The tool serves the curated set from :mod:`~osprey.mcp_server.graph.tools.examples_data`
and nothing else, so the payload is asserted directly through the unwrapped
function. The store-down guarantee — the tool answers while the graph is
unreachable — is proved two ways: a fresh subprocess shows the call never imports
the server-context module, and a poison module in ``sys.modules`` shows the call
never touches it even when it is already loaded.

:class:`TestExampleQueriesGolden` then pins the whole thing: the served payload
against a checked-in golden, and the ``graph`` server's tool vocabulary against
the exact four names it advertises. This server belongs to the main agent, so
any other graph-backed server is built beside it and never on top of it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tests.mcp_server.conftest import get_tool_fn, registered_tool_names

pytestmark = pytest.mark.unit

_SERVER_CONTEXT_MODULE = "osprey.mcp_server.graph.server_context"

#: The served catalogue, frozen. Sorted keys, two-space indent, real Unicode,
#: one trailing newline — see :class:`TestExampleQueriesGolden` to regenerate.
_GOLDEN_PATH = Path(__file__).parent / "goldens" / "graph_example_queries.json"

#: The complete tool vocabulary of the main-agent ``graph`` server. Exactly these
#: four, no more: a fifth name here means something registered itself on the
#: shared ``mcp`` singleton instead of standing up its own server.
_ORIGINAL_TOOLS = ("capabilities", "example_queries", "get_schema", "read_cypher")


def _serialise(payload: dict) -> str:
    """Render a payload the way the golden is stored."""
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


#: Probe run in a fresh interpreter: stub the server context so nothing dials a
#: store, build the server, and print every registered tool name.
_REGISTRATION_PROBE = """
import asyncio, json

import osprey.mcp_server.graph.server_context as graph_ctx

graph_ctx.initialize_server_context = lambda: None

from osprey.mcp_server.graph.server import create_server

mcp = create_server()
tools = mcp.get_tools() if hasattr(mcp, "get_tools") else mcp.list_tools()
if asyncio.iscoroutine(tools):
    tools = asyncio.run(tools)
names = list(tools) if isinstance(tools, dict) else [t.name for t in tools]
print("TOOLS " + json.dumps(sorted(names)))
"""


def _registered_tools_in_subprocess() -> list[str]:
    """Tool names ``create_server()`` registers in a pristine interpreter."""
    env = {k: v for k, v in os.environ.items() if k not in {"OSPREY_CONFIG", "CONFIG_FILE"}}
    proc = subprocess.run(
        [sys.executable, "-c", _REGISTRATION_PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stdout}\n{proc.stderr}"
    marker = [line for line in proc.stdout.splitlines() if line.startswith("TOOLS ")]
    assert marker, f"probe printed no tool list:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(marker[-1].removeprefix("TOOLS "))


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


class TestExampleQueriesGolden:
    r"""Byte-for-byte pin on the served catalogue and on the graph tool vocabulary.

    The graph server is the main agent's server. Work that adds *other*
    graph-backed servers must leave it alone, and "alone" is easy to claim and
    hard to see: a reworded description, a dropped example or an extra
    ``@mcp.tool()`` hung off the shared ``mcp`` singleton all pass every
    shape-level assertion above. The golden and the exact-vocabulary check turn
    any of those into a failing test.

    Regenerate deliberately — only when the catalogue is *meant* to change::

        env PYTHONPATH=src python -c "
        import json, pathlib
        from osprey.mcp_server.graph.tools import example_queries as m
        fn = getattr(m.example_queries, 'fn', m.example_queries)
        pathlib.Path('tests/mcp_server/goldens/graph_example_queries.json').write_text(
            json.dumps(json.loads(fn()), indent=2, sort_keys=True, ensure_ascii=False) + '\n',
            encoding='utf-8')
        "
    """

    def test_served_payload_matches_the_golden(self, payload):
        golden_text = _GOLDEN_PATH.read_text(encoding="utf-8")
        served_text = _serialise(payload)

        assert served_text == golden_text, (
            f"the example_queries payload drifted from {_GOLDEN_PATH.name}. If the "
            "catalogue was meant to change, regenerate the golden with the command in "
            "this class's docstring; if it was not, the graph server was modified by "
            "work that should have left it untouched."
        )

    def test_golden_matches_examples_data_length_and_ids(self):
        """The golden pins the source constant, not just the tool's output."""
        from osprey.mcp_server.graph.tools.examples_data import EXAMPLE_QUERIES

        golden = json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))
        golden_keys = tuple(example["key"] for example in golden["examples"])

        assert len(EXAMPLE_QUERIES) == len(golden["examples"]) == golden["count"]
        assert tuple(example.key for example in EXAMPLE_QUERIES) == golden_keys
        assert golden_keys == _EXPECTED_KEYS

    def test_graph_server_registers_exactly_the_original_tools(self):
        """No tool may be added to — or removed from — the ``graph`` server.

        Run in a fresh interpreter: ``mcp`` is a module singleton, so a module
        imported earlier in this worker could otherwise add a tool that makes
        the "exactly" assertion pass or fail for reasons unrelated to the
        server's own wiring.
        """
        names = set(_registered_tools_in_subprocess())

        assert names == set(_ORIGINAL_TOOLS), (
            f"the graph server's tool vocabulary changed: unexpected "
            f"{sorted(names - set(_ORIGINAL_TOOLS))}, missing "
            f"{sorted(set(_ORIGINAL_TOOLS) - names)}. The main-agent graph server is "
            "not this phase's to extend — register new tools on their own server."
        )
