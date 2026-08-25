"""Unit tests for the graph channel-finder's ``capabilities`` tool.

The tool is a static manifest, so most of these assert its payload directly
through the unwrapped function. The one thing it does read is the server
context — the row cap and the query deadline it reports are the context's, so
they follow a deployment's config instead of restating a constant that config
can override.

That single read is also the thing worth guarding. The fake below answers the
two attributes the tool is allowed to touch and raises on **everything else**,
so a future edit that reaches for ``run_read``, ``is_empty`` or the driver fails
here rather than turning a manifest call into a store dial. A manifest that
dialled the store would stop being answerable on a deployment whose graph is
unconfigured or down — which is exactly when an agent needs to be told what the
paradigm expects of it.
"""

from __future__ import annotations

import json

import pytest

from osprey.mcp_server.channel_finder_graph.tools import capabilities as mod
from osprey.mcp_server.graph.server_context import (
    DEFAULT_QUERY_MAX_ROWS,
    DEFAULT_QUERY_TIMEOUT_S,
    QUERY_MAX_ROWS_CONFIG_KEY,
    QUERY_TIMEOUT_CONFIG_KEY,
)
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn, registered_tool_names

pytestmark = pytest.mark.unit

capabilities_fn = get_tool_fn(mod.capabilities)

#: The full tool vocabulary the paradigm advertises. A literal, so a renamed or
#: dropped tool fails here rather than silently shrinking the manifest.
_EXPECTED_TOOLS = {"example_queries", "get_schema", "read_cypher", "capabilities"}

#: Independent marker list (do not import the production constant) mirroring the
#: registry-wide destructive-name scan in tests/agent_runner/test_write_tools.py.
_DESTRUCTIVE_MARKERS = ("delete", "remove", "clear", "wipe", "purge", "destroy")


class _FakeContext:
    """Stands in for :class:`GraphContext`, exposing only the manifest's reads.

    Args:
        query_max_rows: Row cap the tool should report.
        query_timeout_s: Query deadline the tool should report.

    Attributes:
        read: Names the tool actually read, in order.
    """

    def __init__(
        self,
        *,
        query_max_rows: int = DEFAULT_QUERY_MAX_ROWS,
        query_timeout_s: int = DEFAULT_QUERY_TIMEOUT_S,
    ) -> None:
        self.read: list[str] = []
        self._values = {
            "query_max_rows": query_max_rows,
            "query_timeout_s": query_timeout_s,
        }

    def __getattr__(self, name: str):
        values = self.__dict__["_values"]
        if name not in values:
            raise AssertionError(
                f"capabilities read GraphContext.{name}; the manifest may read only "
                f"{sorted(values)} — anything else is a store dial in a tool that "
                "must answer while the graph is unconfigured or down"
            )
        self.__dict__["read"].append(name)
        return values[name]


def _payload(monkeypatch, context: _FakeContext | None = None) -> dict:
    """Call the tool against *context* and parse its JSON."""
    context = _FakeContext() if context is None else context
    monkeypatch.setattr(mod, "get_server_context", lambda: context)
    return json.loads(capabilities_fn())


@pytest.fixture()
def payload(monkeypatch) -> dict:
    """The manifest as an unconfigured deployment answers it."""
    return _payload(monkeypatch)


class TestRegistration:
    """The tool lands on the channel-finder server, not the main-agent one."""

    def test_capabilities_is_registered_on_the_channel_finder_server(self):
        from osprey.mcp_server.channel_finder_graph import server as cf_graph_server

        assert mod.mcp is cf_graph_server.mcp
        assert "capabilities" in registered_tool_names(mod.mcp)


class TestManifestShape:
    """The keys a caller may rely on, and the identity of the paradigm."""

    def test_payload_has_exactly_the_documented_keys(self, payload):
        assert set(payload) == {
            "paradigm",
            "server",
            "description",
            "tools",
            "workflow",
            "addresses",
            "description_predicates",
            "synonyms",
            "corpus",
            "limits",
            "notes",
        }

    def test_payload_names_the_paradigm_and_the_wire_name(self, payload):
        assert payload["paradigm"] == "graph"
        assert payload["server"] == "channel-finder"
        assert payload["description"].strip()

    def test_payload_lists_every_tool_with_a_purpose(self, payload):
        assert {tool["name"] for tool in payload["tools"]} == _EXPECTED_TOOLS
        for tool in payload["tools"]:
            assert tool["purpose"].strip(), f"{tool['name']} has no purpose line"

    def test_tool_names_are_not_destructive(self, payload):
        for tool in payload["tools"]:
            offenders = [m for m in _DESTRUCTIVE_MARKERS if m in tool["name"].lower()]
            assert offenders == [], (
                f"{tool['name']} reads as destructive ({offenders}); a read-only "
                "server's tools must never be auto-allowed under such a name"
            )

    def test_workflow_runs_examples_then_schema_then_the_query(self, payload):
        """The order is the instruction: learn the shapes before writing Cypher."""
        steps = payload["workflow"]
        assert len(steps) >= 3

        positions = {
            tool: next(i for i, step in enumerate(steps) if tool in step)
            for tool in ("example_queries", "get_schema", "read_cypher")
        }

        assert positions["example_queries"] < positions["get_schema"] < positions["read_cypher"], (
            f"workflow does not route examples -> schema -> query: {steps}"
        )


class TestGraphVocabulary:
    """What the agent has to know before it can write a query that matches."""

    def test_addresses_come_from_full_pv_on_a_binding(self, payload):
        addresses = payload["addresses"]

        assert addresses["property"] == "fullPv"
        assert addresses["node_label"] == "ChannelBinding"

    def test_addresses_report_the_generated_six_token_grammar(self, payload):
        grammar = payload["addresses"]["generated_grammar"]

        assert grammar["tokens"] == [
            "ring",
            "system",
            "family",
            "device",
            "field",
            "subfield",
        ]
        assert grammar["example"].count(":") == 5

    def test_addresses_warn_that_the_grammar_is_the_corpus_own(self, payload):
        """A facility corpus records its own address shape; the grammar is the demo's."""
        notes = " ".join(payload["addresses"]["notes"])

        assert "real facility" in notes
        assert "fullPv" in notes

    def test_description_predicates_are_reported_per_node_type(self, payload):
        predicates = payload["description_predicates"]

        assert {kind: entry["properties"] for kind, entry in predicates.items()} == {
            "binding": ["description", "fieldDescription", "subfieldDescription"],
            "device": ["familyDescription", "systemDescription", "ringDescription"],
            "signal": [],
        }

    def test_advertised_prose_properties_are_the_generators(self, payload):
        """Pin the manifest to the emitter so the two cannot drift apart.

        The manifest spells the six description predicates as literals rather
        than importing them, which keeps the server off the TTL generator's
        import path. This is the seam that would otherwise let a renamed or
        newly added predicate ship in generated corpora while the manifest kept
        advertising the old vocabulary.
        """
        from osprey.services.facility_knowledge.ttl_generator.emitter import PROPERTY_NAMES

        advertised = {
            prop
            for entry in payload["description_predicates"].values()
            for prop in entry["properties"]
        }
        generated = {
            name for name in PROPERTY_NAMES if name == "description" or name.endswith("Description")
        }

        assert advertised == generated, (
            "the manifest's prose vocabulary drifted from the TTL emitter's "
            f"(manifest only: {sorted(advertised - generated)}; "
            f"emitter only: {sorted(generated - advertised)})"
        )

    def test_every_node_type_says_how_it_is_matched(self, payload):
        for kind, entry in payload["description_predicates"].items():
            assert entry["match"].strip(), f"{kind} has no match pattern"

    def test_every_node_type_says_which_corpora_carry_its_prose(self, payload):
        """Advertising the predicates without saying who has them reads as a promise."""
        for kind, entry in payload["description_predicates"].items():
            assert entry["presence"].strip(), f"{kind} does not say when its prose exists"

        assert "facility export" in payload["description_predicates"]["binding"]["presence"]
        assert "Never" in payload["description_predicates"]["signal"]["presence"]

    def test_description_predicates_say_prose_is_corpus_dependent(self, payload):
        """A phrase search on a corpus with no prose returns nothing, not an error."""
        joined = " ".join(payload["notes"] + payload["corpus"]["notes"]).lower()

        assert "prose" in joined

    def test_synonyms_are_a_list_matched_with_any(self, payload):
        synonyms = payload["synonyms"]

        assert synonyms["property"] == "altLabel"
        assert synonyms["node_label"] == "Class"
        assert synonyms["type"] == "list"
        assert synonyms["match"].startswith("ANY(")

    def test_synonyms_warn_against_scalar_matching(self, payload):
        """``CONTAINS`` against a list property matches nothing — say so."""
        notes = " ".join(payload["synonyms"]["notes"])

        assert "CONTAINS" in notes
        assert "ARRAY" in notes

    def test_corpus_identity_is_the_facility_property(self, payload):
        corpus = payload["corpus"]

        assert corpus["property"] == "facility"
        assert "facility" in corpus["read_with"]
        assert corpus["notes"]


class TestLimits:
    """The row cap and the deadline the agent's query will actually meet."""

    def test_limits_name_the_config_keys_that_set_them(self, payload):
        limits = payload["limits"]

        assert limits["max_rows_config_key"] == QUERY_MAX_ROWS_CONFIG_KEY
        assert limits["timeout_config_key"] == QUERY_TIMEOUT_CONFIG_KEY

    def test_limits_default_to_the_graph_context_defaults(self, payload):
        limits = payload["limits"]

        assert limits["max_rows"] == DEFAULT_QUERY_MAX_ROWS
        assert limits["timeout_s"] == DEFAULT_QUERY_TIMEOUT_S

    def test_limits_follow_the_deployment_rather_than_the_constant(self, monkeypatch):
        """A deployment that raised the cap must not be told the default."""
        context = _FakeContext(query_max_rows=25, query_timeout_s=90)

        limits = _payload(monkeypatch, context)["limits"]

        assert (limits["max_rows"], limits["timeout_s"]) == (25, 90)
        assert set(context.read) == {"query_max_rows", "query_timeout_s"}

    def test_notes_tell_the_agent_what_a_truncated_result_means(self, payload):
        notes = " ".join(payload["notes"])

        assert "narrow" in notes.lower()

    def test_notes_name_the_seed_command_for_an_empty_graph(self, payload):
        from osprey.deployment.graphdb_service import GRAPHDB_SEED_COMMAND

        assert GRAPHDB_SEED_COMMAND in " ".join(payload["notes"])

    def test_notes_state_that_this_is_not_a_health_check(self, payload):
        notes = " ".join(payload["notes"]).lower()

        assert "reachable" in notes or "health" in notes


class TestAnswersWithoutAStore:
    """The manifest is what an agent reads when nothing else works."""

    def test_the_manifest_never_reads_beyond_the_two_limit_attributes(self, monkeypatch):
        """The fake raises on any other attribute — including ``_driver``."""
        context = _FakeContext()

        _payload(monkeypatch, context)

        assert set(context.read) <= {"query_max_rows", "query_timeout_s"}

    def test_an_unconfigured_deployment_still_gets_the_whole_manifest(self, monkeypatch):
        """No ``services.graphdb`` block means default limits, not a failed tool."""
        payload = _payload(monkeypatch, _FakeContext())

        assert payload["tools"]
        assert payload["workflow"]
        assert payload["limits"]["max_rows"] == DEFAULT_QUERY_MAX_ROWS
        assert "error" not in payload

    def test_a_context_that_cannot_be_read_still_gets_the_standard_envelope(self, monkeypatch):
        """A broken context is the shared error envelope, not an unhandled exception."""

        def _boom():
            raise RuntimeError("Graph server context not initialized.")

        monkeypatch.setattr(mod, "get_server_context", _boom)

        with assert_raises_error(error_type="internal_error"):
            capabilities_fn()
