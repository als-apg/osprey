"""Tests for the Neo4j/neosemantics graph-store seeder primitives.

Every test drives a fake driver session — no container, no neo4j driver.  What
is under test is the *policy* baked into these functions (when init runs, when
drift is reported, what the marker sequencing guarantees) plus the shape of the
Cypher each one emits, since a silently reworded statement is the one bug the
testcontainer integration test would catch only much later.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from osprey.services.facility_knowledge.seeder import graph_seeder
from osprey.services.facility_knowledge.seeder.graph_seeder import (
    COMMIT_SIZE,
    CONFIG_DRIFT_MESSAGE,
    N10S_GRAPH_CONFIG,
    NARAD_PREFIXES,
    BootstrapStatus,
    bootstrap,
    import_ttl,
    read_marker,
    resource_count,
    ttl_sha256,
    wipe,
    write_marker,
)

# ---------------------------------------------------------------------------
# Fake driver session
# ---------------------------------------------------------------------------


class FakeResult:
    """Stands in for a neo4j ``Result``: iterable, ``single()``, ``consume()``."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self.consumed = False

    def __iter__(self):
        return iter(self._records)

    def single(self) -> dict[str, Any] | None:
        return self._records[0] if self._records else None

    def consume(self) -> None:
        self.consumed = True


class FakeSession:
    """Records every statement, replying from a substring→records table.

    Args:
        responses: Maps a substring of the Cypher to the records to return for
            it.  Unmatched statements return no rows, which is what a write
            statement does anyway.
    """

    def __init__(self, responses: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.results: list[FakeResult] = []

    def run(self, query: str, **params: Any) -> FakeResult:
        self.calls.append((query, params))
        records: list[dict[str, Any]] = []
        for needle, rows in self.responses.items():
            if needle in query:
                records = rows
                break
        result = FakeResult(records)
        self.results.append(result)
        return result

    # -- assertions helpers -------------------------------------------------

    @property
    def queries(self) -> list[str]:
        return [query for query, _ in self.calls]

    def matching(self, needle: str) -> list[tuple[str, dict[str, Any]]]:
        return [(q, p) for q, p in self.calls if needle in q]


def _config_rows(overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Build ``graphconfig.show`` rows from the canonical config, with overrides."""
    values = dict(N10S_GRAPH_CONFIG)
    values.update(overrides or {})
    return [{"param": key, "value": value} for key, value in values.items()]


_SHOW = "n10s.graphconfig.show"
_INIT = "n10s.graphconfig.init"
_PREFIX = "n10s.nsprefixes.add"


# ---------------------------------------------------------------------------
# Canonical settings
# ---------------------------------------------------------------------------


class TestCanonicalSettings:
    """The settings table must match the als-ontology prototype's init cypher."""

    def test_graph_config_matches_prototype(self):
        assert N10S_GRAPH_CONFIG == {
            "handleVocabUris": "MAP",
            "handleMultival": "OVERWRITE",
            "keepLangTag": False,
            "handleRDFTypes": "LABELS_AND_NODES",
            "applyNeo4jNaming": True,
        }

    def test_narad_prefixes_match_prototype(self):
        assert NARAD_PREFIXES == {
            "narad": "https://narad.example.org/",
            "narad_sem": "https://narad.example.org/schema/shared_semantics/",
            "narad_p": "https://narad.example.org/property/",
            "narad_sig": "https://narad.example.org/signal/",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
        }

    def test_commit_size_is_a_positive_int(self):
        assert isinstance(COMMIT_SIZE, int)
        assert COMMIT_SIZE > 0


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


class TestBootstrapConstraint:
    """The uniqueness constraint runs on every path, and is re-runnable."""

    @pytest.mark.parametrize(
        "responses",
        [
            {},  # no config
            {_SHOW: _config_rows()},  # canonical config
            {_SHOW: _config_rows({"handleMultival": "ARRAY"})},  # drifted config
        ],
        ids=["no-config", "canonical-config", "drifted-config"],
    )
    def test_constraint_always_runs(self, responses):
        session = FakeSession(responses)
        bootstrap(session)
        constraint = session.matching("CREATE CONSTRAINT n10s_unique_uri")
        assert len(constraint) == 1
        assert "IF NOT EXISTS" in constraint[0][0]
        assert "FOR (r:Resource) REQUIRE r.uri IS UNIQUE" in constraint[0][0]

    def test_constraint_runs_before_the_config_gate(self):
        session = FakeSession()
        bootstrap(session)
        order = session.queries
        assert order.index(next(q for q in order if "CREATE CONSTRAINT" in q)) < order.index(
            next(q for q in order if _SHOW in q)
        )


class TestBootstrapConfigGate:
    """``graphconfig.init`` is gated on ``graphconfig.show`` reporting no config."""

    def test_no_config_initializes_with_canonical_settings(self):
        session = FakeSession()  # show yields zero rows == unconfigured
        result = bootstrap(session)

        assert result.status is BootstrapStatus.INITIALIZED
        assert result.differing_keys == ()
        assert result.ok is True

        init_calls = session.matching(_INIT)
        assert len(init_calls) == 1
        assert init_calls[0][1]["params"] == N10S_GRAPH_CONFIG

    def test_canonical_config_skips_init_and_reports_no_drift(self):
        session = FakeSession({_SHOW: _config_rows()})
        result = bootstrap(session)

        assert result.status is BootstrapStatus.ALREADY_CANONICAL
        assert result.differing_keys == ()
        assert result.ok is True
        assert session.matching(_INIT) == []

    def test_stringified_booleans_are_not_drift(self):
        """n10s builds differ on whether booleans come back as bools or strings."""
        session = FakeSession(
            {_SHOW: _config_rows({"keepLangTag": "false", "applyNeo4jNaming": "true"})}
        )
        result = bootstrap(session)
        assert result.status is BootstrapStatus.ALREADY_CANONICAL

    def test_drifted_config_names_the_differing_keys(self):
        session = FakeSession(
            {_SHOW: _config_rows({"handleMultival": "ARRAY", "keepLangTag": True})}
        )
        result = bootstrap(session)

        assert result.status is BootstrapStatus.DIFFERS
        assert result.differing_keys == ("handleMultival", "keepLangTag")
        assert result.ok is False
        assert session.matching(_INIT) == [], "must not re-init a configured store"

    def test_missing_key_counts_as_drift(self):
        rows = [row for row in _config_rows() if row["param"] != "handleRDFTypes"]
        result = bootstrap(FakeSession({_SHOW: rows}))
        assert result.status is BootstrapStatus.DIFFERS
        assert result.differing_keys == ("handleRDFTypes",)

    def test_drift_message_tells_the_operator_to_force(self):
        session = FakeSession({_SHOW: _config_rows({"handleVocabUris": "IGNORE"})})
        message = bootstrap(session).message
        assert CONFIG_DRIFT_MESSAGE in message
        assert "--force" in message
        assert "handleVocabUris" in message

    def test_extra_live_keys_are_not_drift(self):
        """n10s reports keys osprey does not pin; only load-bearing keys count."""
        rows = _config_rows() + [{"param": "classLabel", "value": "Class"}]
        result = bootstrap(FakeSession({_SHOW: rows}))
        assert result.status is BootstrapStatus.ALREADY_CANONICAL


class TestBootstrapPrefixes:
    """Every NARAD prefix is registered on every path."""

    @pytest.mark.parametrize(
        "responses",
        [
            {},
            {_SHOW: _config_rows()},
            {_SHOW: _config_rows({"handleMultival": "ARRAY"})},
        ],
        ids=["no-config", "canonical-config", "drifted-config"],
    )
    def test_prefixes_always_registered(self, responses):
        session = FakeSession(responses)
        bootstrap(session)

        added = {params["prefix"]: params["namespace"] for _, params in session.matching(_PREFIX)}
        assert added == NARAD_PREFIXES

    def test_results_are_consumed_so_server_errors_surface(self):
        """Lazily streamed results hide failures until consumed."""
        session = FakeSession()
        bootstrap(session)
        writes = [
            result
            for (query, _), result in zip(session.calls, session.results, strict=True)
            if _SHOW not in query
        ]
        assert writes and all(result.consumed for result in writes)


# ---------------------------------------------------------------------------
# resource_count
# ---------------------------------------------------------------------------


class TestResourceCount:
    """An empty store means zero ``(:Resource)`` nodes, not zero nodes."""

    def test_counts_resource_nodes(self):
        session = FakeSession({"MATCH (r:Resource)": [{"resource_count": 8114}]})
        assert resource_count(session) == 8114

        query = session.queries[0]
        assert "MATCH (r:Resource)" in query
        assert "count(r)" in query
        assert "_OspreySeed" not in query and "_GraphConfig" not in query

    def test_zero_on_empty_store(self):
        session = FakeSession({"MATCH (r:Resource)": [{"resource_count": 0}]})
        assert resource_count(session) == 0

    def test_no_row_reads_as_zero(self):
        assert resource_count(FakeSession()) == 0


# ---------------------------------------------------------------------------
# marker
# ---------------------------------------------------------------------------


class TestSeedMarker:
    """The ``(:_OspreySeed)`` sha256 marker round-trips and stays a singleton."""

    def test_read_marker_returns_stored_sha(self):
        session = FakeSession({"_OspreySeed": [{"sha256": "abc123"}]})
        assert read_marker(session) == "abc123"
        assert "MATCH (m:_OspreySeed)" in session.queries[0]

    def test_read_marker_returns_none_when_absent(self):
        assert read_marker(FakeSession()) is None

    def test_read_marker_returns_none_when_property_missing(self):
        session = FakeSession({"_OspreySeed": [{"sha256": None}]})
        assert read_marker(session) is None

    def test_write_marker_merges_a_single_node(self):
        session = FakeSession()
        write_marker(session, "deadbeef")

        query, params = session.calls[0]
        assert query.startswith("MERGE (m:_OspreySeed")
        assert "SET m.sha256 = $sha256" in query
        assert params == {"sha256": "deadbeef"}
        assert session.results[0].consumed

    def test_write_marker_overwrites_rather_than_accumulating(self):
        """A second seed must not leave two markers behind."""
        session = FakeSession()
        write_marker(session, "one")
        write_marker(session, "two")
        assert all(query.startswith("MERGE") for query in session.queries)
        assert all("CREATE (" not in query for query in session.queries)

    def test_write_marker_is_documented_as_post_import_only(self):
        """The ordering contract lives in the docstring the caller reads."""
        doc = write_marker.__doc__ or ""
        assert "terminationStatus" in doc or "ImportResult.ok" in doc
        assert "--force" in doc or "unmanaged-partial" in doc


# ---------------------------------------------------------------------------
# ttl_sha256
# ---------------------------------------------------------------------------


class TestTtlDigest:
    """The marker's identity is computed in one place, for both writers.

    ``osprey knowledge seed-graph`` and the deploy-time ``_stage_graphdb_store``
    each write markers and each read the other's.  Spelled separately, the two
    would agree only by coincidence: one side normalizing differently would see
    the other's marker as a different corpus and re-seed a store that is already
    correct.
    """

    def test_the_digest_is_sha256_over_the_utf8_text(self):
        """Pinned against an independently spelled sha256 rather than against
        itself, so a reworked implementation has to still produce the digest
        every already-written marker carries."""
        import hashlib

        text = "@prefix ex: <http://example.org/> .\nex:a a ex:Thing .\n"

        assert ttl_sha256(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_the_same_corpus_always_hashes_the_same(self):
        """What makes `unchanged` reportable at all."""
        text = "ex:a a ex:Thing .\n"
        assert ttl_sha256(text) == ttl_sha256(text)

    def test_a_one_byte_change_changes_the_digest(self):
        """And what makes `differs` detectable: an edited corpus must not pass
        as the seeded one."""
        assert ttl_sha256("ex:a a ex:Thing .\n") != ttl_sha256("ex:a a ex:Thingg .\n")

    def test_non_ascii_is_encoded_as_utf8_not_as_the_platform_default(self):
        """A facility export carrying an accented label or a µ must digest
        identically on every host, or a marker written on one machine reads as a
        different corpus on another."""
        import hashlib

        text = "ex:a ex:label 'Sekundärstrahlung µm' .\n"

        assert ttl_sha256(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_both_marker_writers_use_this_one_helper(self):
        """The duplication this helper closed. Grep-level, deliberately: the two
        call sites live in modules whose imports are lazy, so an assertion about
        their behaviour would have to boot a deploy or a CLI to see it."""
        from pathlib import Path

        import osprey.cli.knowledge_cmd as knowledge_cmd
        import osprey.deployment.container_lifecycle as container_lifecycle

        for module in (knowledge_cmd, container_lifecycle):
            assert "ttl_sha256" in Path(module.__file__).read_text(encoding="utf-8")

        # Negative check only where the duplicate actually lived. Asserting the
        # absence of an inline digest across container_lifecycle would fail on
        # any unrelated sha256 that module grows for its own reasons.
        assert "hashlib" not in Path(knowledge_cmd.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# import_ttl
# ---------------------------------------------------------------------------


class TestImportTtl:
    """Import goes through ``n10s.rdf.import.inline`` with an explicit commitSize."""

    def _ok_session(self) -> FakeSession:
        return FakeSession(
            {
                "n10s.rdf.import.inline": [
                    {
                        "terminationStatus": "OK",
                        "triplesLoaded": 8114,
                        "triplesParsed": 8114,
                        "extraInfo": "",
                    }
                ]
            }
        )

    def test_passes_payload_and_commit_size(self):
        session = self._ok_session()
        import_ttl(session, "@prefix ex: <http://example.org/> .")

        query, params = session.calls[0]
        assert "n10s.rdf.import.inline($payload, 'Turtle'" in query
        assert "commitSize: $commitSize" in query
        assert params["payload"] == "@prefix ex: <http://example.org/> ."
        assert params["commitSize"] == COMMIT_SIZE

    def test_yields_termination_status(self):
        session = self._ok_session()
        result = import_ttl(session, "ttl")

        assert "YIELD terminationStatus" in session.queries[0]
        assert result.termination_status == "OK"
        assert result.triples_loaded == 8114
        assert result.triples_parsed == 8114
        assert result.ok is True

    def test_failure_surfaces_status_and_detail_without_raising(self):
        session = FakeSession(
            {
                "n10s.rdf.import.inline": [
                    {
                        "terminationStatus": "KO",
                        "triplesLoaded": 4000,
                        "triplesParsed": 4000,
                        "extraInfo": "Bad syntax on line 42",
                    }
                ]
            }
        )
        result = import_ttl(session, "ttl")

        assert result.ok is False, "partial loads must not read as success"
        assert result.termination_status == "KO"
        assert result.triples_loaded == 4000
        assert "line 42" in result.extra_info

    def test_missing_result_row_is_not_ok(self):
        result = import_ttl(FakeSession(), "ttl")
        assert result.ok is False
        assert result.triples_loaded == 0


# ---------------------------------------------------------------------------
# wipe
# ---------------------------------------------------------------------------


class TestWipe:
    def test_detach_deletes_everything(self):
        session = FakeSession()
        wipe(session)
        assert session.queries == ["MATCH (n) DETACH DELETE n"]
        assert session.results[0].consumed


# ---------------------------------------------------------------------------
# lazy import
# ---------------------------------------------------------------------------


class TestLazyNeo4jImport:
    """The driver is imported inside the function, and its absence is explained."""

    def test_module_import_does_not_load_neo4j(self):
        source = (
            "from osprey.services.facility_knowledge.seeder import graph_seeder\n"
            "import sys\n"
            "assert 'neo4j' not in sys.modules, 'neo4j imported at module scope'\n"
        )
        result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_missing_driver_error_names_the_dependency(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "neo4j", None)
        with pytest.raises(ImportError) as excinfo:
            with graph_seeder.open_session("bolt://localhost:7687", "neo4j", "pw"):
                pass  # pragma: no cover
        message = str(excinfo.value)
        assert "neo4j" in message
        assert "pip install" in message
