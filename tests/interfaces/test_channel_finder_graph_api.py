"""The channel finder's graph paradigm answers membership and enumeration.

Two questions the graph paradigm used to refuse — "is this channel real?" (501)
and "which channels are there?" (404) — are answered here from the channel
roster, the one enumeration of a facility's channels. The store is never dialed
for either: the roster is the staged corpus the store is seeded *from*, read
once when the app starts.

What these tests pin:

- Membership and enumeration answer from the roster, in the shapes the
  file-backed paradigms already answer them in.
- ``chunk_idx`` is refused (422) rather than honoured: chunking exists to cut
  the in-context paradigm's prompt into pieces, and the graph builds no prompt.
- A deployment that stages no corpus — one pointed at an external store — still
  *starts*: both routes answer 503 naming ``services.graphdb.ttl_path``, which
  is the key an operator edits. The same goes for a search index nobody built,
  and for one that is there and cannot be opened.
- The roster is read once at lifespan, not once per request.
- The file-backed paradigms are untouched.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import osprey.channel_roster as channel_roster
from tests._graph_index import build_index_from_ttl, default_index_path

_CONFIG_SEAM = "osprey.utils.workspace.load_osprey_config"
_GRAPH_CONTEXT_SEAM = "osprey.interfaces.channel_finder.app._make_graph_context"

#: The key an operator edits to name the corpus, which every unavailable answer
#: has to put in front of them.
_TTL_KEY = "services.graphdb.ttl_path"

_PREAMBLE = """\
@prefix narad_p: <https://narad.example.org/property/> .
@prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .
"""

#: A corpus small enough to assert against whole: two settable channels and one
#: readable one, so membership, direction-blind enumeration and ordering are all
#: observable.
_CORPUS = {
    "SR:MAG:QF:01:CURRENT:SP": "writesSignal",
    "SR:MAG:QF:01:CURRENT:RB": "readsSignal",
    "SR:DIAG:BPM:01:X": "readsSignal",
}


@pytest.fixture(autouse=True)
def cold_roster_cache() -> Iterator[None]:
    """Start and leave every test with an empty roster cache."""
    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


def _write_corpus(path: Path, bindings: dict[str, str]) -> Path:
    """Write a corpus binding each address to its direction predicate."""
    body = "".join(
        f'<https://narad.example.org/binding/b{index}> narad_p:fullPv "{address}" ;\n'
        f"    narad_p:{predicate} narad_sem:s{index} .\n"
        for index, (address, predicate) in enumerate(bindings.items())
    )
    path.write_text(_PREAMBLE + body, encoding="utf-8")
    return path


def _graph_config(render: Path, ttl_path: str | None = "corpus.ttl") -> dict[str, Any]:
    """A graph-paradigm project rendered into *render*."""
    graphdb: dict[str, Any] = {"uri": "bolt://localhost:7687"}
    if ttl_path is not None:
        graphdb["ttl_path"] = ttl_path
    return {
        "config_dir": str(render),
        "channel_finder": {"pipeline_mode": "graph", "pipelines": None},
        "services": {"graphdb": graphdb},
    }


@contextmanager
def _started(config: dict[str, Any]) -> Iterator[TestClient]:
    """Run the real app lifespan against *config* and yield a client for it.

    The store context is faked: it is resolved at startup and dialed by the
    explorer's routes, and nothing here asks the store anything.
    """
    from osprey.interfaces.channel_finder.app import create_app

    with (
        patch(_CONFIG_SEAM, return_value=config),
        patch(_GRAPH_CONTEXT_SEAM, return_value=MagicMock()),
    ):
        with TestClient(create_app(project_cwd="/tmp/test-project")) as client:
            yield client


@pytest.fixture
def graph_client(tmp_path: Path) -> Iterator[TestClient]:
    """A started graph-mode app whose corpus holds :data:`_CORPUS`."""
    config = _graph_config(tmp_path)
    build_index_from_ttl(_write_corpus(tmp_path / "corpus.ttl", _CORPUS), config)
    with _started(config) as client:
        yield client


def _unavailable_text(body: dict[str, Any]) -> str:
    """Everything a 503 body puts in front of an operator, as one string."""
    return " ".join([body["detail"], *body["suggestions"]])


class TestGraphMembership:
    """POST /api/validate answers from the roster's addresses."""

    def test_a_channel_the_corpus_declares_is_valid(self, graph_client):
        resp = graph_client.post("/api/validate", json={"channels": ["SR:MAG:QF:01:CURRENT:SP"]})

        assert resp.status_code == 200
        assert resp.json() == {
            "results": [{"channel": "SR:MAG:QF:01:CURRENT:SP", "valid": True}],
            "valid_count": 1,
            "invalid_count": 0,
            "total": 1,
        }

    def test_a_channel_the_corpus_does_not_declare_is_invalid(self, graph_client):
        resp = graph_client.post("/api/validate", json={"channels": ["SR:MAG:NOPE:01"]})

        assert resp.status_code == 200
        assert resp.json()["results"] == [{"channel": "SR:MAG:NOPE:01", "valid": False}]
        assert resp.json()["invalid_count"] == 1

    def test_membership_is_answered_for_every_channel_asked_about(self, graph_client):
        asked = ["SR:DIAG:BPM:01:X", "SR:MAG:NOPE:01", "SR:MAG:QF:01:CURRENT:RB"]

        body = graph_client.post("/api/validate", json={"channels": asked}).json()

        assert [entry["channel"] for entry in body["results"]] == asked
        assert body["valid_count"] == 2
        assert body["invalid_count"] == 1
        assert body["total"] == 3

    def test_direction_does_not_narrow_membership(self, graph_client):
        """A settable channel and a readable one are equally real."""
        body = graph_client.post(
            "/api/validate",
            json={"channels": ["SR:MAG:QF:01:CURRENT:SP", "SR:DIAG:BPM:01:X"]},
        ).json()

        assert body["valid_count"] == 2


class TestGraphEnumeration:
    """GET /api/channels serves the roster's addresses."""

    def test_serves_every_address_the_corpus_declares(self, graph_client):
        resp = graph_client.get("/api/channels")

        assert resp.status_code == 200
        assert resp.json() == {
            # The item shape every paradigm answers this route in: the channel
            # under "channel", with the file-backed paradigms' extra columns
            # beside it where they have any.
            "channels": [{"channel": address} for address in sorted(_CORPUS)],
            "total": len(_CORPUS),
        }

    def test_an_address_bound_twice_is_enumerated_once(self, tmp_path):
        """Otherwise the total disagrees with what membership can find."""
        path = tmp_path / "corpus.ttl"
        path.write_text(
            _PREAMBLE
            + '<https://narad.example.org/binding/a> narad_p:fullPv "SR:DIAG:BPM:01:X" ;\n'
            "    narad_p:readsSignal narad_sem:s0 .\n"
            '<https://narad.example.org/binding/b> narad_p:fullPv "SR:DIAG:BPM:01:X" ;\n'
            "    narad_p:readsSignal narad_sem:s1 .\n",
            encoding="utf-8",
        )

        config = _graph_config(tmp_path)
        build_index_from_ttl(path, config)

        with _started(config) as client:
            body = client.get("/api/channels").json()

            assert body == {"channels": [{"channel": "SR:DIAG:BPM:01:X"}], "total": 1}

    def test_chunk_idx_is_refused_as_an_in_context_contract(self, graph_client):
        resp = graph_client.get("/api/channels?chunk_idx=0")

        assert resp.status_code == 422
        assert "chunk_idx" in resp.json()["detail"]

    def test_chunk_idx_is_refused_even_when_it_would_be_in_range(self, graph_client):
        """Not a range check: the graph builds no prompt to chunk at all."""
        assert graph_client.get("/api/channels?chunk_idx=0&chunk_size=1").status_code == 422

    def test_the_index_is_read_once_for_the_whole_process(self, tmp_path):
        config = _graph_config(tmp_path)
        build_index_from_ttl(_write_corpus(tmp_path / "corpus.ttl", _CORPUS), config)
        reads: list[dict[str, Any]] = []
        real = channel_roster.registered_channels

        def counted(config):
            reads.append(config)
            return real(config)

        with patch.object(channel_roster, "registered_channels", counted):
            with _started(config) as client:
                client.get("/api/channels")
                client.get("/api/channels")
                client.post("/api/validate", json={"channels": ["SR:DIAG:BPM:01:X"]})

        assert len(reads) == 1


class TestADeploymentThatStagesNoCorpus:
    """The external-store deployment: a graph store nobody staged a corpus for."""

    @pytest.fixture
    def client(self, tmp_path: Path) -> Iterator[TestClient]:
        with _started(_graph_config(tmp_path, ttl_path=None)) as started:
            yield started

    def test_the_app_still_starts_and_serves_the_graph_paradigm(self, client):
        assert client.get("/health").json()["pipeline_type"] == "graph"
        assert client.get("/api/info").json()["graph_backed"] is True

    def test_validate_503s_naming_the_corpus_key(self, client):
        resp = client.post("/api/validate", json={"channels": ["SR:DIAG:BPM:01:X"]})

        assert resp.status_code == 503
        assert _TTL_KEY in _unavailable_text(resp.json())

    def test_channels_503s_naming_the_corpus_key(self, client):
        resp = client.get("/api/channels")

        assert resp.status_code == 503
        assert _TTL_KEY in _unavailable_text(resp.json())

    def test_the_body_carries_the_remedy_the_other_graph_routes_carry(self, client):
        body = client.get("/api/channels").json()

        assert body["error_type"] == "service_unavailable"
        assert body["suggestions"]

    def test_the_reason_is_the_roster_absence_verbatim(self, client, tmp_path):
        from osprey.channel_roster import resolve_roster_source

        absence = resolve_roster_source(_graph_config(tmp_path, ttl_path=None)).absence

        assert client.get("/api/channels").json()["detail"] == absence.message()


class TestAnIndexThatCannotBeRead:
    """A staged index that is missing or unreadable is not an absent one."""

    @pytest.fixture
    def client(self, tmp_path: Path) -> Iterator[TestClient]:
        _write_corpus(tmp_path / "corpus.ttl", _CORPUS)
        index_path = default_index_path(tmp_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_bytes(b"this is not a database {{{")
        with _started(_graph_config(tmp_path)) as started:
            yield started

    def test_the_app_still_starts(self, client):
        assert client.get("/health").status_code == 200

    def test_both_routes_503_naming_the_corpus_key(self, client):
        validate = client.post("/api/validate", json={"channels": ["SR:DIAG:BPM:01:X"]})
        channels = client.get("/api/channels")

        assert validate.status_code == 503
        assert channels.status_code == 503
        assert _TTL_KEY in _unavailable_text(validate.json())
        assert _TTL_KEY in _unavailable_text(channels.json())

    def test_the_detail_names_the_index_and_why_the_driver_refused_it(self, client, tmp_path):
        """A file that is there and unreadable is diagnosed, not just reported.

        The driver's own sentence travels with the absence, so an operator
        looking at the 503 can tell "nobody built it" from "what is there is
        not a database" -- and it names the file it opened to say so.
        """
        detail = client.get("/api/channels").json()["detail"]

        assert str(default_index_path(tmp_path)) in detail
        assert "DuckDB" in detail

    def test_an_index_that_was_never_built_reads_the_same_way(self, tmp_path):
        """The corpus is staged and nothing derived an index from it.

        Named as the config spells it, not as the server resolved it: the
        index path is relative (here, defaulted), and its resolved name is a
        path inside whatever tree this process happens to be serving from.
        """
        _write_corpus(tmp_path / "corpus.ttl", _CORPUS)

        with _started(_graph_config(tmp_path)) as client:
            resp = client.get("/api/channels")
            body = resp.json()

            assert resp.status_code == 503
            assert _TTL_KEY in _unavailable_text(body)
            assert "./data/channel_databases/graph.duckdb" in body["detail"]
            assert str(tmp_path) not in body["detail"]


class TestACorpusThatDeclaresNoChannels:
    """A corpus that parses and binds nothing is a seeding gap, not a facility."""

    @pytest.fixture
    def client(self, tmp_path: Path) -> Iterator[TestClient]:
        config = _graph_config(tmp_path)
        build_index_from_ttl(_write_corpus(tmp_path / "corpus.ttl", {}), config)
        with _started(config) as started:
            yield started

    def test_enumeration_503s_rather_than_serving_an_empty_facility(self, client):
        resp = client.get("/api/channels")

        assert resp.status_code == 503
        assert "declares no channels" in resp.json()["detail"]

    def test_membership_503s_rather_than_calling_every_channel_invalid(self, client):
        resp = client.post("/api/validate", json={"channels": ["SR:DIAG:BPM:01:X"]})

        assert resp.status_code == 503

    def test_the_app_still_starts(self, client):
        assert client.get("/health").status_code == 200


class TestARosterThatCannotSayWhichChannelsAreSettable:
    """Membership does not depend on direction, so it is still answerable."""

    def test_records_beside_an_absence_are_served(self, graph_client):
        from osprey.channel_roster import (
            ChannelRecord,
            RosterAbsence,
            RosterAbsenceReason,
            RosterResult,
            RosterSource,
            RosterSourceKind,
        )

        source = RosterSource(kind=RosterSourceKind.DATABASE, path=Path("/tmp/channels.json"))
        state = graph_client.app.state
        state.channel_roster = RosterResult(
            records=(ChannelRecord(address="FAC:PS:01:CURRENT", source=source),),
            source=source,
            absence=RosterAbsence(
                reason=RosterAbsenceReason.DIRECTION_UNDERIVABLE, path=source.path
            ),
        )
        state.channel_addresses = ("FAC:PS:01:CURRENT",)
        state.channel_address_index = frozenset(state.channel_addresses)

        validated = graph_client.post("/api/validate", json={"channels": ["FAC:PS:01:CURRENT"]})
        enumerated = graph_client.get("/api/channels")

        assert validated.json()["valid_count"] == 1
        assert enumerated.json()["total"] == 1


class TestTheFileBackedParadigmsAreUntouched:
    """The in-context paradigm answers both routes from its database, as before."""

    @pytest.fixture
    def database(self) -> MagicMock:
        db = MagicMock()
        db.get_all_channels.return_value = [{"channel": "SR:DIAG:BPM:01:X"}]
        db.chunk_database.return_value = [[{"channel": "SR:DIAG:BPM:01:X"}]]
        db.format_chunk_for_prompt.return_value = "SR:DIAG:BPM:01:X"
        db.validate_channels.return_value = [{"channel": "SR:DIAG:BPM:01:X", "valid": True}]
        db.get_valid_channels.return_value = ["SR:DIAG:BPM:01:X"]
        db.get_invalid_channels.return_value = []
        return db

    @pytest.fixture
    def client(self, database: MagicMock) -> Iterator[TestClient]:
        registry = MagicMock()
        registry.database = database
        registry.facility_name = "TEST"
        config = {
            "channel_finder": {
                "pipeline_mode": "in_context",
                "pipelines": {"in_context": {"database": {"path": "/tmp/db.json"}}},
            },
        }
        from osprey.interfaces.channel_finder.app import create_app

        with (
            patch(_CONFIG_SEAM, return_value=config),
            patch(
                "osprey.mcp_server.channel_finder_in_context.server_context"
                ".initialize_cf_ic_context",
                return_value=registry,
            ),
        ):
            with TestClient(create_app(project_cwd="/tmp/test-project")) as started:
                yield started

    def test_channels_still_come_from_the_database(self, client, database):
        resp = client.get("/api/channels")

        assert resp.status_code == 200
        assert resp.json() == {"channels": database.get_all_channels.return_value, "total": 1}

    def test_chunking_still_serves_the_prompt(self, client):
        resp = client.get("/api/channels?chunk_idx=0")

        assert resp.status_code == 200
        assert resp.json()["chunk_idx"] == 0
        assert resp.json()["formatted"] == "SR:DIAG:BPM:01:X"

    def test_validation_still_comes_from_the_database(self, client, database):
        resp = client.post("/api/validate", json={"channels": ["SR:DIAG:BPM:01:X"]})

        assert resp.status_code == 200
        assert resp.json()["valid_channels"] == ["SR:DIAG:BPM:01:X"]
        database.validate_channels.assert_called_once_with(["SR:DIAG:BPM:01:X"])

    def test_no_roster_is_read_for_a_file_backed_paradigm(self, client):
        """The database is that paradigm's enumeration; a second read is the bug."""
        assert getattr(client.app.state, "channel_roster", None) is None
