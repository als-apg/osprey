"""The channel finder's graph paradigm answers membership and enumeration.

Two questions the graph paradigm used to refuse — "is this channel real?" (501)
and "which channels are there?" (404) — are answered here from the channel
roster, the one enumeration of a facility's channels. The store is never dialed
for either: the roster is the facility file the build writes at the root of
the render, read once when the app starts.

What these tests pin:

- Membership and enumeration answer from the roster, in the shapes the
  file-backed paradigms already answer them in.
- ``chunk_idx`` is refused (422) rather than honoured: chunking exists to cut
  the in-context paradigm's prompt into pieces, and the graph builds no prompt.
- A render no build has written a facility file into still *starts*: both
  routes answer 503 naming ``osprey build``, which is what an operator runs.
- The roster is read once at lifespan, not once per request.
- The file-backed paradigms are untouched.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import osprey.channel_roster as channel_roster
from osprey.services.channel_finder.databases import FlatChannelDatabase
from tests._facility_file import channel_tree, write_facility_file

_CONFIG_SEAM = "osprey.utils.workspace.load_osprey_config"
_GRAPH_CONTEXT_SEAM = "osprey.interfaces.channel_finder.app._make_graph_context"

#: The action every unavailable answer puts in front of an operator.
_REMEDY = "osprey build"

#: A facility small enough to assert against whole: one settable channel, its
#: readback and one more readable one, so membership, direction-blind
#: enumeration and ordering are all observable.
_CHANNELS = channel_tree(
    {"SR:MAG:QF:01:CURRENT:SP": "SR:MAG:QF:01:CURRENT:RB"}, readbacks=["SR:DIAG:BPM:01:X"]
)

#: The addresses :data:`_CHANNELS` declares, in the order enumeration serves them.
_ADDRESSES = ("SR:DIAG:BPM:01:X", "SR:MAG:QF:01:CURRENT:RB", "SR:MAG:QF:01:CURRENT:SP")


@pytest.fixture(autouse=True)
def cold_roster_cache() -> Iterator[None]:
    """Start and leave every test with an empty roster cache."""
    channel_roster._roster_cache.clear()
    yield
    channel_roster._roster_cache.clear()


def _graph_config(render: Path) -> dict[str, Any]:
    """A graph-paradigm project rendered into *render*."""
    return {
        "config_dir": str(render),
        "channel_finder": {"pipeline_mode": "graph", "pipelines": None},
        "services": {"graphdb": {"uri": "bolt://localhost:7687"}},
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
    """A started graph-mode app whose facility file holds :data:`_CHANNELS`."""
    write_facility_file(tmp_path, _CHANNELS)
    with _started(_graph_config(tmp_path)) as client:
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
            "channels": [{"channel": address} for address in _ADDRESSES],
            "total": 3,
        }

    def test_an_address_bound_twice_is_enumerated_once(self, tmp_path):
        """Otherwise the total disagrees with what membership can find."""
        (tmp_path / "facility.json").write_text(
            json.dumps({"channels": [{"id": "SR:DIAG:BPM:01:X", "role": "readback"}] * 2})
        )

        with _started(_graph_config(tmp_path)) as client:
            body = client.get("/api/channels").json()

            assert body == {"channels": [{"channel": "SR:DIAG:BPM:01:X"}], "total": 1}

    def test_chunk_idx_is_refused_as_an_in_context_contract(self, graph_client):
        resp = graph_client.get("/api/channels?chunk_idx=0")

        assert resp.status_code == 422
        assert "chunk_idx" in resp.json()["detail"]

    def test_chunk_idx_is_refused_even_when_it_would_be_in_range(self, graph_client):
        """Not a range check: the graph builds no prompt to chunk at all."""
        assert graph_client.get("/api/channels?chunk_idx=0&chunk_size=1").status_code == 422

    def test_the_facility_file_is_read_once_for_the_whole_process(self, tmp_path):
        config = _graph_config(tmp_path)
        write_facility_file(tmp_path, _CHANNELS)
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


class TestARenderWithNoFacilityFile:
    """A render no build has written a facility file into."""

    @pytest.fixture
    def client(self, tmp_path: Path) -> Iterator[TestClient]:
        with _started(_graph_config(tmp_path)) as started:
            yield started

    def test_the_app_still_starts_and_serves_the_graph_paradigm(self, client):
        assert client.get("/health").json()["pipeline_type"] == "graph"
        assert client.get("/api/info").json()["graph_backed"] is True

    def test_validate_503s_naming_the_build(self, client):
        resp = client.post("/api/validate", json={"channels": ["SR:DIAG:BPM:01:X"]})

        assert resp.status_code == 503
        assert _REMEDY in _unavailable_text(resp.json())

    def test_channels_503s_naming_the_build(self, client):
        resp = client.get("/api/channels")

        assert resp.status_code == 503
        assert _REMEDY in _unavailable_text(resp.json())

    def test_the_body_carries_the_remedy_the_other_graph_routes_carry(self, client):
        body = client.get("/api/channels").json()

        assert body["error_type"] == "service_unavailable"
        assert any(_REMEDY in s for s in body["suggestions"])
        assert not any("ttl_path" in s for s in body["suggestions"])

    def test_the_reason_is_the_roster_absence_verbatim(self, client, tmp_path):
        absence = channel_roster.registered_channels(_graph_config(tmp_path)).absence

        assert client.get("/api/channels").json()["detail"] == absence.message()


class TestAFacilityFileThatDeclaresNoChannels:
    """A facility file that holds no channel record is a seeding gap, not a facility."""

    @pytest.fixture
    def client(self, tmp_path: Path) -> Iterator[TestClient]:
        write_facility_file(tmp_path, None)
        with _started(_graph_config(tmp_path)) as started:
            yield started

    def test_enumeration_503s_rather_than_serving_an_empty_facility(self, client):
        resp = client.get("/api/channels")

        assert resp.status_code == 503
        assert "declares no channels" in resp.json()["detail"]

    def test_the_remedy_names_the_facility_tree_rather_than_the_build(self, client):
        suggestions = client.get("/api/channels").json()["suggestions"]

        assert any("data/facility" in s for s in suggestions)
        assert not any(_REMEDY in s for s in suggestions)

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

        source = RosterSource(kind=RosterSourceKind.FACILITY, path=Path("/tmp/facility.json"))
        state = graph_client.app.state
        state.channel_roster = RosterResult(
            records=(ChannelRecord(address="FAC:PS:01:CURRENT", source=source),),
            source=source,
            absence=RosterAbsence(
                reason=RosterAbsenceReason.CORRUPT_SOURCE,
                path=source.path,
                detail="one channel record states no role",
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
        db = MagicMock(spec=FlatChannelDatabase)
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
