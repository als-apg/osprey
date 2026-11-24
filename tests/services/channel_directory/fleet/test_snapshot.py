"""The fleet snapshot (C5): its type, its JSON form, and its file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

pytest.importorskip("osprey_connectors.safety_file")

from osprey.services.channel_directory.fleet.snapshot import (
    Liveness,
    Outcome,
    ReadPathStatus,
    ServerFleetRow,
    Snapshot,
    SnapshotFormatError,
    SourceStatus,
    read_snapshot,
    snapshot_path,
    write_snapshot,
)
from osprey.services.channel_directory.fleet.states import (
    Finding,
    HealthReading,
    HealthStatus,
    LivenessKind,
    ReadPath,
    State,
)

FIXTURES = Path(__file__).parent / "fixtures"

C5_TOP = {
    "outcome",
    "target",
    "generated_at",
    "cycle",
    "period_s",
    "cycle_duration_s",
    "read_path",
    "sources",
    "counts",
    "group_by",
    "servers",
}
C5_ROW = {
    "name",
    "host",
    "class_name",
    "group",
    "state",
    "since",
    "reason",
    "findings",
    "directory_status",
    "listed",
    "sources",
    "liveness",
    "health",
}


def _row(name: str = "ioc-a", *, state: str = "alive", findings=(), **overrides) -> ServerFleetRow:
    fields = {
        "name": name,
        "host": "host-1",
        "class_name": "default",
        "group": None,
        "state": State(state),
        "since": 1_000.0,
        "reason": "counter moved",
        "findings": tuple(Finding(f) for f in findings),
        "directory_status": "Active",
        "listed": True,
        "sources": (),
        "liveness": Liveness(
            channel=f"{name}:HEARTBEAT", kind=LivenessKind.COUNTER, value=42, read_at=1_010.0
        ),
        "health": (HealthReading(channel=f"{name}:UPTIME", value=3.5, status=HealthStatus.OK),),
    }
    fields.update(overrides)
    return ServerFleetRow(**fields)


def _snapshot(rows=None, **overrides) -> Snapshot:
    fields = {
        "target": "live",
        "generated_at": 1_700_000_000.5,
        "cycle": 12,
        "period_s": 30.0,
        "cycle_duration_s": 2.25,
        "read_path": ReadPathStatus(status=ReadPath.OK, channel="SR:PROBE"),
        "sources": [
            SourceStatus(name="directory", outcome="ok"),
            SourceStatus(name="roster", outcome="unavailable", reason="timed out"),
        ],
        "group_by": "host",
        "servers": rows if rows is not None else [_row()],
    }
    fields.update(overrides)
    return Snapshot.ok(**fields)


def _order_case() -> dict:
    return yaml.safe_load((FIXTURES / "states.yaml").read_text())["order"]


# --- Type ---------------------------------------------------------------------------


def test_rows_are_in_c4_order() -> None:
    case = _order_case()
    rows = [_row(r["name"], state=r["state"], findings=r["findings"]) for r in case["rows"]]
    snapshot = _snapshot(rows)
    assert [row.name for row in snapshot.servers] == case["expect"]


def test_counts_by_state_and_finding_cover_every_value() -> None:
    snapshot = _snapshot(
        [
            _row("a", state="alive"),
            _row("b", state="alive", findings=["degraded", "alive_unexpected"]),
            _row("c", state="stalled", findings=["expected_running_not_alive"]),
            _row("d", state="unknown", reason="read failed"),
        ]
    )
    assert snapshot.counts.by_state == {
        "alive": 2,
        "stalled": 1,
        "silent": 0,
        "unreached": 0,
        "unknown": 1,
    }
    assert snapshot.counts.by_finding == {
        "degraded": 1,
        "expected_running_not_alive": 1,
        "expected_stopped_alive": 0,
        "alive_unexpected": 1,
    }


def test_empty_fleet_has_zero_counts() -> None:
    snapshot = _snapshot([])
    assert set(snapshot.counts.by_state.values()) == {0}
    assert snapshot.servers == ()


@pytest.mark.parametrize("field", ["host", "class_name", "group"])
def test_groups_by_the_configured_field_in_c4_order(field: str) -> None:
    rows = [
        _row("a", state="alive", **{field: "x"}),
        _row("b", state="stalled", **{field: "y"}),
        _row("c", state="silent", **{field: "x"}),
    ]
    if field == "class_name":
        rows.append(_row("d", state="alive", class_name="z"))
    else:
        rows.append(_row("d", state="alive", **{field: None}))
    groups = _snapshot(rows, group_by=field).groups()
    assert {k: [r.name for r in v] for k, v in groups.items()} == (
        {"x": ["c", "a"], "y": ["b"], "z": ["d"]}
        if field == "class_name"
        else {"x": ["c", "a"], "y": ["b"], None: ["d"]}
    )
    assert list(groups)[-1] in ("z", None)


def test_unknown_group_by_is_refused() -> None:
    with pytest.raises(ValueError, match="group_by"):
        _snapshot(group_by="rack")


def test_a_server_appears_once() -> None:
    with pytest.raises(ValueError, match="twice"):
        _snapshot([_row("a"), _row("a", state="stalled")])


def test_unknown_carries_no_finding() -> None:
    with pytest.raises(ValueError, match="unknown"):
        _row("a", state="unknown", findings=["degraded"])


def test_row_lookup() -> None:
    snapshot = _snapshot([_row("a"), _row("b")])
    assert snapshot.row("b").name == "b"
    assert snapshot.row("nope") is None


def test_snapshot_is_immutable() -> None:
    snapshot = _snapshot()
    with pytest.raises(AttributeError):
        snapshot.cycle = 13  # type: ignore[misc]


@pytest.mark.parametrize("outcome", ["no_snapshot", "not_refreshed", "cannot_list"])
def test_non_ok_outcomes_are_snapshots_with_a_reason(outcome: str) -> None:
    snapshot = Snapshot.not_ok(outcome, target="va", reason="because")
    assert snapshot.outcome is Outcome(outcome)
    assert snapshot.servers == ()
    assert snapshot.groups() == {}
    assert snapshot.to_json() == {"outcome": outcome, "target": "va", "reason": "because"}


def test_non_ok_needs_a_reason_and_ok_goes_through_ok() -> None:
    with pytest.raises(ValueError, match="reason"):
        Snapshot.not_ok("cannot_list", target="va", reason="")
    with pytest.raises(ValueError):
        Snapshot.not_ok("ok", target="va", reason="x")
    with pytest.raises(ValueError):
        Snapshot.not_ok("unavailable", target="va", reason="x")


# --- JSON form ------------------------------------------------------------------------


def test_json_has_exactly_the_c5_fields() -> None:
    document = _snapshot(
        [_row("a", findings=["degraded"]), _row("b", state="stalled", listed=False)]
    ).to_json()
    assert set(document) == C5_TOP
    assert document["outcome"] == "ok"
    assert document["read_path"] == {"status": "ok", "channel": "SR:PROBE"}
    assert document["sources"][1] == {
        "name": "roster",
        "outcome": "unavailable",
        "reason": "timed out",
    }
    assert set(document["counts"]) == {"by_state", "by_finding"}
    for row in document["servers"]:
        assert set(row) == C5_ROW
        assert set(row["liveness"]) == {"channel", "kind", "value", "read_at", "ambiguous"}
        for health in row["health"]:
            assert set(health) == {"channel", "value", "status"}
    assert [row["name"] for row in document["servers"]] == ["b", "a"]
    json.dumps(document, allow_nan=False)


def test_json_round_trips() -> None:
    snapshot = _snapshot(
        [
            _row("a", findings=["degraded"], group="g1", sources=("ccdb",)),
            _row(
                "b",
                state="silent",
                host=None,
                since=None,
                directory_status=None,
                liveness=Liveness(channel=None, kind=LivenessKind.VALUE, ambiguous=True),
                health=(
                    HealthReading(channel="b:T", value=None, status=HealthStatus.UNREAD),
                    HealthReading(channel="b:U", value="hot", status=HealthStatus.OUT_OF_RANGE),
                ),
            ),
        ],
        read_path=ReadPathStatus(status=ReadPath.UNVERIFIED, channel=None),
    )
    document = json.loads(json.dumps(snapshot.to_json()))
    again = Snapshot.from_json(document)
    assert again == snapshot
    assert again.to_json() == snapshot.to_json()


def test_counts_that_disagree_with_rows_are_refused() -> None:
    document = _snapshot().to_json()
    document["counts"]["by_state"]["alive"] = 7
    with pytest.raises(SnapshotFormatError, match="counts"):
        Snapshot.from_json(document)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("cycle"),
        lambda d: d.update(extra=1),
        lambda d: d.update(cycle=True),
        lambda d: d.update(cycle=1.5),
        lambda d: d.update(outcome="maybe"),
        lambda d: d["servers"][0].update(state="dead"),
        lambda d: d["servers"][0].update(listed="yes"),
        lambda d: d["servers"][0]["liveness"].update(kind="status"),
        lambda d: d["servers"][0]["health"][0].update(status="bad"),
        lambda d: d.update(servers={}),
        lambda d: d.update(read_path=None),
    ],
)
def test_malformed_json_is_refused(mutate) -> None:
    document = _snapshot().to_json()
    mutate(document)
    with pytest.raises(SnapshotFormatError):
        Snapshot.from_json(document)


def test_non_object_json_is_refused() -> None:
    with pytest.raises(SnapshotFormatError):
        Snapshot.from_json([1, 2])
    with pytest.raises(SnapshotFormatError):
        Snapshot.from_json({"outcome": "cannot_list", "target": "va"})


# --- File -------------------------------------------------------------------------------


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    snapshot = _snapshot([_row("a"), _row("b", state="stalled")])
    path = write_snapshot(tmp_path / "state", snapshot)
    assert path == snapshot_path(tmp_path / "state", "live")
    assert json.loads(path.read_text()) == snapshot.to_json()
    assert read_snapshot(tmp_path / "state", "live") == snapshot


def test_a_non_ok_outcome_round_trips_through_the_file(tmp_path: Path) -> None:
    snapshot = Snapshot.not_ok("not_refreshed", target="va", reason="served in process")
    write_snapshot(tmp_path, snapshot)
    assert read_snapshot(tmp_path, "va") == snapshot


def test_write_replaces_the_previous_file(tmp_path: Path) -> None:
    write_snapshot(tmp_path, _snapshot(cycle=1))
    write_snapshot(tmp_path, _snapshot(cycle=2))
    assert read_snapshot(tmp_path, "live").cycle == 2
    assert [p.name for p in tmp_path.iterdir()] == [snapshot_path(tmp_path, "live").name]


def test_write_failure_leaves_previous_file(tmp_path: Path, monkeypatch) -> None:
    from osprey.services.channel_directory.fleet import snapshot as module

    write_snapshot(tmp_path, _snapshot(cycle=1))

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(module, "atomic_replace", boom)
    with pytest.raises(OSError, match="disk full"):
        write_snapshot(tmp_path, _snapshot(cycle=2))
    assert read_snapshot(tmp_path, "live").cycle == 1


def test_missing_file_reads_as_no_snapshot(tmp_path: Path) -> None:
    snapshot = read_snapshot(tmp_path, "live")
    assert snapshot.outcome is Outcome.NO_SNAPSHOT
    assert snapshot.target == "live"
    assert snapshot.reason


@pytest.mark.parametrize("content", [b"{not json", b"[]", b'{"outcome": "ok"}', b"\xff\xfe"])
def test_corrupt_file_reads_as_no_snapshot(tmp_path: Path, content: bytes, caplog) -> None:
    snapshot_path(tmp_path, "live").write_bytes(content)
    snapshot = read_snapshot(tmp_path, "live")
    assert snapshot.outcome is Outcome.NO_SNAPSHOT
    assert "corrupt" in snapshot.reason
    assert "corrupt" in caplog.text


def test_a_file_naming_another_target_reads_as_no_snapshot(tmp_path: Path) -> None:
    write_snapshot(tmp_path, _snapshot(target="other"))
    snapshot_path(tmp_path, "other").rename(snapshot_path(tmp_path, "live"))
    snapshot = read_snapshot(tmp_path, "live")
    assert snapshot.outcome is Outcome.NO_SNAPSHOT
    assert "other" in snapshot.reason


@pytest.mark.parametrize("target", ["", "../x", "a/b", ".hidden"])
def test_target_names_that_escape_the_directory_are_refused(tmp_path: Path, target: str) -> None:
    with pytest.raises(ValueError):
        snapshot_path(tmp_path, target)
