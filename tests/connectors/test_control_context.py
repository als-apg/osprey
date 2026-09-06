"""Contract tests for the control-context record.

One record per deployment instance, written by its owner and read by every
chat session, notebook kernel, executor sandbox, sibling MCP server and
stdlib hook. The tests here pin the parts every one of those readers has to
agree on: where the file is, what a payload must contain to be honoured, how
a degraded file reads, and when a change is seen.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from osprey_connectors import control_context, session_store
from osprey_connectors.types import CONTROL_TARGETS

# --- fixtures --------------------------------------------------------------


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Stamp ``OSPREY_AGENT_DATA_ROOT`` at a scratch root, cache cleared."""
    monkeypatch.setenv(session_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
    control_context.invalidate_cache()
    yield tmp_path
    control_context.invalidate_cache()


@pytest.fixture
def rootless(monkeypatch):
    """No stamp and no derivable root: the record has nowhere to live."""
    monkeypatch.delenv(session_store.AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(
        session_store,
        "resolve_shared_data_root",
        lambda: (_ for _ in ()).throw(RuntimeError("no project root")),
    )
    control_context.invalidate_cache()
    yield
    control_context.invalidate_cache()


def _payload(**overrides):
    """A complete, valid payload; *overrides* replace or add top-level keys."""
    payload = {
        "schema": 1,
        "owner": {"kind": "web_terminal", "pid": os.getpid(), "port": 8080},
        "target": "va",
        "generation": 7,
        "posture": {"live": "sandbox"},
        "last_switch": {
            "request_id": "r-1",
            "target": "va",
            "requested_at": 1000.0,
            "requested_by": 4242,
            "status": "applied",
            "reason": None,
            "detail": None,
            "generation": 7,
        },
    }
    payload.update(overrides)
    return payload


def _write_raw(root: Path, payload, *, encoding: str = "utf-8") -> Path:
    path = control_context.record_path_under(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_bytes(text.encode(encoding))
    return path


# --- path resolution -------------------------------------------------------


def test_record_lives_beside_the_posture_store(data_root):
    assert control_context.STATE_DIR_NAME == "control_target"
    assert control_context.RECORD_FILENAME == "control_context.json"
    assert control_context.record_path() == data_root / "control_target" / "control_context.json"
    assert control_context.record_path() == control_context.record_path_under(data_root)
    assert control_context.state_dir() == session_store.state_dir()


def test_record_path_is_none_without_a_root(rootless):
    assert control_context.record_path() is None
    assert control_context.state_dir() is None


# --- writing ---------------------------------------------------------------


def test_write_then_read_round_trips(data_root):
    record = control_context.ControlContext(
        target="va",
        generation=7,
        owner=control_context.Owner(kind="web_terminal", pid=os.getpid(), port=8080),
        posture={"live": "sandbox"},
        last_switch={"request_id": "r-1", "status": "applied", "generation": 7},
    )
    path = control_context.write_record(record)
    assert path == control_context.record_path()
    assert control_context.read_record() == record


def test_write_creates_the_state_directory(data_root):
    assert not (data_root / "control_target").exists()
    control_context.write_record(control_context.ControlContext(target="live", generation=0))
    assert control_context.record_path().is_file()


def test_write_payload_carries_exactly_the_record_contract(data_root):
    control_context.write_record(
        control_context.ControlContext(
            target="standin",
            generation=2,
            owner=control_context.Owner(kind="controls_server", pid=4242),
        )
    )
    payload = json.loads(control_context.record_path().read_text(encoding="utf-8"))
    assert set(payload) == {"schema", "owner", "target", "generation", "posture", "last_switch"}
    assert payload["schema"] == control_context.SCHEMA_VERSION == 1
    assert payload["owner"] == {"kind": "controls_server", "pid": 4242, "port": None}
    assert payload["posture"] == {}
    assert payload["last_switch"] is None


def test_write_replaces_atomically_and_leaves_no_litter(data_root):
    first = control_context.write_record(
        control_context.ControlContext(target="live", generation=0)
    )
    before = first.stat().st_ino
    control_context.write_record(control_context.ControlContext(target="va", generation=1))
    assert first.stat().st_ino != before
    assert [p.name for p in (data_root / "control_target").iterdir()] == ["control_context.json"]


def test_write_raises_when_the_record_has_nowhere_to_live(rootless):
    with pytest.raises(RuntimeError):
        control_context.write_record(control_context.ControlContext(target="live", generation=0))


def test_refusal_terminus_moves_nothing_else(data_root):
    import dataclasses

    applied = control_context.ControlContext(
        target="va",
        generation=7,
        owner=control_context.Owner(kind="web_terminal", pid=os.getpid(), port=8080),
        posture={"live": "sandbox"},
        last_switch={"request_id": "r-1", "status": "applied", "generation": 7},
    )
    control_context.write_record(applied)
    refusal = dataclasses.replace(
        applied,
        last_switch={
            "request_id": "r-2",
            "target": "live",
            "status": "refused",
            "reason": "target_unreachable",
            "detail": "live did not answer",
            "generation": None,
        },
    )
    control_context.write_record(refusal)

    after = control_context.read_record()
    assert after.target == applied.target
    assert after.generation == applied.generation
    assert after.posture == applied.posture
    assert after.owner == applied.owner
    assert after.last_switch["status"] == "refused"
    assert after.last_switch["reason"] == "target_unreachable"


# --- reading: the record survives ------------------------------------------


def test_read_parses_a_hand_written_payload(data_root):
    _write_raw(data_root, _payload())
    record = control_context.read_record()
    assert record.target == "va"
    assert record.generation == 7
    assert record.owner == control_context.Owner(kind="web_terminal", pid=os.getpid(), port=8080)
    assert record.posture == {"live": "sandbox"}
    assert record.last_switch["request_id"] == "r-1"


@pytest.mark.parametrize("target", CONTROL_TARGETS)
def test_read_accepts_every_control_target(data_root, target):
    _write_raw(data_root, _payload(target=target))
    assert control_context.read_record().target == target


def test_read_accepts_an_explicit_path(tmp_path):
    path = tmp_path / "elsewhere.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    assert control_context.read_record(path=path).generation == 7


# --- reading: degraded outcomes are all None -------------------------------


def test_missing_record_reads_as_none(data_root):
    assert control_context.read_record() is None


def test_unresolvable_root_reads_as_none(rootless):
    assert control_context.read_record() is None


def test_unparseable_record_reads_as_none(data_root):
    _write_raw(data_root, "{not json")
    assert control_context.read_record() is None


def test_non_object_record_reads_as_none(data_root):
    _write_raw(data_root, [1, 2, 3])
    assert control_context.read_record() is None


def test_non_utf8_record_reads_as_none(data_root):
    path = control_context.record_path_under(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe{}")
    assert control_context.read_record() is None


def test_unreadable_record_reads_as_none(data_root):
    # A directory where the file belongs: the read fails with OSError, which
    # must arrive as "no record" like every other degraded outcome.
    control_context.record_path_under(data_root).mkdir(parents=True)
    assert control_context.read_record() is None


@pytest.mark.parametrize("schema", [0, 2, "1", None, True])
def test_wrong_schema_reads_as_none(data_root, schema):
    _write_raw(data_root, _payload(schema=schema))
    assert control_context.read_record() is None


def test_absent_schema_reads_as_none(data_root):
    payload = _payload()
    del payload["schema"]
    _write_raw(data_root, payload)
    assert control_context.read_record() is None


@pytest.mark.parametrize("target", ["", "banana", None, 7, ["va"]])
def test_unknown_target_reads_as_none(data_root, target):
    _write_raw(data_root, _payload(target=target))
    assert control_context.read_record() is None


def test_absent_target_reads_as_none(data_root):
    payload = _payload()
    del payload["target"]
    _write_raw(data_root, payload)
    assert control_context.read_record() is None


@pytest.mark.parametrize("generation", [-1, "7", 7.0, None, True])
def test_unusable_generation_reads_as_none(data_root, generation):
    _write_raw(data_root, _payload(generation=generation))
    assert control_context.read_record() is None


def test_absent_generation_reads_as_none(data_root):
    payload = _payload()
    del payload["generation"]
    _write_raw(data_root, payload)
    assert control_context.read_record() is None


# --- reading: the annotations degrade on their own -------------------------


@pytest.mark.parametrize(
    "owner",
    [
        None,
        {},
        {"kind": "web_terminal"},
        {"kind": "poltergeist", "pid": 4242},
        {"kind": "web_terminal", "pid": 0},
        {"kind": "web_terminal", "pid": "4242"},
        {"kind": "web_terminal", "pid": True},
        "web_terminal",
    ],
)
def test_unusable_owner_reads_as_ownerless_and_keeps_the_record(data_root, owner):
    _write_raw(data_root, _payload(owner=owner))
    record = control_context.read_record()
    assert record is not None
    assert record.owner is None
    assert record.target == "va"


def test_absent_owner_reads_as_ownerless(data_root):
    payload = _payload()
    del payload["owner"]
    _write_raw(data_root, payload)
    assert control_context.read_record().owner is None


@pytest.mark.parametrize("port", [None, "8080", 0, True, -1])
def test_unusable_owner_port_reads_as_no_port(data_root, port):
    _write_raw(data_root, _payload(owner={"kind": "controls_server", "pid": 4242, "port": port}))
    owner = control_context.read_record().owner
    assert owner == control_context.Owner(kind="controls_server", pid=4242, port=None)


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        ({"live": "sandbox"}, {"live": "sandbox"}),
        ({"live": "sandbox", "va": "writes"}, {"live": "sandbox"}),
        ({"live": "writes"}, {}),
        ("sandbox", dict.fromkeys(CONTROL_TARGETS, "sandbox")),
        ("writes", {}),
        ({}, {}),
        (None, {}),
        (["live"], {}),
        ({"live": True}, {}),
    ],
)
def test_posture_follows_the_store_entry_grammar(data_root, stored, expected):
    _write_raw(data_root, _payload(posture=stored))
    assert control_context.read_record().posture == expected


def test_absent_posture_reads_as_empty(data_root):
    payload = _payload()
    del payload["posture"]
    _write_raw(data_root, payload)
    assert control_context.read_record().posture == {}


@pytest.mark.parametrize("last_switch", [None, "applied", 7, ["applied"]])
def test_unusable_last_switch_reads_as_none_and_keeps_the_record(data_root, last_switch):
    _write_raw(data_root, _payload(last_switch=last_switch))
    record = control_context.read_record()
    assert record is not None
    assert record.last_switch is None
    assert record.generation == 7


def test_last_switch_is_kept_verbatim(data_root):
    terminus = {
        "request_id": "r-9",
        "target": "live",
        "requested_at": 1234.5,
        "requested_by": 99,
        "status": "refused",
        "reason": "busy",
        "detail": "sandbox 3f2a is writing",
        "generation": None,
    }
    _write_raw(data_root, _payload(last_switch=terminus))
    assert control_context.read_record().last_switch == terminus


# --- when a change is seen -------------------------------------------------


def test_an_unchanged_signature_is_not_re_parsed(data_root):
    path = _write_raw(data_root, _payload(generation=7))
    first = control_context.read_record()
    signature = path.stat()

    # Overwrite in place — same inode, same size — and restore the timestamp,
    # so the file's signature is exactly what it was. A reader that re-parsed
    # would see 8; the cache is what makes it still answer 7.
    with open(path, "r+b") as handle:
        handle.write(json.dumps(_payload(generation=8)).encode("utf-8"))
    os.utime(path, ns=(signature.st_atime_ns, signature.st_mtime_ns))
    assert path.stat().st_size == signature.st_size

    assert control_context.read_record() is first
    control_context.invalidate_cache()
    assert control_context.read_record().generation == 8


def test_an_atomic_replace_is_seen_even_at_the_same_mtime_and_size(data_root):
    control_context.write_record(control_context.ControlContext(target="va", generation=1))
    path = control_context.record_path()
    before = path.stat()
    assert control_context.read_record().generation == 1

    # Same length, same timestamp: only the inode moved, which is exactly the
    # case two switches inside one filesystem clock tick produce.
    control_context.write_record(control_context.ControlContext(target="va", generation=2))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_ino != before.st_ino

    assert control_context.read_record().generation == 2


def test_a_removed_record_stops_being_served_from_the_cache(data_root):
    control_context.write_record(control_context.ControlContext(target="va", generation=1))
    assert control_context.read_record() is not None
    control_context.record_path().unlink()
    assert control_context.read_record() is None


# --- parsing without a file ------------------------------------------------


def test_parse_record_accepts_text_and_objects():
    payload = _payload()
    assert control_context.parse_record(payload) == control_context.parse_record(
        json.dumps(payload)
    )
    assert control_context.parse_record("{not json") is None


# --- process liveness ------------------------------------------------------


def _kill_with_dead(dead_pids):
    """An ``os.kill`` that reports *dead_pids* gone and everything else alive."""

    def fake_kill(pid, sig):
        if pid in dead_pids:
            raise ProcessLookupError(pid)
        return None

    return fake_kill


def test_this_process_is_alive():
    assert control_context.is_process_alive(os.getpid()) is True


def test_a_gone_pid_is_not_alive(monkeypatch):
    monkeypatch.setattr(os, "kill", _kill_with_dead({4321}))
    assert control_context.is_process_alive(4321) is False


def test_a_process_we_may_not_signal_counts_as_alive(monkeypatch):
    def denied(pid, sig):
        raise PermissionError(pid)

    monkeypatch.setattr(os, "kill", denied)
    assert control_context.is_process_alive(4321) is True


def test_an_unexpected_os_error_counts_as_alive(monkeypatch):
    def odd(pid, sig):
        raise OSError("platform oddity")

    monkeypatch.setattr(os, "kill", odd)
    assert control_context.is_process_alive(4321) is True


@pytest.mark.parametrize("value", [0, -1, True, False, "4321", 4321.0, None, [4321]])
def test_nothing_but_a_positive_int_names_a_process(monkeypatch, value):
    """``True`` is ``1`` to ``os.kill``, and PID 1 is always alive."""

    def explode(pid, sig):  # pragma: no cover - must not be called
        raise AssertionError(f"os.kill called with {pid!r}")

    monkeypatch.setattr(os, "kill", explode)
    assert control_context.is_process_alive(value) is False


# --- the per-server reports ------------------------------------------------


def _report_payload(pid, **overrides):
    """A complete, valid report payload; *overrides* replace top-level keys."""
    payload = {
        "server_pid": pid,
        "session": "sess-1",
        "applied_target": "va",
        "applied_generation": 7,
        "children": [111, 222],
        "reachability": {"va": {"reachable": True, "probed_at": "2026-01-01T00:00:00+00:00"}},
        "last_switch": {"generation": 7, "status": "applied", "at": "2026-01-01T00:00:00+00:00"},
        "last_posture_realign": {"at": "2026-01-01T00:00:00+00:00"},
        "targets": {"va": {"selected_role": "operator"}},
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def _write_report(root: Path, pid, payload=None) -> Path:
    path = control_context.report_path_under(root, pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    return path


def test_reports_live_beside_the_record(data_root):
    assert control_context.REPORT_FILE_PREFIX == "server_"
    assert control_context.REPORT_FILE_GLOB == "server_*.json"
    expected = data_root / "control_target" / "server_4321.json"
    assert control_context.report_path(4321) == expected
    assert control_context.report_path_under(data_root, 4321) == expected


def test_report_path_is_none_without_a_root(rootless):
    assert control_context.report_path(4321) is None
    assert control_context.report_paths() == []


def test_a_report_round_trips_through_its_payload():
    payload = _report_payload(4321)
    report = control_context.parse_report(payload)
    assert report is not None
    assert report.to_payload() == payload
    assert control_context.parse_report(report.to_payload()) == report


@pytest.mark.parametrize("value", [None, 0, -1, True, "4321", 4321.0])
def test_a_report_without_a_server_pid_is_no_report(value):
    assert control_context.parse_report(_report_payload(value)) is None


def test_a_report_that_is_not_an_object_is_no_report():
    assert control_context.parse_report([1, 2, 3]) is None
    assert control_context.parse_report(None) is None


def test_a_server_that_has_not_answered_yet_reports_no_applied_target():
    report = control_context.parse_report(
        _report_payload(4321, applied_target=None, applied_generation=None)
    )
    assert report.applied_target is None
    assert report.applied_generation is None


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("session", 42, None),
        ("session", "", None),
        ("applied_target", "nowhere", None),
        ("applied_target", True, None),
        ("applied_generation", -1, None),
        ("applied_generation", True, None),
        ("applied_generation", "7", None),
        ("children", "111", ()),
        ("children", [111, 111, 0, "x", 222], (111, 222)),
        ("reachability", "up", {}),
        ("last_switch", "applied", None),
        ("last_posture_realign", 3, None),
        ("targets", [], {}),
        ("updated_at", 17, None),
    ],
)
def test_a_degraded_report_field_degrades_alone(field, value, expected):
    """The identity is ``server_pid``; every annotation on it stands or falls alone."""
    report = control_context.parse_report(_report_payload(4321, **{field: value}))
    assert report is not None
    assert report.server_pid == 4321
    assert getattr(report, field) == expected


def test_an_absent_or_degraded_report_file_reads_as_none(data_root):
    assert control_context.read_report(control_context.report_path(4321)) is None
    _write_report(data_root, 4322, "{not json")
    assert control_context.read_report(control_context.report_path(4322)) is None
    _write_report(data_root, 4323, "[1, 2, 3]")
    assert control_context.read_report(control_context.report_path(4323)) is None


def test_an_unchanged_report_signature_is_not_re_parsed(data_root):
    path = _write_report(data_root, 4321, _report_payload(4321, applied_generation=7))
    first = control_context.read_report(path)
    signature = path.stat()

    with open(path, "r+b") as handle:
        handle.write(json.dumps(_report_payload(4321, applied_generation=8)).encode("utf-8"))
    os.utime(path, ns=(signature.st_atime_ns, signature.st_mtime_ns))
    assert path.stat().st_size == signature.st_size

    assert control_context.read_report(path) is first
    control_context.invalidate_cache()
    assert control_context.read_report(path).applied_generation == 8


def test_a_removed_report_stops_being_served_from_the_cache(data_root):
    path = _write_report(data_root, 4321, _report_payload(4321))
    assert control_context.read_report(path) is not None
    path.unlink()
    assert control_context.read_report(path) is None


def test_caching_a_report_does_not_evict_the_record(data_root):
    control_context.write_record(control_context.ControlContext(target="va", generation=1))
    record = control_context.read_record()
    _write_report(data_root, 4321, _report_payload(4321))
    control_context.read_report(control_context.report_path(4321))
    assert control_context.read_record() is record


def test_live_reports_skip_dead_servers_and_degraded_files(data_root, monkeypatch):
    _write_report(data_root, os.getpid(), _report_payload(os.getpid()))
    _write_report(data_root, 4321, _report_payload(4321))
    _write_report(data_root, 4322, "{not json")
    monkeypatch.setattr(os, "kill", _kill_with_dead({4321}))

    assert [r.server_pid for r in control_context.live_reports()] == [os.getpid()]


def test_live_reports_ask_the_liveness_predicate_they_are_given(data_root):
    _write_report(data_root, 4321, _report_payload(4321))
    assert control_context.live_reports(is_alive=lambda pid: False) == []
    assert len(control_context.live_reports(is_alive=lambda pid: True)) == 1


def test_live_report_payloads_keep_the_keys_this_version_does_not_know(data_root):
    _write_report(data_root, os.getpid(), _report_payload(os.getpid(), owner_ppid=os.getppid()))

    payloads = control_context.live_report_payloads(control_context.report_paths())

    assert [p["owner_ppid"] for p in payloads] == [os.getppid()]


def test_live_report_payloads_drop_a_report_whose_server_pid_is_a_bool(data_root, monkeypatch):
    """``True`` would reach ``os.kill(1, 0)`` and read as alive forever."""
    _write_report(data_root, 4321, _report_payload(True))

    def explode(pid, sig):  # pragma: no cover - must not be called
        raise AssertionError(f"os.kill called with {pid!r}")

    monkeypatch.setattr(os, "kill", explode)
    assert control_context.live_report_payloads(control_context.report_paths()) == []


# --- sweeping --------------------------------------------------------------


def _sweep(directory, **kwargs):
    return control_context.sweep_dead(
        directory,
        prefix=control_context.REPORT_FILE_PREFIX,
        suffix=control_context.REPORT_FILE_SUFFIX,
        **kwargs,
    )


def test_a_dead_owners_file_is_swept(data_root, monkeypatch):
    dead = _write_report(data_root, 4321, _report_payload(4321))
    monkeypatch.setattr(os, "kill", _kill_with_dead({4321}))

    assert _sweep(control_context.state_dir()) == [dead]
    assert not dead.exists()


def test_a_live_owners_file_is_left_alone(data_root, monkeypatch):
    alive = _write_report(data_root, 4321, _report_payload(4321))
    monkeypatch.setattr(os, "kill", _kill_with_dead(set()))

    assert _sweep(control_context.state_dir()) == []
    assert alive.exists()


def test_the_kept_file_is_never_probed(data_root, monkeypatch):
    own = _write_report(data_root, 1234, _report_payload(1234))
    monkeypatch.setattr(os, "kill", _kill_with_dead({1234}))

    assert _sweep(control_context.state_dir(), keep=own.name) == []
    assert own.exists()


def test_a_filename_that_encodes_no_pid_is_swept(data_root):
    directory = data_root / "control_target"
    directory.mkdir(parents=True)
    junk = directory / "server_notapid.json"
    junk.write_text("{}", encoding="utf-8")

    assert _sweep(directory) == [junk]
    assert not junk.exists()


def test_salvage_sees_the_file_before_it_goes(data_root, monkeypatch):
    _write_report(data_root, 4321, _report_payload(4321))
    monkeypatch.setattr(os, "kill", _kill_with_dead({4321}))
    seen = []

    _sweep(
        control_context.state_dir(),
        salvage=lambda path: seen.append(control_context.read_report(path)),
    )

    assert [r.server_pid for r in seen] == [4321]


def test_a_missing_directory_sweeps_to_empty(data_root):
    assert _sweep(data_root / "control_target") == []


# --- convergence -----------------------------------------------------------


def _context(**overrides):
    """The record convergence is judged against; *overrides* replace fields."""
    fields = {"target": "va", "generation": 7}
    fields.update(overrides)
    return control_context.ControlContext(**fields)


def _stamp(offset_s, *, tz=UTC):
    """An ISO-8601 wall clock *offset_s* from now, in *tz* (``None``: naive)."""
    moment = datetime.now(UTC) + timedelta(seconds=offset_s)
    return (moment.astimezone(tz) if tz is not None else moment.replace(tzinfo=None)).isoformat()


def _report(pid, **overrides):
    """One parsed report; the defaults are a server applied at ``va``/7."""
    fields = {
        "server_pid": pid,
        "session": "sess-1",
        "applied_target": "va",
        "applied_generation": 7,
        "last_switch": {"generation": 7, "status": "applied", "at": _stamp(-60)},
    }
    fields.update(overrides)
    return control_context.ServerReport(**fields)


def _applying(pid, *, generation=7, expires_in=300.0, session="sess-1", **overrides):
    """A server mid-swap: bound at the previous generation, ``applying`` at this one."""
    block = {"generation": generation, "status": "applying", "at": _stamp(-1)}
    if expires_in is not None:
        block["expires_at"] = _stamp(expires_in)
    fields = {
        "session": session,
        "applied_target": "live",
        "applied_generation": generation - 1,
        "last_switch": block,
    }
    fields.update(overrides)
    return _report(pid, **fields)


def test_an_empty_fleet_is_converged():
    """No live server: the record alone routes, and nothing is mid-swap."""
    record = _context()
    for session in (None, "sess-1", "kernel:abc"):
        assert control_context.converged(record, [], session) is True
        assert control_context.blocking_pids(record, [], session) == ()


def test_an_applying_report_leaves_every_session_unconverged():
    record = _context()
    reports = [_applying(4321)]

    for session in (None, "sess-1", "sess-other", "kernel:abc"):
        assert control_context.converged(record, reports, session) is False
        assert control_context.blocking_pids(record, reports, session) == (4321,)


def test_an_applying_report_for_another_generation_leaves_the_fleet_converged():
    """The record moved past that swap; the server re-adopts on its next tick."""
    record = _context()
    reports = [_applying(4321, generation=6, applied_target="va", applied_generation=7)]

    assert control_context.converged(record, reports, "sess-other") is True


def test_an_applying_report_with_no_bound_is_never_converged():
    """Only the publishing server knows the bound; without one there is none."""
    record = _context()
    reports = [_applying(4321, expires_in=None)]

    assert control_context.converged(record, reports, None) is False
    far_future = datetime.now(UTC).timestamp() + 86_400
    assert control_context.converged(record, reports, None, now=far_future) is False


@pytest.mark.parametrize("value", ["", "soon", 1234, None, True, {"at": "later"}])
def test_an_applying_report_with_an_unparseable_bound_is_never_converged(value):
    record = _context()
    block = {"generation": 7, "status": "applying", "at": _stamp(-1), "expires_at": value}
    reports = [_applying(4321, expires_in=None, last_switch=block)]

    assert control_context.converged(record, reports, None) is False


def test_an_expired_applying_report_leaves_only_its_own_session_unconverged():
    """Past its bound it counts as failed, and the refusal names the stale pid."""
    record = _context()
    reports = [_applying(4321, expires_in=-1.0, session="sess-1")]

    assert control_context.converged(record, reports, "sess-1") is False
    assert control_context.blocking_pids(record, reports, "sess-1") == (4321,)
    for session in (None, "sess-other", "kernel:abc"):
        assert control_context.converged(record, reports, session) is True


def test_an_applying_report_is_unconverged_up_to_its_bound_inclusive():
    record = _context()
    expires_at = datetime.now(UTC) + timedelta(seconds=30)
    block = {"generation": 7, "status": "applying", "expires_at": expires_at.isoformat()}
    reports = [_applying(4321, expires_in=None, last_switch=block)]

    assert control_context.converged(record, reports, None, now=expires_at.timestamp()) is False
    assert (
        control_context.converged(record, reports, None, now=expires_at.timestamp() + 0.001) is True
    )


def test_a_naive_bound_is_unconverged_as_utc():
    """Two processes on one host write the same wall clock, tz-stamped or not."""
    record = _context()
    block = {"generation": 7, "status": "applying", "expires_at": _stamp(300, tz=None)}
    reports = [_applying(4321, expires_in=None, last_switch=block)]

    assert control_context.converged(record, reports, None) is False


@pytest.mark.parametrize("block", [None, "applying", {"status": "applying"}, {"generation": 7}])
def test_a_block_naming_no_applying_generation_leaves_everyone_converged(block):
    record = _context()
    reports = [_report(4321, session="sess-other", last_switch=block)]

    assert control_context.converged(record, reports, None) is True


def test_a_session_applied_at_the_records_target_and_generation_is_converged():
    record = _context()
    reports = [_report(4321, session="sess-1")]

    assert control_context.converged(record, reports, "sess-1") is True


@pytest.mark.parametrize(
    ("applied_target", "applied_generation"),
    [("va", 6), ("live", 7), ("live", 6)],
)
def test_a_bound_report_off_the_record_leaves_its_own_session_unconverged(
    applied_target, applied_generation
):
    """Both halves of the binding must match — a target alone is not arrival."""
    record = _context()
    reports = [
        _report(
            4321,
            session="sess-1",
            applied_target=applied_target,
            applied_generation=applied_generation,
        )
    ]

    assert control_context.converged(record, reports, "sess-1") is False
    assert control_context.blocking_pids(record, reports, "sess-1") == (4321,)
    for session in (None, "sess-other", "kernel:abc"):
        assert control_context.converged(record, reports, session) is True


@pytest.mark.parametrize(
    ("applied_target", "applied_generation"),
    [(None, None), ("va", None), (None, 7), (None, 6)],
)
def test_a_report_that_is_not_bound_yet_leaves_everyone_converged(
    applied_target, applied_generation
):
    """Null is "this server has not got there yet", never "baseline"."""
    record = _context()
    reports = [
        _report(
            4321,
            session="sess-1",
            applied_target=applied_target,
            applied_generation=applied_generation,
            last_switch=None,
        )
    ]

    assert control_context.converged(record, reports, "sess-1") is True


def test_a_failed_swap_leaves_only_its_own_session_unconverged():
    record = _context()
    failed = {"generation": 7, "status": "failed", "reason": "spawn_failed", "at": _stamp(-5)}
    reports = [
        _report(4321, session="sess-a", applied_generation=6, last_switch=failed),
        _report(4322, session="sess-b"),
    ]

    assert control_context.converged(record, reports, "sess-a") is False
    assert control_context.blocking_pids(record, reports, "sess-a") == (4321,)
    for session in (None, "sess-b", "kernel:abc"):
        assert control_context.converged(record, reports, session) is True


def test_a_failed_first_launch_leaves_its_own_session_unconverged():
    """Nothing came up, so there is no binding to mismatch — the block says it."""
    record = _context()
    failed = {"generation": 7, "status": "failed", "reason": "spawn_failed", "at": _stamp(-5)}
    reports = [
        _report(
            4321,
            session="sess-1",
            applied_target=None,
            applied_generation=None,
            last_switch=failed,
        )
    ]

    assert control_context.converged(record, reports, "sess-1") is False
    assert control_context.converged(record, reports, None) is True


def test_a_failure_at_an_older_generation_is_converged():
    """The server failed, then arrived: what it is bound to is the answer."""
    record = _context()
    failed = {"generation": 6, "status": "failed", "at": _stamp(-600)}
    reports = [_report(4321, session="sess-1", last_switch=failed)]

    assert control_context.converged(record, reports, "sess-1") is True


def test_a_near_miss_session_name_stays_converged():
    record = _context()
    reports = [_report(4321, session="sess-1", applied_generation=6)]

    assert control_context.converged(record, reports, "sess-10") is True
    assert control_context.converged(record, reports, "SESS-1") is True


@pytest.mark.parametrize("session", [None, ""])
def test_a_session_less_reader_owns_no_report_and_stays_converged(session):
    """A bare ``claude`` and the owner's fleet check are blocked only by applying."""
    record = _context()
    reports = [_report(4321, session=None, applied_generation=6)]

    assert control_context.converged(record, reports, session) is True


def test_a_kernel_session_owns_no_report_and_stays_converged():
    record = _context()
    reports = [_report(4321, session="sess-1", applied_generation=6)]

    assert control_context.converged(record, reports, "kernel:abcd1234") is True


def test_every_unconverged_reason_is_named_once_and_in_pid_order():
    record = _context()
    reports = [
        _applying(4322, session="sess-other"),
        _report(4321, session="sess-1", applied_generation=6),
        _applying(4320, session="sess-1"),
    ]

    assert control_context.blocking_pids(record, reports, "sess-1") == (4320, 4321, 4322)
    assert control_context.converged(record, reports, "sess-1") is False


def test_converged_is_the_absence_of_a_reason():
    record = _context()
    reports = [_report(4321, session="sess-1", applied_generation=6)]

    for session in (None, "sess-1", "kernel:abc"):
        blocked = control_context.blocking_pids(record, reports, session)
        assert control_context.converged(record, reports, session) is (not blocked)


def test_converged_reads_a_one_shot_iterable_of_reports_once():
    record = _context()
    reports = iter([_applying(4321)])

    assert control_context.blocking_pids(record, reports, "sess-1") == (4321,)


def test_converged_reads_no_files(rootless):
    """A pure judgement over parsed data: sandboxes and kernels call it hot."""
    record = _context()
    assert control_context.converged(record, [_report(4321)], "sess-1") is True


# --- the shared test fixtures ----------------------------------------------
#
# ``tests/_control_context_fixtures.py`` is what every suite downstream of the
# control target writes its record and its reports with. It emits the payloads
# the production readers parse, so these tests are the one place that proves
# the two agree — a field that moves in the schema fails here first, and in
# fifteen unrelated suites a second later.


def test_the_record_fixture_round_trips_through_read_record(data_root):
    """What the fixture writes is what ``read_record`` gives back."""
    from tests._control_context_fixtures import write_control_context

    path = write_control_context(
        data_root,
        target="va",
        generation=7,
        posture={"va": session_store.POSTURE_SANDBOX},
        last_switch={"request_id": "req-1", "status": control_context.SWITCH_APPLIED},
    )

    assert path == control_context.record_path()
    record = control_context.read_record()
    assert record is not None
    assert record.target == "va"
    assert record.generation == 7
    assert record.posture == {"va": session_store.POSTURE_SANDBOX}
    assert record.last_switch == {"request_id": "req-1", "status": control_context.SWITCH_APPLIED}


def test_the_record_fixture_defaults_to_an_owner_that_is_alive(data_root):
    """Ownership is what makes a record actionable, so the default is live.

    A fixture that defaulted to no owner — or to a PID that has exited — would
    silently put every suite on the claim path, where a reader is entitled to
    take the record over instead of following it.
    """
    from tests._control_context_fixtures import write_control_context

    write_control_context(data_root)

    record = control_context.read_record()
    assert record is not None
    assert record.owner is not None
    assert record.owner.kind == control_context.OWNER_WEB_TERMINAL
    assert record.owner.pid == os.getpid()
    assert control_context.is_process_alive(record.owner.pid) is True


def test_the_record_fixture_writes_an_ownerless_record_on_request(data_root):
    from tests._control_context_fixtures import write_control_context

    write_control_context(data_root, owned_by=None)

    record = control_context.read_record()
    assert record is not None
    assert record.owner is None


def test_the_record_fixture_owns_by_a_named_process_on_request(data_root):
    from tests._control_context_fixtures import owner, write_control_context

    write_control_context(
        data_root,
        owned_by=owner(control_context.OWNER_CONTROLS_SERVER, pid=4321),
    )

    record = control_context.read_record()
    assert record is not None
    assert record.owner == control_context.Owner(
        kind=control_context.OWNER_CONTROLS_SERVER, pid=4321, port=None
    )


def test_the_report_fixture_round_trips_through_read_report(data_root):
    """Same contract for the ten-field report the readers walk per server."""
    from tests._control_context_fixtures import write_server_report

    path = write_server_report(
        data_root,
        4321,
        session="sess-1",
        applied_target="va",
        applied_generation=7,
        children=[9001, 9002],
        reachability={"va": {"reachable": True}},
        last_switch={"generation": 7, "status": control_context.REPORT_APPLIED},
        last_posture_realign={"at": "2026-01-01T00:00:00Z"},
        targets={"va": {"selected_role": "primary"}},
        updated_at="2026-01-01T00:00:01Z",
    )

    assert path == control_context.report_path(4321)
    report = control_context.read_report(path)
    assert report == control_context.ServerReport(
        server_pid=4321,
        session="sess-1",
        applied_target="va",
        applied_generation=7,
        children=(9001, 9002),
        reachability={"va": {"reachable": True}},
        last_switch={"generation": 7, "status": control_context.REPORT_APPLIED},
        last_posture_realign={"at": "2026-01-01T00:00:00Z"},
        targets={"va": {"selected_role": "primary"}},
        updated_at="2026-01-01T00:00:01Z",
    )


def test_the_report_fixture_defaults_to_a_server_that_has_answered_nothing(data_root):
    """Null-bound, not baseline: a fresh server has not got anywhere yet."""
    from tests._control_context_fixtures import write_server_report

    write_server_report(data_root, 4321)

    report = control_context.read_report(control_context.report_path(4321))
    assert report is not None
    assert report.applied_target is None
    assert report.applied_generation is None
    assert report.children == ()
    assert report.reachability == {}
    assert report.last_switch is None


def test_the_fixtures_are_seen_when_they_rewrite_within_one_clock_tick(data_root):
    """The reader caches on ``(mtime_ns, size, ino)``; the fixture drops it.

    Two writes of the same size inside one tick are exactly the case that key
    cannot distinguish, and a suite that flips a posture back and forth does it
    on every other line.
    """
    from tests._control_context_fixtures import write_control_context

    write_control_context(data_root, target="va", generation=7)
    assert control_context.read_record().generation == 7

    write_control_context(data_root, target="va", generation=8)
    assert control_context.read_record().generation == 8


def test_the_record_fixture_writes_a_non_mapping_posture_verbatim(data_root):
    """The posture parsers' degradation cases have to be statable.

    A bare ``"sandbox"`` is the shape the session-wide posture wrote before
    targets existed, and both parsers still have a rule for it. A fixture that
    could only write mappings would leave every such suite hand-rolling a
    payload beside this writer.
    """
    from tests._control_context_fixtures import write_control_context

    path = write_control_context(data_root, posture="sandbox")

    assert json.loads(path.read_text(encoding="utf-8"))["posture"] == "sandbox"
    record = control_context.read_record()
    assert record is not None
    assert record.posture == dict.fromkeys(CONTROL_TARGETS, session_store.POSTURE_SANDBOX)


def test_the_fixture_writes_a_degraded_record_through_the_raw_payload_hatch(data_root):
    """A record too broken to build from the dataclass still has to be writable."""
    from tests._control_context_fixtures import write_payload

    write_payload(control_context.record_path_under(data_root), {"target": "va"})

    assert control_context.read_record() is None
