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
import stat
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from osprey_connectors import _control_state, control_context, testing, write_switch
from osprey_connectors.connector_types import CONTROL_TARGETS
from osprey_connectors.errors import NaiveDatetimeError, StateUnavailableError
from osprey_connectors.model import evolve
from osprey_connectors.refusals import VerdictReason
from osprey_connectors_tests._paths import REPO_ROOT

# --- fixtures --------------------------------------------------------------


@pytest.fixture(autouse=True)
def proc_root(tmp_path_factory, monkeypatch):
    """An empty ``/proc``, so a PID faked through ``os.kill`` never meets a real thread."""
    root = tmp_path_factory.mktemp("proc")
    monkeypatch.setattr(_control_state, "_PROC_ROOT", str(root))
    return root


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """Stamp ``OSPREY_AGENT_DATA_ROOT`` at a scratch root, cache cleared.

    The container bind is cleared with it: it is the rung ABOVE the stamp, so a
    suite run inside a deployment that carries one would resolve every path in
    this module to that deployment's directory and never look at the root the
    test just laid down.
    """
    monkeypatch.setenv(_control_state._AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
    monkeypatch.delenv(write_switch.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    testing.invalidate_cache()
    yield tmp_path
    testing.invalidate_cache()


@pytest.fixture
def rootless(monkeypatch):
    """No stamp, no bind and no derivable root: the record has nowhere to live."""
    monkeypatch.delenv(_control_state._AGENT_DATA_ROOT_ENV_VAR, raising=False)
    monkeypatch.delenv(write_switch.CONTROL_CONTEXT_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(
        _control_state,
        "_config_agent_data_root",
        lambda: (_ for _ in ()).throw(RuntimeError("no project root")),
    )
    testing.invalidate_cache()
    yield
    testing.invalidate_cache()


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
            "requested_at": "2026-01-01T00:00:00+00:00",
            "requested_by": "pid:4242",
            "status": "applied",
            "reason": None,
            "detail": "The control target is now 'va' (generation 7).",
            "generation": 7,
        },
    }
    payload.update(overrides)
    return payload


def _write_raw(root: Path, payload, *, encoding: str = "utf-8") -> Path:
    path = control_context._record_path_under(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_bytes(text.encode(encoding))
    return path


# --- path resolution -------------------------------------------------------


def test_record_lives_in_the_state_dir(data_root, monkeypatch):
    """One file per identity, under one directory, named the same by both readers.

    The identity is pinned rather than taken from the account running pytest:
    this is the one test that spells the hops literally, and a literal that
    reads ``getpass.getuser()``'s answer states nothing about the layout.
    """
    from osprey_connectors_tests._harness import pin_identity

    identity = pin_identity(monkeypatch)

    assert write_switch.CONTROL_STATE_DIR_NAME == "control_target"
    assert control_context.RECORD_FILENAME == "control_context.json"
    assert (
        control_context.record_path()
        == data_root / "control_target" / identity / "control_context.json"
    )
    assert control_context.record_path() == control_context._record_path_under(data_root)


@pytest.mark.usefixtures("rootless")
def test_record_path_is_none_without_a_root():
    assert control_context.record_path() is None
    assert write_switch.state_dir() is None


# --- writing ---------------------------------------------------------------


@pytest.mark.usefixtures("data_root")
def test_write_then_read_round_trips():
    record = control_context.ControlContext(
        target="va",
        generation=7,
        writer=control_context.RecordWriter(kind="web_terminal", pid=os.getpid(), port=8080),
        last_switch=control_context.SwitchTerminus.applied(
            request_id="r-1", target="va", generation=7, requested_at=None, requested_by="pid:1"
        ),
    )
    path = control_context.write_record(record)
    assert path == control_context.record_path()
    stored = control_context.read_record()
    # A first write stamps the epoch (rule 3); every caller-set field round-trips.
    assert stored == evolve(record, switch_epoch=stored.switch_epoch)
    assert isinstance(stored.switch_epoch, int) and stored.switch_epoch > 0


@pytest.mark.usefixtures("data_root")
def test_a_narrowing_input_is_not_stored():
    """No writer takes a narrowing: the JSON form writes the mirror of the switch."""
    record = control_context.ControlContext(target="live", generation=1, switch={"va": "on"})
    assert record.model_dump(mode="json")["posture"] == {"live": "sandbox", "standin": "sandbox"}

    control_context.write_record(record)
    payload = json.loads(control_context.record_path().read_text(encoding="utf-8"))
    # write_record keeps the on-disk switch (empty on a fresh root), and the
    # mirror follows what is stored, not what the caller passed.
    assert payload["switch"] == {}
    assert payload["posture"] == dict.fromkeys(CONTROL_TARGETS, write_switch.LAUNCH_PIN_SANDBOX)


@pytest.mark.parametrize(
    ("switch", "mirror"),
    [
        ({}, {"live": "sandbox", "va": "sandbox", "standin": "sandbox"}),
        ({"live": "on"}, {"va": "sandbox", "standin": "sandbox"}),
        ({"live": "on", "va": "on", "standin": "on"}, {}),
        ({"elsewhere": "on"}, {"live": "sandbox", "va": "sandbox", "standin": "sandbox"}),
    ],
    ids=["all-off", "live-on", "all-on", "unknown-key"],
)
def test_the_posture_mirror_refuses_every_machine_not_turned_on(switch, mirror):
    assert control_context._posture_mirror(switch) == mirror


@pytest.mark.usefixtures("data_root")
def test_the_stored_mirror_follows_the_switch():
    control_context.write_record(control_context.ControlContext(target="live", generation=0))

    def stored() -> dict:
        return json.loads(control_context.record_path().read_text(encoding="utf-8"))

    epoch = stored()["switch_epoch"]
    control_context.set_switch(["live"], "asked", epoch=epoch)
    assert stored()["posture"] == {"va": "sandbox", "standin": "sandbox"}

    control_context.set_switch(["live"], "off")
    assert stored()["posture"] == dict.fromkeys(CONTROL_TARGETS, write_switch.LAUNCH_PIN_SANDBOX)


def test_write_creates_the_state_directory(data_root):
    assert not (data_root / "control_target").exists()
    control_context.write_record(control_context.ControlContext(target="live", generation=0))
    assert control_context.record_path().is_file()
    assert control_context.record_path().parent == write_switch.state_dir()


@pytest.mark.usefixtures("data_root")
def test_write_payload_carries_exactly_the_record_contract():
    control_context.write_record(
        control_context.ControlContext(
            target="standin",
            generation=2,
            writer=control_context.RecordWriter(kind="controls_server", pid=4242),
        )
    )
    payload = json.loads(control_context.record_path().read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema",
        "owner",
        "target",
        "generation",
        "posture",
        "last_switch",
        "switch",
        "switch_epoch",
    }
    assert payload["schema"] == control_context._SCHEMA_VERSION == 1
    assert payload["owner"] == {"kind": "controls_server", "pid": 4242, "port": None}
    assert payload["posture"] == dict.fromkeys(CONTROL_TARGETS, write_switch.LAUNCH_PIN_SANDBOX)
    assert payload["last_switch"] is None
    assert payload["switch"] == {}
    assert isinstance(payload["switch_epoch"], int) and payload["switch_epoch"] > 0


@pytest.mark.usefixtures("data_root")
def test_write_replaces_atomically_and_leaves_no_litter():
    first = control_context.write_record(
        control_context.ControlContext(target="live", generation=0)
    )
    before = first.stat().st_ino
    control_context.write_record(control_context.ControlContext(target="va", generation=1))
    assert first.stat().st_ino != before
    # The record and its lock (rule 3) — no temp file left behind.
    assert sorted(p.name for p in write_switch.state_dir().iterdir()) == [
        "control_context.json",
        "control_context.json.lock",
    ]


@pytest.mark.usefixtures("data_root")
def test_a_written_record_is_group_readable_and_not_owner_only():
    """Mode 0640 exactly: the worker uid reads the record through the shared group."""
    state = write_switch.state_dir()
    state.mkdir(parents=True)
    os.chmod(state, 0o2770)

    path = control_context.write_record(control_context.ControlContext(target="live", generation=0))

    assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.usefixtures("data_root")
def test_every_state_dir_file_is_written_group_readable_mode(tmp_path):
    """Not just the record — reports and switch requests take the same mode."""
    target = tmp_path / "state" / "server_4242.json"

    control_context.write_json_atomic(target, {"schema": 1})

    assert stat.S_IMODE(target.stat().st_mode) == 0o640


@pytest.mark.usefixtures("data_root")
def test_the_record_lands_group_readable_under_an_owner_only_umask():
    """The mode is set on the file itself, so umask 077 cannot narrow it."""
    previous = os.umask(0o077)
    try:
        path = control_context.write_record(
            control_context.ControlContext(target="live", generation=0)
        )
    finally:
        os.umask(previous)

    assert stat.S_IMODE(path.stat().st_mode) == 0o640


@pytest.mark.usefixtures("data_root")
def test_a_writer_created_state_dir_is_exactly_mode_2770():
    """Setgid group-shared, and never left world-readable by the writer."""
    control_context.write_record(control_context.ControlContext(target="live", generation=0))

    directory = control_context.record_path().parent
    assert stat.S_IMODE(directory.stat().st_mode) == 0o2770


def test_every_level_the_writer_creates_is_exactly_mode_2770(data_root):
    """The tree root as well, not only the directory the record lands in.

    A writer that reaches a deployment's tree before the deploy path does
    creates the root on the way to its own directory. A root left at the umask
    lets ``other`` list every identity name under it, however tight the
    directories below it are.
    """
    control_context.write_record(control_context.ControlContext(target="live", generation=0))

    tree_root = control_context.record_path().parent.parent
    assert tree_root != data_root, "the row needs a root the writer itself created"
    assert stat.S_IMODE(tree_root.stat().st_mode) == 0o2770


@pytest.mark.usefixtures("data_root")
def test_an_existing_state_dir_keeps_the_mode_it_was_provisioned_with():
    """Only a directory the writer creates gets a mode; deployment owns the rest.

    The provisioned directory is this identity's own — the one the record lands
    in. Provisioning the parent instead would leave the writer to create the
    identity hop and set ITS mode, so the assertion would pass on a directory
    nobody had provisioned.
    """
    state = write_switch.state_dir()
    state.mkdir(parents=True)
    os.chmod(state, 0o2750)

    control_context.write_record(control_context.ControlContext(target="live", generation=0))

    assert stat.S_IMODE(state.stat().st_mode) == 0o2750


@pytest.mark.usefixtures("rootless")
def test_write_raises_when_the_record_has_nowhere_to_live():
    with pytest.raises(StateUnavailableError):
        control_context.write_record(control_context.ControlContext(target="live", generation=0))


@pytest.mark.usefixtures("data_root")
def test_refusal_terminus_moves_nothing_else():
    applied = control_context.ControlContext(
        target="va",
        generation=7,
        writer=control_context.RecordWriter(kind="web_terminal", pid=os.getpid(), port=8080),
        last_switch=control_context.SwitchTerminus.applied(
            request_id="r-1", target="va", generation=7, requested_at=None, requested_by="pid:1"
        ),
    )
    control_context.write_record(applied)
    refusal = applied.answered(
        control_context.SwitchTerminus.refused(
            request_id="r-2",
            target="live",
            reason="target_unreachable",
            detail="live did not answer",
            requested_at=None,
            requested_by="pid:1",
        )
    )
    control_context.write_record(refusal)

    after = control_context.read_record()
    assert after.target == applied.target
    assert after.generation == applied.generation
    assert after.writer == applied.writer
    assert after.last_switch is not None
    assert after.last_switch.status == "refused"
    assert after.last_switch.reason == "target_unreachable"
    assert after.last_switch.generation is None


# --- reading: the record survives ------------------------------------------


def test_read_parses_a_hand_written_payload(data_root):
    _write_raw(data_root, _payload())
    record = control_context.read_record()
    assert record.target == "va"
    assert record.generation == 7
    assert record.writer == control_context.RecordWriter(
        kind="web_terminal", pid=os.getpid(), port=8080
    )
    assert record.last_switch is not None
    assert record.last_switch.request_id == "r-1"


@pytest.mark.parametrize("target", CONTROL_TARGETS)
def test_read_accepts_every_control_target(data_root, target):
    _write_raw(data_root, _payload(target=target))
    assert control_context.read_record().target == target


def test_read_accepts_an_explicit_path(tmp_path):
    path = tmp_path / "elsewhere.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    assert control_context.read_record(path=path).generation == 7


# --- reading: degraded outcomes are all None -------------------------------


@pytest.mark.usefixtures("data_root")
def test_missing_record_reads_as_none():
    assert control_context.read_record() is None


@pytest.mark.usefixtures("rootless")
def test_unresolvable_root_reads_as_none():
    assert control_context.read_record() is None


def test_unparseable_record_reads_as_none(data_root):
    _write_raw(data_root, "{not json")
    assert control_context.read_record() is None


def test_non_object_record_reads_as_none(data_root):
    _write_raw(data_root, [1, 2, 3])
    assert control_context.read_record() is None


def test_non_utf8_record_reads_as_none(data_root):
    path = control_context._record_path_under(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xff\xfe{}")
    assert control_context.read_record() is None


def test_unreadable_record_reads_as_none(data_root):
    # A directory where the file belongs: the read fails with OSError, which
    # must arrive as "no record" like every other degraded outcome.
    control_context._record_path_under(data_root).mkdir(parents=True)
    assert control_context.read_record() is None


@pytest.mark.parametrize("schema", [0, 2, "1", None, True])
def test_wrong_schema_reads_as_none(data_root, schema):
    _write_raw(data_root, _payload(schema=schema))
    assert control_context.read_record() is None


@pytest.mark.parametrize("field", ["schema", "target", "generation"])
def test_an_absent_required_field_reads_as_none(data_root, field):
    payload = _payload()
    del payload[field]
    _write_raw(data_root, payload)
    assert control_context.read_record() is None


@pytest.mark.parametrize("target", ["", "banana", None, 7, ["va"]])
def test_unknown_target_reads_as_none(data_root, target):
    _write_raw(data_root, _payload(target=target))
    assert control_context.read_record() is None


@pytest.mark.parametrize("generation", [-1, "7", 7.0, None, True])
def test_unusable_generation_reads_as_none(data_root, generation):
    _write_raw(data_root, _payload(generation=generation))
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
    assert record.writer is None
    assert record.target == "va"


def test_absent_owner_reads_as_ownerless(data_root):
    payload = _payload()
    del payload["owner"]
    _write_raw(data_root, payload)
    assert control_context.read_record().writer is None


@pytest.mark.parametrize("port", [None, "8080", 0, True, -1])
def test_unusable_owner_port_reads_as_no_port(data_root, port):
    _write_raw(data_root, _payload(owner={"kind": "controls_server", "pid": 4242, "port": port}))
    writer = control_context.read_record().writer
    assert writer == control_context.RecordWriter(kind="controls_server", pid=4242, port=None)


@pytest.mark.parametrize("stored", [{"live": "sandbox"}, "sandbox", "writes", None, ["live"], 7])
def test_the_stored_posture_is_not_read(data_root, stored):
    """The mirror is the hooks' to read: whatever it holds, the record is the same."""
    _write_raw(data_root, _payload(posture=stored))
    record = control_context.read_record()
    assert record is not None
    assert not hasattr(record, "posture")
    assert record == control_context.parse_record(_payload())


def test_absent_posture_reads_the_same_record(data_root):
    payload = _payload()
    del payload["posture"]
    _write_raw(data_root, payload)
    assert control_context.read_record() == control_context.parse_record(_payload())


@pytest.mark.parametrize(
    "last_switch",
    [
        None,
        "applied",
        7,
        ["applied"],
        {"status": "applied"},
        {"request_id": 7, "status": "applied"},
        {"request_id": "r-1"},
        {"request_id": "r-1", "status": "unchanged"},
        {"request_id": "r-1", "status": None},
    ],
)
def test_unusable_last_switch_reads_as_none_and_keeps_the_record(data_root, last_switch):
    _write_raw(data_root, _payload(last_switch=last_switch))
    record = control_context.read_record()
    assert record is not None
    assert record.last_switch is None
    assert record.generation == 7


def test_a_well_formed_last_switch_reads_as_written(data_root):
    terminus = {
        "request_id": "r-9",
        "target": "live",
        "requested_at": "2026-01-01T00:00:00.250000+00:00",
        "requested_by": "sess-1",
        "status": "refused",
        "reason": "busy",
        "detail": "sandbox 3f2a is writing",
        "generation": None,
    }
    _write_raw(data_root, _payload(last_switch=terminus))
    assert control_context.read_record().last_switch.model_dump(mode="json") == terminus


@pytest.mark.parametrize(
    ("key", "bad", "read"),
    [
        ("target", 7, ""),
        ("requested_at", 1234.5, None),
        ("requested_at", "2026-01-01T00:00:00", None),
        ("requested_at", "not a stamp", None),
        ("requested_by", 99, ""),
        ("reason", 3, None),
        ("detail", None, ""),
        ("generation", -1, None),
        ("generation", True, None),
        ("generation", "7", None),
    ],
)
def test_a_bad_last_switch_field_degrades_alone(data_root, key, bad, read):
    """Rule 2: one bad field of the block reads as "not stated"; the block is kept."""
    block = {**_payload()["last_switch"], key: bad}
    _write_raw(data_root, _payload(last_switch=block))
    terminus = control_context.read_record().last_switch
    assert terminus is not None
    assert getattr(terminus, key) == read
    assert terminus.request_id == "r-1"
    assert terminus.status == "applied"


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
    testing.invalidate_cache()
    assert control_context.read_record().generation == 8


@pytest.mark.usefixtures("data_root")
def test_an_atomic_replace_is_seen_even_at_the_same_mtime_and_size():
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


@pytest.mark.usefixtures("data_root")
def test_a_removed_record_stops_being_served_from_the_cache():
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

    def fake_kill(pid, _sig):
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
    def denied(pid, _sig):
        raise PermissionError(pid)

    monkeypatch.setattr(os, "kill", denied)
    assert control_context.is_process_alive(4321) is True


def test_an_unexpected_os_error_counts_as_alive(monkeypatch):
    def odd(_pid, _sig):
        raise OSError("platform oddity")

    monkeypatch.setattr(os, "kill", odd)
    assert control_context.is_process_alive(4321) is True


@pytest.mark.parametrize("value", [0, -1, True, False, "4321", 4321.0, None, [4321]])
def test_nothing_but_a_positive_int_names_a_process(monkeypatch, value):
    """``True`` is ``1`` to ``os.kill``, and PID 1 is always alive."""

    def explode(pid, _sig):  # pragma: no cover - must not be called
        raise AssertionError(f"os.kill called with {pid!r}")

    monkeypatch.setattr(os, "kill", explode)
    assert control_context.is_process_alive(value) is False


def _status(root, pid, tgid):
    """Lay down ``<root>/<pid>/status`` as the kernel words the two lines read."""
    directory = root / str(pid)
    directory.mkdir()
    (directory / "status").write_bytes(
        f"Name:\tosprey\nUmask:\t0022\nTgid:\t{tgid}\nNgid:\t0\nPid:\t{pid}\n".encode()
    )


def test_a_thread_of_a_process_is_not_a_process(proc_root, monkeypatch):
    """``os.kill`` answers for any thread id; the deployment's record named one."""
    monkeypatch.setattr(os, "kill", _kill_with_dead(set()))
    _status(proc_root, 424, tgid=193)

    assert control_context.is_process_alive(424) is False


def test_a_thread_group_leader_is_a_process(proc_root, monkeypatch):
    monkeypatch.setattr(os, "kill", _kill_with_dead(set()))
    _status(proc_root, 193, tgid=193)

    assert control_context.is_process_alive(193) is True


def test_a_thread_we_may_not_signal_is_not_a_process(proc_root, monkeypatch):
    def denied(pid, _sig):
        raise PermissionError(pid)

    monkeypatch.setattr(os, "kill", denied)
    _status(proc_root, 424, tgid=193)

    assert control_context.is_process_alive(424) is False


@pytest.mark.parametrize(
    "status",
    [b"", b"Name:\tosprey\n", b"Tgid:\n", b"Tgid:\tnot-a-number\n", b"\xff\xfe\n"],
    ids=["empty", "no-tgid-line", "no-value", "not-a-number", "not-text"],
)
def test_a_status_that_says_nothing_leaves_the_kernels_answer(proc_root, monkeypatch, status):
    monkeypatch.setattr(os, "kill", _kill_with_dead(set()))
    (proc_root / "424").mkdir()
    (proc_root / "424" / "status").write_bytes(status)

    assert control_context.is_process_alive(424) is True


def test_a_record_naming_a_thread_has_no_live_owner(proc_root, monkeypatch):
    """The nine terminals of the report: each followed a thread of itself."""
    monkeypatch.setattr(os, "kill", _kill_with_dead(set()))
    _status(proc_root, 424, tgid=193)
    owner = {"kind": "web_terminal", "pid": 424, "port": 19101}
    record = control_context.parse_record(json.dumps(_payload(owner=owner)))

    assert record is not None
    assert record.writer is not None and record.writer.pid == 424
    assert control_context.live_writer(record) is None


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="needs a Linux /proc")
def test_a_real_thread_id_passes_the_kernel_check_and_is_no_process(monkeypatch):
    monkeypatch.setattr(_control_state, "_PROC_ROOT", "/proc")
    named = threading.Event()
    release = threading.Event()
    tids = []

    def run():
        tids.append(threading.get_native_id())
        named.set()
        release.wait(30)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert named.wait(30)
        os.kill(tids[0], 0)  # the kernel's answer, which is what misled the owner check
        assert tids[0] != os.getpid()
        assert control_context.is_process_alive(tids[0]) is False
        assert control_context.is_process_alive(os.getpid()) is True
    finally:
        release.set()
        thread.join(30)


# --- the record's owner ----------------------------------------------------


def _kill_denied_for(foreign_pids):
    """An ``os.kill`` under which *foreign_pids* run as another user."""

    def fake_kill(pid, _sig):
        if pid in foreign_pids:
            raise PermissionError(pid)
        return None

    return fake_kill


def _owned_record(pid):
    return control_context.ControlContext(
        target="va",
        generation=3,
        writer=control_context.RecordWriter(kind="web_terminal", pid=pid, port=8090),
    )


def test_a_running_owner_of_this_user_is_the_live_owner(monkeypatch):
    monkeypatch.setattr(os, "kill", _kill_with_dead(set()))
    record = _owned_record(4321)
    assert control_context.live_writer(record) == record.writer


def test_an_exited_owner_is_no_live_owner(monkeypatch):
    monkeypatch.setattr(os, "kill", _kill_with_dead({4321}))
    assert control_context.live_writer(_owned_record(4321)) is None


def test_another_users_process_under_the_owners_pid_is_no_live_owner(monkeypatch):
    """The upgrade that put an init process in front of the terminal.

    The record on the bind mount still names PID 1, which the previous
    container's terminal was. In the new container PID 1 is root's init and
    the terminal runs unprivileged beside it: alive, not ours to signal, and
    not the owner of anything in this directory.
    """
    monkeypatch.setattr(os, "kill", _kill_denied_for({1}))
    assert control_context.is_process_alive(1) is True
    assert control_context.live_writer(_owned_record(1)) is None


def test_no_record_and_no_owner_have_no_live_owner():
    assert control_context.live_writer(None) is None
    ownerless = control_context.ControlContext(target="va", generation=0)
    assert control_context.live_writer(ownerless) is None


def test_only_a_refused_signal_says_another_users_process(monkeypatch):
    monkeypatch.setattr(os, "kill", _kill_denied_for({1}))
    assert control_context._is_foreign_process(1) is True
    assert control_context._is_foreign_process(4321) is False

    monkeypatch.setattr(os, "kill", _kill_with_dead({4321}))
    assert control_context._is_foreign_process(4321) is False

    def odd(_pid, _sig):
        raise OSError("platform oddity")

    monkeypatch.setattr(os, "kill", odd)
    assert control_context._is_foreign_process(4321) is False


@pytest.mark.parametrize("value", [0, -1, True, False, "4321", 4321.0, None, [4321]])
def test_nothing_but_a_positive_int_is_another_users_process(monkeypatch, value):
    def explode(pid, _sig):  # pragma: no cover - must not be called
        raise AssertionError(f"os.kill called with {pid!r}")

    monkeypatch.setattr(os, "kill", explode)
    assert control_context._is_foreign_process(value) is False


# --- the claim rule --------------------------------------------------------


def test_claimed_unswitched_record_follows_baseline(caplog):
    """A record nobody has switched follows the deployment's baseline."""
    owner = control_context.RecordWriter(kind="web_terminal", pid=4242, port=8090)
    record = control_context.ControlContext(target="live", generation=0, writer=owner)

    with caplog.at_level("WARNING", logger="osprey_connectors.control_context"):
        result = control_context.claimed(record, "va")

    assert result.target == "va"
    assert result.generation == 0
    assert result.writer == owner
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "live" in warnings[0] and "va" in warnings[0]


@pytest.mark.parametrize(
    ("target", "generation"),
    [
        # A switch mints generation + 1, so a record past generation 0 was chosen.
        pytest.param("live", 3, id="switched-record-is-kept"),
        # A record already on the baseline is returned as it is, with nothing logged.
        pytest.param("va", 0, id="same-target"),
    ],
)
def test_claimed_returns_the_record_itself_without_a_warning(caplog, target, generation):
    record = control_context.ControlContext(target=target, generation=generation)

    with caplog.at_level("WARNING", logger="osprey_connectors.control_context"):
        result = control_context.claimed(record, "va")

    assert result is record
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_claimed_keeps_the_switch_and_the_last_terminus():
    """Following the baseline moves the target and nothing a person or a request set."""
    refused = control_context.SwitchTerminus.refused(
        request_id="r-1",
        target="standin",
        reason="target_unconfigured",
        detail="",
        requested_at=None,
        requested_by="pid:1",
    )
    record = control_context.ControlContext(
        target="live", generation=0, last_switch=refused, switch={"va": "asked"}
    )

    result = control_context.claimed(record, "va")

    assert result.target == "va"
    assert result.switch == {"va": "asked"}
    assert result.last_switch == refused


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
    path = control_context._report_path_under(root, pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    path.write_text(text, encoding="utf-8")
    return path


def test_reports_live_beside_the_record(data_root):
    """Every family of file in this directory moves together, identity hop included."""
    expected = write_switch.state_dir() / "server_4321.json"
    assert control_context._REPORT_FILE_PREFIX == "server_"
    assert control_context.REPORT_FILE_GLOB == "server_*.json"
    assert control_context.report_path(4321) == expected
    assert control_context._report_path_under(data_root, 4321) == expected
    assert expected.parent == control_context.record_path().parent


@pytest.mark.usefixtures("rootless")
def test_report_path_is_none_without_a_root():
    assert control_context.report_path(4321) is None
    assert control_context.report_paths() == []


def test_a_report_round_trips_through_its_payload():
    payload = _report_payload(4321)
    report = control_context._parse_report(payload)
    assert report is not None
    assert report.model_dump(mode="json") == payload
    assert control_context._parse_report(report.model_dump(mode="json")) == report


@pytest.mark.parametrize(
    "stamp", ["2026-01-01T01:00:00Z", "2026-01-01T01:00:00+00:00", "2025-12-31T18:00:00-07:00"]
)
def test_a_report_stamp_reads_as_its_instant_and_is_written_in_utc(stamp):
    report = control_context._parse_report(_report_payload(4321, updated_at=stamp))
    assert report is not None
    assert report.updated_at == datetime(2026, 1, 1, 1, tzinfo=UTC)
    assert report.model_dump(mode="json")["updated_at"] == "2026-01-01T01:00:00+00:00"


def _answered(requested_at):
    return control_context.ControlContext(target="va", generation=0).answered(
        control_context.SwitchTerminus.applied(
            request_id="r-1",
            target="live",
            generation=1,
            requested_at=requested_at,
            requested_by="pid:1",
        )
    )


def _stored_terminus(record):
    return record.model_dump(mode="json")["last_switch"]


def test_a_switch_answer_stores_its_request_stamp_in_utc():
    stamp = datetime(2026, 1, 1, 9, 0, 0, 250_000, tzinfo=timezone(timedelta(hours=9)))
    assert _stored_terminus(_answered(stamp))["requested_at"] == "2026-01-01T00:00:00.250000+00:00"
    assert _stored_terminus(_answered(None))["requested_at"] is None


@pytest.mark.parametrize(
    "build",
    [
        control_context.SwitchTerminus.applied,
        control_context.SwitchTerminus.unchanged,
        lambda **terms: control_context.SwitchTerminus.refused(
            reason="r", detail="d", **{k: v for k, v in terms.items() if k != "generation"}
        ),
    ],
)
def test_a_switch_answer_refuses_a_naive_request_stamp(build):
    with pytest.raises(NaiveDatetimeError):
        build(
            request_id="r-1",
            target="live",
            generation=1,
            requested_at=datetime(2026, 1, 1),
            requested_by="pid:1",
        )


def test_an_applied_answer_moves_the_target_and_mints_its_generation():
    record = control_context.ControlContext(target="va", generation=2, switch={"va": "asked"})
    terminus = control_context.SwitchTerminus.applied(
        request_id="r-1", target="live", generation=3, requested_at=None, requested_by="pid:1"
    )
    answered = record.answered(terminus)
    assert (answered.target, answered.generation, answered.last_switch) == ("live", 3, terminus)
    assert answered.switch == record.switch
    assert terminus.detail == "The control target is now 'live' (generation 3)."


def test_an_unchanged_answer_is_an_applied_terminus_that_mints_nothing():
    record = control_context.ControlContext(target="va", generation=2)
    terminus = control_context.SwitchTerminus.unchanged(
        request_id="r-1", target="va", generation=2, requested_at=None, requested_by="pid:1"
    )
    answered = record.answered(terminus)
    assert (answered.target, answered.generation) == ("va", 2)
    assert terminus.status == "applied"
    assert terminus.reason is None
    assert terminus.detail == "The control target is already 'va' (generation 2)."


def test_a_refused_answer_moves_nothing_but_the_block():
    record = control_context.ControlContext(target="va", generation=2)
    terminus = control_context.SwitchTerminus.refused(
        request_id="r-1",
        target="nowhere",
        reason="",
        detail="",
        requested_at=None,
        requested_by="pid:1",
    )
    answered = record.answered(terminus)
    assert (answered.target, answered.generation) == ("va", 2)
    assert (terminus.status, terminus.reason, terminus.generation) == ("refused", "", None)


@pytest.mark.parametrize("value", [None, 0, -1, True, "4321", 4321.0])
def test_a_report_without_a_server_pid_is_no_report(value):
    assert control_context._parse_report(_report_payload(value)) is None


def test_a_report_that_is_not_an_object_is_no_report():
    assert control_context._parse_report([1, 2, 3]) is None
    assert control_context._parse_report(None) is None


def test_a_server_that_has_not_answered_yet_reports_no_applied_target():
    report = control_context._parse_report(
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
        ("switch_progress", "applied", None),
        ("last_posture_realign", 3, None),
        ("targets", [], {}),
        ("updated_at", 17, None),
        ("updated_at", "soon", None),
        ("updated_at", "", None),
        ("updated_at", "2026-01-01T00:00:00", None),
    ],
)
def test_a_degraded_report_field_degrades_alone(field, value, expected):
    """The identity is ``server_pid``; every annotation on it stands or falls alone."""
    key = "last_switch" if field == "switch_progress" else field
    report = control_context._parse_report(_report_payload(4321, **{key: value}))
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
    testing.invalidate_cache()
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


def test_live_reports_drop_a_report_whose_server_pid_is_a_bool(data_root, monkeypatch):
    """``True`` would reach ``os.kill(1, 0)`` and read as alive forever."""
    _write_report(data_root, 4321, _report_payload(True))

    def explode(pid, _sig):  # pragma: no cover - must not be called
        raise AssertionError(f"os.kill called with {pid!r}")

    monkeypatch.setattr(os, "kill", explode)
    assert control_context.live_reports() == []


# --- display-only reports --------------------------------------------------


_STATEMENT = {
    "reachable": {"state": "yes", "since": None, "reason": None},
    "writes": {"allowed": False, "position": None, "reason": "switch_off", "kind": "simulator"},
}


def _display_payload(pid, **overrides):
    """The report a display process (the CANOPY panel backend) writes: binding and statement only."""
    payload = {
        "server_pid": pid,
        "session": None,
        "applied_target": "va",
        "applied_generation": 7,
        "children": [],
        "reachability": {},
        "last_switch": None,
        "last_posture_realign": None,
        "targets": {},
        "updated_at": "2026-01-01T00:00:00+00:00",
        "statement": _STATEMENT,
    }
    payload.update(overrides)
    return payload


def test_a_display_only_report_round_trips_eleven_fields():
    payload = _display_payload(4321)
    report = control_context._parse_report(payload)

    assert report.display_only is True
    assert report.statement is not None
    assert report.statement.model_dump(mode="json") == _STATEMENT
    assert report.model_dump(mode="json") == payload
    assert len(report.model_dump(mode="json")) == 11
    assert control_context._parse_report(report.model_dump(mode="json")) == report


def test_a_controls_server_report_stays_ten_fields_and_not_display_only():
    report = control_context._parse_report(_report_payload(4321))

    assert report.display_only is False
    assert report.statement is None
    assert "statement" not in report.model_dump(mode="json")
    assert len(report.model_dump(mode="json")) == 10


@pytest.mark.parametrize(
    "statement",
    [
        None,
        "on",
        [],
        {},
        {"reachable": {"state": "yes"}},
        {"writes": _STATEMENT["writes"]},
        {"reachable": "yes", "writes": _STATEMENT["writes"]},
        {"reachable": {"state": "yes"}, "writes": 1},
        {"reachable": {"state": "unknown"}, "writes": None},
        {"reachable": {"state": "no", "since": 7}, "writes": None},
        {"reachable": None, "writes": {"state": "on", "reason": None, "cause": None}},
        {"reachable": None, "writes": {"allowed": "yes", "position": None, "reason": None}},
        {"reachable": None, "writes": {"allowed": True, "position": "on", "kind": "real"}},
        {"reachable": None, "writes": {"allowed": True, "position": "asked", "kind": "live"}},
        {"reachable": None, "writes": {"allowed": False, "reason": 3, "kind": "real"}},
    ],
)
def test_a_statement_of_another_shape_states_nothing(statement):
    """A malformed statement never marks a report display-only (it reads as a controls server's)."""
    report = control_context._parse_report(_display_payload(4321, statement=statement))

    assert report is not None
    assert report.statement is None
    assert report.display_only is False
    assert "statement" not in report.model_dump(mode="json")
    assert control_context._parse_report(report.model_dump(mode="json")).display_only is False


def test_a_statement_not_yet_stated_on_either_axis_is_well_formed():
    report = control_context._parse_report(
        _display_payload(4321, statement={"reachable": None, "writes": None})
    )
    assert report.statement == control_context.MachineStatement(reachable=None, writes=None)
    assert report.display_only is True


@pytest.mark.parametrize("state", ["yes", "no"])
def test_reachable_is_stated_as_one_of_the_two_roots(state):
    statement = {
        "reachable": {"state": state, "since": "2026-01-01T00:00:00+00:00"},
        "writes": None,
    }
    report = control_context._parse_report(_display_payload(4321, statement=statement))
    assert report.statement.model_dump(mode="json")["reachable"] == {
        "state": state,
        "since": "2026-01-01T00:00:00+00:00",
        "reason": None,
    }


def test_a_reachable_since_that_states_no_instant_is_not_stated():
    """The statement stays a display process's; only the stamp degrades."""
    statement = {"reachable": {"state": "no", "since": "t", "reason": None}, "writes": None}
    report = control_context._parse_report(_display_payload(4321, statement=statement))
    assert report.display_only is True
    assert report.statement.reachable == control_context.Reachable(state="no")


def test_a_writes_statement_carries_the_verdicts_terms_and_nothing_else():
    verdict = write_switch.Refused(
        reason="session_held",
        target="va",
        kind="simulator",
        cause=write_switch._SessionHeld(sources=("watchdog:a",)),
    )
    writes = control_context.WritesStatement.of(verdict).model_dump(mode="json")

    assert writes == {
        "allowed": False,
        "position": None,
        "reason": "session_held",
        "kind": "simulator",
    }
    assert tuple(writes) == ("allowed", "position", "reason", "kind")
    report = control_context._parse_report(
        _display_payload(4321, statement={"reachable": None, "writes": writes})
    )
    assert report.statement.writes == control_context.WritesStatement.of(verdict)

    allowed = write_switch.Allowed(target="live", kind="real", position="asked")
    assert control_context.WritesStatement.of(allowed).model_dump(mode="json") == {
        "allowed": True,
        "position": "asked",
        "reason": None,
        "kind": "real",
    }


@pytest.mark.parametrize(
    "terms",
    [
        {"allowed": True, "position": "asked", "reason": "switch_off", "kind": "real"},
        {"allowed": False, "position": None, "reason": None, "kind": "real"},
        {"allowed": False, "position": "asked", "reason": "switch_off", "kind": "real"},
    ],
)
def test_a_writes_statement_that_disagrees_with_itself_cannot_be_built(terms):
    with pytest.raises(ValueError, match="writes statement"):
        control_context.WritesStatement.model_validate(terms)


@pytest.mark.parametrize(
    "writes",
    [
        {"allowed": True, "position": "asked", "reason": "switch_off", "kind": "real"},
        {"allowed": False, "position": None, "reason": None, "kind": "real"},
        {"allowed": False, "position": "allowed", "reason": "switch_off", "kind": "real"},
    ],
)
def test_a_stored_writes_axis_that_disagrees_with_itself_reads_as_not_stated(writes):
    """The report stays a display process's, so no fleet reader starts counting it."""
    report = control_context._parse_report(
        _display_payload(4321, statement={"reachable": None, "writes": writes})
    )
    assert report.display_only is True
    assert report.statement.writes is None


def test_a_stored_writes_reason_that_is_no_refusal_word_reads_as_not_stated():
    """An unknown word degrades the axis; the report still parses and stays display-only."""
    writes = {"allowed": False, "position": None, "reason": "no_such_word", "kind": "real"}
    report = control_context._parse_report(
        _display_payload(4321, statement={"reachable": None, "writes": writes})
    )
    assert report is not None
    assert report.display_only is True
    assert report.statement.writes is None


def test_a_writes_statement_holds_its_reason_as_the_refusal_member():
    writes = {"allowed": False, "position": None, "reason": "switch_off", "kind": "real"}
    statement = control_context.WritesStatement.model_validate(writes)
    assert statement.reason is VerdictReason.SWITCH_OFF
    with pytest.raises(ValueError, match="unknown refusal reason"):
        control_context.WritesStatement.model_validate({**writes, "reason": "no_such_word"})


def test_fleet_readers_leave_out_a_display_only_report(data_root):
    _write_report(data_root, 4321, _report_payload(4321))
    _write_report(data_root, 4322, _display_payload(4322))
    alive = {"is_alive": lambda pid: True}

    assert [r.server_pid for r in control_context.live_reports(**alive)] == [4321]
    reports = [control_context.read_report(path) for path in control_context.report_paths()]
    statements = [r for r in reports if r is not None and r.display_only]
    assert [r.server_pid for r in statements] == [4322]


def test_a_report_with_a_malformed_statement_is_read_as_a_controls_server(data_root):
    _write_report(data_root, 4322, _display_payload(4322, statement="malformed"))
    alive = {"is_alive": lambda pid: True}

    assert [r.server_pid for r in control_context.live_reports(**alive)] == [4322]
    [report] = [control_context.read_report(path) for path in control_context.report_paths()]
    assert report is not None and not report.display_only


@pytest.mark.parametrize("session", [None, "", "sess-1", "sess-other", "kernel:abc"])
@pytest.mark.parametrize(
    ("target", "generation"), [("va", 7), ("live", 6), (None, None), ("va", 8)]
)
def test_blocking_pids_passes_over_a_display_only_report(session, target, generation):
    """A display process states no session and no switch, so it never holds anybody up."""
    data = control_context._parse_report(
        _display_payload(4322, applied_target=target, applied_generation=generation)
    )
    record = _context()

    assert control_context.blocking_pids(record, [data], session) == ()
    assert control_context.converged(record, [data], session) is True


# --- sweeping --------------------------------------------------------------


def _sweep(directory, **kwargs):
    return control_context.sweep_dead(
        directory,
        prefix=control_context._REPORT_FILE_PREFIX,
        suffix=control_context._REPORT_FILE_SUFFIX,
        **kwargs,
    )


def test_a_dead_owners_file_is_swept(data_root, monkeypatch):
    dead = _write_report(data_root, 4321, _report_payload(4321))
    monkeypatch.setattr(os, "kill", _kill_with_dead({4321}))

    assert _sweep(write_switch.state_dir()) == [dead]
    assert not dead.exists()


def test_a_live_owners_file_is_left_alone(data_root, monkeypatch):
    alive = _write_report(data_root, 4321, _report_payload(4321))
    monkeypatch.setattr(os, "kill", _kill_with_dead(set()))

    assert _sweep(write_switch.state_dir()) == []
    assert alive.exists()


def test_the_kept_file_is_never_probed(data_root, monkeypatch):
    own = _write_report(data_root, 1234, _report_payload(1234))
    monkeypatch.setattr(os, "kill", _kill_with_dead({1234}))

    assert _sweep(write_switch.state_dir(), keep=own.name) == []
    assert own.exists()


@pytest.mark.usefixtures("data_root")
def test_a_filename_that_encodes_no_pid_is_swept():
    directory = write_switch.state_dir()
    directory.mkdir(parents=True)
    junk = directory / "server_notapid.json"
    junk.write_text("{}", encoding="utf-8")

    assert _sweep(directory) == [junk]
    assert not junk.exists()


@pytest.mark.usefixtures("data_root")
def test_a_missing_directory_sweeps_to_empty():
    assert not write_switch.state_dir().exists()
    assert _sweep(write_switch.state_dir()) == []


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
    """One report as a fleet reader parses it off its file; the defaults are a server applied at ``va``/7."""
    fields = {
        "server_pid": pid,
        "session": "sess-1",
        "applied_target": "va",
        "applied_generation": 7,
        "last_switch": {"generation": 7, "status": "applied", "at": _stamp(-60)},
    }
    fields.update(overrides)
    report = control_context._parse_report(fields)
    assert report is not None
    return report


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


def _launching(pid, *, generation=7, expires_in=300.0, session="sess-1", **overrides):
    """A server acquiring its FIRST connector: nothing bound, ``applying`` at *generation*.

    The three axes that could state a connector all state none, which is what
    a server whose report has never carried a binding looks like for the whole
    of its first launch.
    """
    block = {"generation": generation, "status": "applying", "at": _stamp(-1)}
    if expires_in is not None:
        block["expires_at"] = _stamp(expires_in)
    fields = {
        "session": session,
        "applied_target": None,
        "applied_generation": None,
        "children": (),
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
    far_future = datetime.now(UTC) + timedelta(days=1)
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


def test_a_first_launch_holds_only_its_own_session():
    """A server holding no connector is not between two targets."""
    record = _context()
    reports = [_launching(4321, session="sess-1")]

    assert control_context.blocking_pids(record, reports, "sess-1") == (4321,)
    assert control_context.converged(record, reports, "sess-1") is False
    for session in (None, "", "sess-other", "kernel:abcd1234"):
        assert control_context.blocking_pids(record, reports, session) == ()
        assert control_context.converged(record, reports, session) is True


@pytest.mark.parametrize(
    "stated",
    [{"applied_target": "live"}, {"applied_generation": 6}, {"children": (5001,)}],
)
def test_a_report_stating_a_connector_on_any_axis_holds_every_session(stated):
    record = _context()
    reports = [_launching(4321, **stated)]

    assert control_context.converged(record, reports, "sess-other") is False
    assert control_context.blocking_pids(record, reports, "sess-other") == (4321,)


@pytest.mark.parametrize("expires_in", [-1.0, None])
def test_a_first_launch_past_its_bound_still_holds_only_its_own_session(expires_in):
    """The bound decides nothing for a server that holds no connector."""
    record = _context()
    reports = [_launching(4321, expires_in=expires_in)]

    assert control_context.blocking_pids(record, reports, "sess-1") == (4321,)
    assert control_context.blocking_pids(record, reports, "sess-other") == ()


def test_a_first_launch_and_a_swap_in_flight_are_both_named():
    record = _context()
    reports = [_launching(4320, session="sess-1"), _applying(4321, session="sess-other")]

    assert control_context.blocking_pids(record, reports, "sess-1") == (4320, 4321)
    assert control_context.blocking_pids(record, reports, "sess-other") == (4321,)
    assert control_context.blocking_pids(record, reports, None) == (4321,)


def test_an_applying_report_is_unconverged_up_to_its_bound_inclusive():
    record = _context()
    expires_at = datetime.now(UTC) + timedelta(seconds=30)
    block = {"generation": 7, "status": "applying", "expires_at": expires_at.isoformat()}
    reports = [_applying(4321, expires_in=None, last_switch=block)]

    assert control_context.converged(record, reports, None, now=expires_at) is False
    just_past = expires_at + timedelta(microseconds=1)
    assert control_context.converged(record, reports, None, now=just_past) is True


def test_the_bound_is_compared_as_an_instant_whatever_its_zone():
    """A bound stated in another zone is the same moment; ``now`` in another zone too."""
    record = _context()
    expires_at = datetime.now(UTC) + timedelta(seconds=30)
    elsewhere = timezone(timedelta(hours=-7))
    block = {
        "generation": 7,
        "status": "applying",
        "expires_at": expires_at.astimezone(elsewhere).isoformat(),
    }
    reports = [_applying(4321, expires_in=None, last_switch=block)]

    at_bound = expires_at.astimezone(timezone(timedelta(hours=9)))
    assert control_context.converged(record, reports, None, now=at_bound) is False
    just_past = at_bound + timedelta(microseconds=1)
    assert control_context.converged(record, reports, None, now=just_past) is True


def test_a_naive_now_is_refused():
    record = _context()
    reports = [_applying(4321)]

    with pytest.raises(NaiveDatetimeError):
        control_context.blocking_pids(record, reports, None, now=datetime(2026, 1, 1))
    with pytest.raises(NaiveDatetimeError):
        control_context.converged(record, reports, None, now=datetime(2026, 1, 1))


@pytest.mark.parametrize("offset_s", [300.0, -300.0], ids=["ahead", "past"])
def test_a_naive_bound_is_no_bound(offset_s):
    """A stamp with no offset states no moment, so it keeps blocking, even one long past.

    Read as UTC before instants were aware datetimes, a naive bound in the past
    let the fleet converge; it now reads as no bound at all.
    """
    record = _context()
    block = {"generation": 7, "status": "applying", "expires_at": _stamp(offset_s, tz=None)}
    reports = [_applying(4321, expires_in=None, last_switch=block)]

    for session in (None, "sess-other"):
        assert control_context.converged(record, reports, session) is False
        assert control_context.blocking_pids(record, reports, session) == (4321,)


@pytest.mark.parametrize("block", [None, {"status": "applying"}, {"generation": 7}])
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


@pytest.mark.parametrize(
    ("report_session", "session"),
    [
        # A bare ``claude`` and the owner's fleet check are blocked only by applying.
        pytest.param(None, None, id="session-less-none"),
        pytest.param(None, "", id="session-less-blank"),
        pytest.param("sess-1", "kernel:abcd1234", id="kernel-session"),
    ],
)
def test_a_reader_with_no_session_to_own_a_report_stays_converged(report_session, session):
    record = _context()
    reports = [_report(4321, session=report_session, applied_generation=6)]

    assert control_context.converged(record, reports, session) is True


def test_every_unconverged_reason_is_named_once_and_in_pid_order():
    record = _context()
    reports = [
        _applying(4322, session="sess-other"),
        _report(4321, session="sess-1", applied_generation=6),
        _applying(4320, session="sess-1"),
    ]

    assert control_context.blocking_pids(record, reports, "sess-1") == (4320, 4321, 4322)
    assert control_context.converged(record, reports, "sess-1") is False


#: The safety table of the typed switch block (rule 5): each row is a block a
#: fleet reader parses off a live server's report with one field bad, and
#: whether it still holds every session. A bad field degrades alone, so an
#: ``applying`` block keeps blocking whatever else is wrong with it, an
#: unreadable bound is no bound, and a generation or status that does not read
#: names no swap the fleet coordinates on — as before the block was typed.
_SAFETY_TABLE = [
    pytest.param({"detail": 7}, True, id="applying-bad-detail-blocks"),
    pytest.param({"detail": ["x"]}, True, id="applying-list-detail-blocks"),
    pytest.param({"reason": 3}, True, id="applying-bad-reason-blocks"),
    pytest.param({"reason": {"why": "x"}}, True, id="applying-mapping-reason-blocks"),
    pytest.param({"at": "yesterday"}, True, id="applying-bad-at-blocks"),
    pytest.param({"at": 1234.5}, True, id="applying-numeric-at-blocks"),
    pytest.param({"at": "2026-01-01T00:00:00"}, True, id="applying-naive-at-blocks"),
    pytest.param({"expires_at": "soon"}, True, id="unreadable-bound-blocks"),
    pytest.param({"expires_at": 1234}, True, id="numeric-bound-blocks"),
    pytest.param({"expires_at": None}, True, id="absent-bound-blocks"),
    pytest.param({"status": "Applying"}, False, id="unknown-status-does-not-block"),
    pytest.param({"status": None}, False, id="no-status-does-not-block"),
    pytest.param({"status": 7}, False, id="non-word-status-does-not-block"),
    pytest.param({"generation": "7"}, False, id="text-generation-does-not-block"),
    pytest.param({"generation": True}, False, id="boolean-generation-does-not-block"),
    pytest.param({"generation": -7}, False, id="negative-generation-does-not-block"),
    pytest.param({"generation": None}, False, id="no-generation-does-not-block"),
]


@pytest.mark.parametrize(("bad", "blocks"), _SAFETY_TABLE)
def test_the_switch_block_safety_table(data_root, bad, blocks):
    """Read off the file a live server wrote, through the readers the gates use."""
    block = {
        "generation": 7,
        "status": "applying",
        "reason": None,
        "detail": None,
        "at": _stamp(-1),
        "expires_at": _stamp(300.0),
        **bad,
    }
    payload = {
        "server_pid": 4321,
        "session": "sess-1",
        "applied_target": "live",
        "applied_generation": 6,
        "children": [5001],
        "last_switch": block,
    }
    _write_report(data_root, 4321, payload)
    reports = control_context.live_reports(is_alive=lambda pid: True)
    [report] = reports
    assert report.switch_progress is not None, "the block is never dropped for one bad field"
    record = _context()
    for session in (None, "sess-other", "kernel:abc"):
        assert control_context.converged(record, reports, session) is not blocks
        assert control_context.blocking_pids(record, reports, session) == (
            (4321,) if blocks else ()
        )


def test_converged_reads_a_one_shot_iterable_of_reports_once():
    record = _context()
    reports = iter([_applying(4321)])

    assert control_context.blocking_pids(record, reports, "sess-1") == (4321,)


# --- the shared test fixtures ----------------------------------------------
#
# ``osprey_connectors_tests._harness._control_context_fixtures`` is what every
# suite downstream of the control target writes its record and its reports with. It emits the payloads
# the production readers parse, so these tests are the one place that proves
# the two agree — a field that moves in the schema fails here first, and in
# fifteen unrelated suites a second later.


def test_the_record_fixture_round_trips_through_read_record(data_root):
    """What the fixture writes is what ``read_record`` gives back."""
    from osprey_connectors_tests._harness import write_control_context

    path = write_control_context(
        data_root,
        target="va",
        generation=7,
        last_switch={"request_id": "req-1", "status": "applied"},
    )

    assert path == control_context.record_path()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["posture"] == control_context._posture_mirror({})
    assert stored["last_switch"] == {"request_id": "req-1", "status": "applied"}
    record = control_context.read_record()
    assert record is not None
    assert record.target == "va"
    assert record.generation == 7
    assert record.last_switch is not None
    assert (record.last_switch.request_id, record.last_switch.status) == ("req-1", "applied")


def test_the_record_fixture_defaults_to_an_owner_that_is_alive(data_root):
    """Ownership is what makes a record actionable, so the default is live.

    A fixture that defaulted to no owner — or to a PID that has exited — would
    silently put every suite on the claim path, where a reader is entitled to
    take the record over instead of following it.
    """
    from osprey_connectors_tests._harness import write_control_context

    write_control_context(data_root)

    record = control_context.read_record()
    assert record is not None
    assert record.writer is not None
    assert record.writer.kind == "web_terminal"
    assert record.writer.pid == os.getpid()
    assert control_context.is_process_alive(record.writer.pid) is True


def test_the_record_fixture_writes_an_ownerless_record_on_request(data_root):
    from osprey_connectors_tests._harness import write_control_context

    write_control_context(data_root, owned_by=None)

    record = control_context.read_record()
    assert record is not None
    assert record.writer is None


def test_the_record_fixture_owns_by_a_named_process_on_request(data_root):
    from osprey_connectors_tests._harness import owner, write_control_context

    write_control_context(
        data_root,
        owned_by=owner("controls_server", pid=4321),
    )

    record = control_context.read_record()
    assert record is not None
    assert record.writer == control_context.RecordWriter(
        kind="controls_server", pid=4321, port=None
    )


def test_the_report_fixture_round_trips_through_read_report(data_root):
    """Same contract for the ten-field report the readers walk per server."""
    from osprey_connectors_tests._harness import write_server_report

    path = write_server_report(
        data_root,
        4321,
        session="sess-1",
        applied_target="va",
        applied_generation=7,
        children=[9001, 9002],
        reachability={"va": {"reachable": True}},
        last_switch={"generation": 7, "status": "applied"},
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
        last_switch={"generation": 7, "status": "applied"},
        last_posture_realign={"at": "2026-01-01T00:00:00Z"},
        targets={"va": {"selected_role": "primary"}},
        updated_at="2026-01-01T00:00:01Z",
    )


def test_the_report_fixture_defaults_to_a_server_that_has_answered_nothing(data_root):
    """Null-bound, not baseline: a fresh server has not got anywhere yet."""
    from osprey_connectors_tests._harness import write_server_report

    write_server_report(data_root, 4321)

    report = control_context.read_report(control_context.report_path(4321))
    assert report is not None
    assert report.applied_target is None
    assert report.applied_generation is None
    assert report.children == ()
    assert report.reachability == {}
    assert report.switch_progress is None


def test_the_record_fixture_drops_the_read_cache_after_every_write(data_root, monkeypatch):
    """The reader caches on ``(mtime_ns, size, ino)``; the fixture drops it.

    Two writes of the same size inside one tick are exactly the case that key
    cannot distinguish, and a suite that flips a posture back and forth does it
    on every other line. Today the atomic replace also changes the inode, so a
    re-read alone cannot show the drop: the spy pins the fixture's own contract.
    """
    from osprey_connectors_tests._harness import write_control_context

    drops = []
    real_invalidate = testing.invalidate_cache

    def spy():
        drops.append(True)
        real_invalidate()

    monkeypatch.setattr(testing, "invalidate_cache", spy)

    write_control_context(data_root, target="va", generation=7)
    assert drops, "write_control_context did not drop the read cache"
    assert control_context.read_record().generation == 7

    drops.clear()
    write_control_context(data_root, target="va", generation=8)
    assert drops, "write_control_context did not drop the read cache"
    assert control_context.read_record().generation == 8


def test_the_record_fixture_writes_a_non_mapping_posture_verbatim(data_root):
    """The hook's posture parser's degradation cases have to be statable.

    A bare ``"sandbox"`` is the shape the session-wide posture wrote before
    targets existed, and the hook's parser still has a rule for it. A fixture
    that could only write mappings would leave every such suite hand-rolling a
    payload beside this writer.
    """
    from osprey_connectors_tests._harness import write_control_context

    path = write_control_context(data_root, posture="sandbox")

    assert json.loads(path.read_text(encoding="utf-8"))["posture"] == "sandbox"
    assert control_context.read_record() is not None


def test_the_fixture_writes_a_degraded_record_through_the_raw_payload_hatch(data_root):
    """A record too broken to build from the dataclass still has to be writable."""
    from osprey_connectors_tests._harness import write_payload

    write_payload(control_context._record_path_under(data_root), {"target": "va"})

    assert control_context.read_record() is None


# --- the switch fields -------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"live": "on", "va": "on"}, {"live": "asked", "va": "asked"}),
        ({"live": "on", "va": "off"}, {"live": "asked"}),
        ({"live": True, "va": "ON", "standin": 1}, {}),
        ({"unknown_target": "on"}, {"unknown_target": "asked"}),
        ("on", {}),
        (["live"], {}),
        (None, {}),
    ],
)
def test_only_a_position_survives_in_the_switch(value, expected):
    """Rule 2: a degraded switch degrades to off, alone; the identity stands.

    The legacy ``"on"`` is a position: it reads as ``"asked"``.
    """
    record = control_context.parse_record(_payload(switch=value))
    assert record is not None
    assert record.switch == expected
    assert record.target == "va" and record.generation == 7


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), (1792772360631776000, 1792772360631776000), (True, None), (-1, None), ("5", None)],
)
def test_the_switch_epoch_is_a_non_negative_int_or_none(value, expected):
    record = control_context.parse_record(_payload(switch_epoch=value))
    assert record is not None
    assert record.switch_epoch == expected


def test_a_record_without_switch_fields_reads_switched_off_and_epoch_less():
    record = control_context.parse_record(_payload())
    assert record.switch == {}
    assert record.switch_epoch is None


def test_the_switch_fields_round_trip_through_the_payload():
    record = control_context.ControlContext(
        target="live", generation=3, switch={"live": "asked"}, switch_epoch=42
    )
    payload = record.model_dump(mode="json")
    assert payload["switch"] == {"live": "asked"} and payload["switch_epoch"] == 42
    assert control_context.parse_record(payload) == record
    assert control_context.parse_record(json.dumps(payload)) == record


# --- the status reader -------------------------------------------------------


def test_status_reads_ok_with_the_record(data_root):
    from osprey_connectors_tests._harness._control_context_fixtures import write_control_context

    write_control_context(data_root, target="standin", generation=2)
    read = control_context.read_record_status()
    record, status = read.value, read.status
    assert status == "ok"
    assert record is not None and record.target == "standin"
    assert record == control_context.read_record()


@pytest.mark.usefixtures("data_root")
def test_status_reads_missing_when_there_is_no_file():
    read = control_context.read_record_status()
    assert (read.status, read.value, read.path) == ("missing", None, control_context.record_path())
    assert control_context.read_record() is None


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",
        json.dumps(_payload(schema=2)),
        json.dumps(_payload(target="elsewhere")),
        json.dumps(_payload(generation=-1)),
        json.dumps([1, 2]),
    ],
    ids=["corrupt", "foreign_schema", "unknown_target", "bad_generation", "not_an_object"],
)
def test_status_reads_unreadable_for_a_file_that_states_nothing(data_root, raw):
    _write_raw(data_root, raw)
    assert control_context.read_record_status().status == "unreadable"
    assert control_context.read_record() is None


def test_status_reads_unreadable_for_a_record_that_is_not_utf8(data_root):
    _write_raw(data_root, json.dumps(_payload(detail="é")), encoding="utf-16")
    assert control_context.read_record_status().status == "unreadable"


def test_status_reads_unreadable_when_the_path_cannot_be_stat_ed(data_root):
    """A path component that is a file is no record — but not a missing one."""
    blocker = data_root / "blocker"
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("", encoding="utf-8")
    assert control_context.read_record_status(path=blocker / "record.json").status == "unreadable"


@pytest.mark.usefixtures("rootless")
def test_status_reads_unreadable_without_a_root():
    assert control_context.read_record_status().status == "unreadable"
    assert control_context.read_record() is None


@pytest.mark.usefixtures("data_root")
def test_status_honours_an_explicit_path(tmp_path):
    elsewhere = tmp_path / "elsewhere.json"
    assert control_context.read_record_status(path=elsewhere).status == "missing"
    control_context.write_record(
        control_context.ControlContext(target="va", generation=1), path=elsewhere
    )
    read = control_context.read_record_status(path=elsewhere)
    record, status = read.value, read.status
    assert status == "ok" and record.target == "va"
    assert control_context.read_record_status().status == "missing"


def test_status_follows_a_record_that_is_corrupted_after_it_was_read(data_root):
    from osprey_connectors_tests._harness._control_context_fixtures import write_control_context

    write_control_context(data_root)
    assert control_context.read_record_status().status == "ok"
    _write_raw(data_root, "{broken")
    assert control_context.read_record_status().status == "unreadable"


# --- a record this version writes, read by the readers that predate it ---------


def _load_target_state_hook(monkeypatch):
    """The shipped stdlib hook, by path; its ``sys.path`` insert is undone."""
    import importlib.util
    import sys

    monkeypatch.setattr(sys, "path", list(sys.path))
    hook = REPO_ROOT / "src/osprey/templates/claude_code/claude/hooks/osprey_target_state.py"
    spec = importlib.util.spec_from_file_location("_record_switch_target_state", hook)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_record_with_switch_fields_reads_unchanged_through_every_older_reader(
    data_root, monkeypatch
):
    from osprey_connectors_tests._harness._control_context_fixtures import write_control_context

    path = write_control_context(
        data_root,
        target="standin",
        generation=5,
        switch={"standin": "asked"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["switch"] == {"standin": "asked"}
    assert isinstance(payload["switch_epoch"], int)
    # The mirror: every machine not turned on reads as narrowed to an older reader.
    mirror = {"live": "sandbox", "va": "sandbox"}
    assert payload["posture"] == mirror

    record = control_context.read_record()
    assert (record.target, record.generation) == ("standin", 5)

    hook = _load_target_state_hook(monkeypatch)
    assert hook.parse_record(path.read_text(encoding="utf-8")) == {
        "target": "standin",
        "generation": 5,
        "posture": mirror,
    }


# --- the shared fixture's switch ------------------------------------------------


def test_the_record_fixture_writes_a_switch_on_request(data_root):
    from osprey_connectors_tests._harness._control_context_fixtures import write_control_context

    path = write_control_context(data_root, switch={"va": "asked"})
    first_epoch = control_context.read_record().switch_epoch
    assert control_context.read_record().switch == {"va": "asked"}
    assert isinstance(first_epoch, int) and first_epoch > 0

    # A rewrite without switch= keeps what is on disk, the production rule;
    # a turn-on never moves the epoch.
    write_control_context(data_root, generation=2)
    assert control_context.read_record().switch == {"va": "asked"}
    write_control_context(data_root, generation=3, switch={})
    assert control_context.read_record().switch == {}
    assert control_context.read_record().switch_epoch == first_epoch
    assert path == control_context.record_path()


def test_the_record_fixture_writes_a_degraded_switch_verbatim(data_root):
    from osprey_connectors_tests._harness._control_context_fixtures import write_control_context

    path = write_control_context(data_root, switch={"live": "yes"})
    assert json.loads(path.read_text(encoding="utf-8"))["switch"] == {"live": "yes"}
    assert control_context.read_record().switch == {}


# --- one writer ------------------------------------------------------------------

#: The source trees whose modules ship.
_SRC_ROOTS = (
    REPO_ROOT / "src",
    REPO_ROOT / "packages",
)

#: Calls that put bytes in a file. A scope that names the record and makes one
#: of these is a second writer of it.
_WRITE_CALLS = frozenset(
    {
        "write_text",
        "write_bytes",
        "write_json_atomic",
        "atomic_replace",
        "rename",
        "dump",
        "_write_record_unlocked",
    }
)


def _scopes(tree):
    """Every function body and the module body, each as its own node list."""
    import ast

    module_level = [
        n
        for n in tree.body
        if not isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    yield "<module>", module_level
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node.name, [node]


def _names_the_record(nodes) -> bool:
    import ast

    for node in nodes:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id == "RECORD_FILENAME":
                return True
            if isinstance(sub, ast.Attribute) and sub.attr == "RECORD_FILENAME":
                return True
            if isinstance(sub, ast.Constant) and sub.value == control_context.RECORD_FILENAME:
                return True
            if isinstance(sub, ast.Call):
                func = sub.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in {"record_path", "_record_path_under"}:
                    return True
    return False


def _writes(nodes) -> list[str]:
    import ast

    found = []
    for node in nodes:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            func = sub.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in _WRITE_CALLS:
                found.append(name)
            elif (
                name == "replace"
                and isinstance(func, ast.Attribute)
                and (
                    # ``os.replace(src, dst)`` or ``Path.replace(dst)``; not
                    # ``str.replace(old, new)`` nor ``dataclasses.replace``.
                    (isinstance(func.value, ast.Name) and func.value.id == "os")
                    or (len(sub.args) == 1 and not sub.keywords)
                )
            ):
                found.append(name)
            elif name == "open" and any(
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, str)
                and set(arg.value) & set("wax+")
                for arg in [*sub.args[1:2], *(k.value for k in sub.keywords if k.arg == "mode")]
            ):
                found.append("open")
    return found


def test_no_shipped_module_but_control_context_writes_the_record():
    """``control_context`` is the record's one writer (rule 3).

    Any other scope — a function, or a module body — that names the record (its
    filename, or the path functions) must not also write a file or reach the
    private writer, which skips the lock-held switch substitution. The record's
    readers pass; a second writer, however it spells the path, does not.
    """
    import ast

    own = Path(control_context.__file__).resolve()
    offenders = []
    for root in _SRC_ROOTS:
        for source in root.rglob("*.py"):
            if source.resolve() == own or "tests" in source.parts:
                continue
            text = source.read_text(encoding="utf-8")
            if "_write_record_unlocked" in text:
                offenders.append(f"{source}: calls the private writer")
            if not (
                "RECORD_FILENAME" in text
                or control_context.RECORD_FILENAME in text
                or "record_path" in text
            ):
                continue
            tree = ast.parse(text)
            for scope, nodes in _scopes(tree):
                if _names_the_record(nodes) and (writes := _writes(nodes)):
                    offenders.append(f"{source}:{scope} names the record and calls {writes}")
    assert offenders == []


def test_the_one_writer_guard_sees_a_writer():
    """The detector above is not vacuous: ``control_context``'s own writer trips it."""
    import ast

    tree = ast.parse(Path(control_context.__file__).read_text(encoding="utf-8"))
    flagged = {
        scope for scope, nodes in _scopes(tree) if _names_the_record(nodes) and _writes(nodes)
    }
    assert "write_record" in flagged
