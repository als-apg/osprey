"""Inbox: durable put before ack, removal after handling, attempts, quarantine, cap."""

from __future__ import annotations

import json
import os
import threading

import pytest

from osprey.bridges.slack import inbox as inbox_module
from osprey.bridges.slack.inbox import Inbox


def _payload(n: int) -> dict:
    return {"type": "event_callback", "event_id": f"Ev{n:04d}", "event": {"text": f"q{n}"}}


def _live_files(directory) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))


def _quarantined(directory) -> list[str]:
    q = directory / "quarantine"
    return sorted(p.name for p in q.iterdir()) if q.is_dir() else []


# --- put / pending / remove ------------------------------------------------------


def test_put_then_pending_returns_key_and_payload(tmp_path):
    box = Inbox(tmp_path)
    assert box.put("Ev0001", _payload(1)) is True
    assert box.pending() == [("Ev0001", _payload(1))]


def test_pending_is_oldest_first(tmp_path):
    box = Inbox(tmp_path)
    # Keys whose name order is the reverse of their arrival order.
    for key in ("EvZZZ", "EvMMM", "EvAAA"):
        assert box.put(key, {"k": key})
    assert [k for k, _ in box.pending()] == ["EvZZZ", "EvMMM", "EvAAA"]


def test_order_survives_a_restart(tmp_path):
    box = Inbox(tmp_path)
    for key in ("EvC", "EvB", "EvA"):
        box.put(key, {"k": key})
    reopened = Inbox(tmp_path)
    reopened.put("Ev0", {"k": "Ev0"})
    assert [k for k, _ in reopened.pending()] == ["EvC", "EvB", "EvA", "Ev0"]


def test_remove_drops_the_record(tmp_path):
    box = Inbox(tmp_path)
    box.put("Ev0001", _payload(1))
    box.put("Ev0002", _payload(2))
    box.remove("Ev0001")
    assert box.pending() == [("Ev0002", _payload(2))]


def test_remove_of_unknown_key_is_a_no_op(tmp_path):
    box = Inbox(tmp_path)
    box.remove("EvNOPE")
    assert box.pending() == []


def test_survives_a_simulated_crash_between_put_and_remove(tmp_path):
    box = Inbox(tmp_path)
    assert box.put("Ev0001", _payload(1))
    del box  # the process dies here: handle_event never returned, remove never ran
    restarted = Inbox(tmp_path)
    assert restarted.pending() == [("Ev0001", _payload(1))]
    restarted.remove("Ev0001")
    assert Inbox(tmp_path).pending() == []


def test_put_leaves_no_temp_files_and_is_private(tmp_path):
    box = Inbox(tmp_path)
    box.put("Ev0001", _payload(1))
    names = [p.name for p in tmp_path.iterdir() if p.is_file()]
    assert len(names) == 1 and not names[0].startswith(".")
    assert (tmp_path / names[0]).stat().st_mode & 0o777 == 0o600


def test_directory_is_created(tmp_path):
    target = tmp_path / "state" / "inbox"
    box = Inbox(target)
    assert box.put("Ev1", {"a": 1})
    assert target.is_dir()


def test_put_uses_atomic_replace_with_required_sync(tmp_path, monkeypatch):
    calls = []
    real = inbox_module.atomic_replace

    def spy(path, data, *, mode, sync="required"):
        calls.append((mode, sync))
        return real(path, data, mode=mode, sync=sync)

    monkeypatch.setattr(inbox_module, "atomic_replace", spy)
    Inbox(tmp_path).put("Ev1", {"a": 1})
    assert calls == [(0o600, "required")]


def test_redelivered_key_is_accepted_without_resetting_the_record(tmp_path):
    box = Inbox(tmp_path)
    box.put("Ev1", {"v": 1})
    box.note_failure("Ev1")
    box.put("Ev2", {"v": 2})
    assert box.put("Ev1", {"v": "redelivered"}) is True
    # Still one record for Ev1, its original place and payload, and its attempt kept.
    assert box.pending() == [("Ev1", {"v": 1}), ("Ev2", {"v": 2})]
    box.note_failure("Ev1")
    box.note_failure("Ev1")
    assert [k for k, _ in box.pending()] == ["Ev2"]


# --- keys --------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["../escape", "a/b", "Ev.1", "..", ".hidden", "Ev 1\x00"])
def test_unsafe_keys_stay_inside_the_directory(tmp_path, key):
    directory = tmp_path / "inbox"
    box = Inbox(directory)
    assert box.put(key, {"k": key})
    assert [k for k, _ in box.pending()] == [key]
    # Nothing escaped, nothing hidden (hidden names are reserved for temp files).
    assert sorted(p.name for p in tmp_path.iterdir()) == ["inbox"]
    assert len(_live_files(directory)) == 1
    box.remove(key)
    assert box.pending() == []


def test_distinct_keys_never_share_a_file(tmp_path):
    box = Inbox(tmp_path)
    box.put("a/b", {"n": 1})
    box.put("a_b", {"n": 2})
    assert box.pending() == [("a/b", {"n": 1}), ("a_b", {"n": 2})]


def test_empty_key_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        Inbox(tmp_path).put("", {})


# --- cap -----------------------------------------------------------------------------


def test_cap_is_honoured(tmp_path):
    box = Inbox(tmp_path, cap=3)
    assert all(box.put(f"Ev{n}", _payload(n)) for n in range(3))
    assert box.put("Ev3", _payload(3)) is False
    assert [k for k, _ in box.pending()] == ["Ev0", "Ev1", "Ev2"]
    box.remove("Ev0")
    assert box.put("Ev3", _payload(3)) is True


def test_default_cap_is_64(tmp_path):
    box = Inbox(tmp_path)
    assert all(box.put(f"Ev{n}", {}) for n in range(64))
    assert box.put("Ev64", {}) is False


def test_cap_counts_records_left_by_a_previous_process(tmp_path):
    first = Inbox(tmp_path, cap=2)
    first.put("Ev0", {})
    first.put("Ev1", {})
    assert Inbox(tmp_path, cap=2).put("Ev2", {}) is False


def test_redelivery_at_the_cap_is_still_accepted(tmp_path):
    box = Inbox(tmp_path, cap=1)
    box.put("Ev0", {})
    assert box.put("Ev0", {}) is True


def test_quarantined_records_do_not_count_toward_the_cap(tmp_path):
    box = Inbox(tmp_path, cap=1, max_attempts=1)
    box.put("Ev0", {})
    box.note_failure("Ev0")
    assert box.put("Ev1", {}) is True


# --- OS errors ---------------------------------------------------------------------


def test_put_returns_false_on_os_error(tmp_path, monkeypatch):
    box = Inbox(tmp_path)

    def boom(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(inbox_module, "atomic_replace", boom)
    assert box.put("Ev1", {"a": 1}) is False
    monkeypatch.undo()
    assert box.pending() == []


# --- attempts and quarantine ------------------------------------------------------


def test_note_failure_quarantines_at_max_attempts(tmp_path):
    box = Inbox(tmp_path, max_attempts=3)
    box.put("Ev1", _payload(1))
    box.note_failure("Ev1")
    box.note_failure("Ev1")
    assert box.pending() == [("Ev1", _payload(1))]
    box.note_failure("Ev1")
    assert box.pending() == []
    quarantined = _quarantined(tmp_path)
    assert len(quarantined) == 1
    record = json.loads((tmp_path / "quarantine" / quarantined[0]).read_text())
    assert record["payload"] == _payload(1)  # kept for an operator, never deleted


def test_attempts_survive_a_restart(tmp_path):
    box = Inbox(tmp_path, max_attempts=2)
    box.put("Ev1", {})
    box.note_failure("Ev1")
    restarted = Inbox(tmp_path, max_attempts=2)
    restarted.note_failure("Ev1")
    assert restarted.pending() == []
    assert len(_quarantined(tmp_path)) == 1


def test_note_failure_on_unknown_key_is_a_no_op(tmp_path):
    box = Inbox(tmp_path)
    box.note_failure("EvNOPE")
    assert box.pending() == [] and _quarantined(tmp_path) == []


def test_repeated_quarantine_of_one_key_keeps_both_copies(tmp_path):
    box = Inbox(tmp_path, max_attempts=1)
    for _ in range(2):
        box.put("Ev1", {})
        box.note_failure("Ev1")
    assert len(_quarantined(tmp_path)) == 2


@pytest.mark.parametrize(
    "content",
    [b"{not json", b"\xff\xfe\x00garbage", b"[1, 2, 3]", b'{"no": "payload"}', b""],
)
def test_corrupt_file_is_quarantined_and_skipped(tmp_path, content):
    box = Inbox(tmp_path)
    box.put("Ev1", _payload(1))
    (tmp_path / "Ev2.json").write_bytes(content)
    box.put("Ev3", _payload(3))
    assert box.pending() == [("Ev1", _payload(1)), ("Ev3", _payload(3))]
    assert len(_quarantined(tmp_path)) == 1
    assert (tmp_path / "quarantine" / _quarantined(tmp_path)[0]).read_bytes() == content
    # Quarantined once; the next scan does not see it again.
    assert box.pending() == [("Ev1", _payload(1)), ("Ev3", _payload(3))]
    assert len(_quarantined(tmp_path)) == 1


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads files regardless of mode")
def test_unreadable_file_is_quarantined(tmp_path):
    box = Inbox(tmp_path)
    box.put("Ev1", _payload(1))
    locked = tmp_path / "Ev2.json"
    locked.write_text(json.dumps({"payload": {}}))
    locked.chmod(0)
    try:
        assert box.pending() == [("Ev1", _payload(1))]
        assert len(_quarantined(tmp_path)) == 1
    finally:
        for p in (tmp_path / "quarantine").iterdir():
            p.chmod(0o600)


def test_pending_ignores_temp_files_and_subdirectories(tmp_path):
    box = Inbox(tmp_path)
    box.put("Ev1", {})
    (tmp_path / ".Ev9.json.abc123.tmp").write_text("{half")
    (tmp_path / "somedir").mkdir()
    assert box.pending() == [("Ev1", {})]
    assert _quarantined(tmp_path) == []


# --- threads -----------------------------------------------------------------------


def test_concurrent_puts_respect_the_cap(tmp_path):
    box = Inbox(tmp_path, cap=10)
    results: list[bool] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker(t: int) -> None:
        barrier.wait()
        for n in range(5):
            ok = box.put(f"Ev{t}-{n}", {"t": t, "n": n})
            with lock:
                results.append(ok)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert results.count(True) == 10
    assert len(box.pending()) == 10


def test_concurrent_failures_count_every_attempt(tmp_path):
    box = Inbox(tmp_path, max_attempts=4)
    box.put("Ev1", {})
    barrier = threading.Barrier(4)

    def worker() -> None:
        barrier.wait()
        box.note_failure("Ev1")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert box.pending() == []
    assert len(_quarantined(tmp_path)) == 1
