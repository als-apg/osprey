"""Unit tests for the persisted per-room poll offsets in
``osprey.bridges.nextcloud_talk.offsets``.

:class:`OffsetStore` is two guarantees on top of the core JSON store, and each
one maps to a concrete way the bridge could misbehave in a room:

  * an unknown room reads as ``None`` (first-ever start → the poller skips the
    room's backlog), and a known room resumes exactly where it stopped;
  * ``advance`` is monotonic, so a re-fetched batch after a crash — or any late
    call carrying an id already passed — cannot rewind the cursor and replay
    history;
  * a corrupt or garbage file loads as empty instead of crashing the bridge at
    startup, while an unusable path fails loudly (silently losing offsets would
    replay or skip history unnoticed).

Every test is hermetic: offsets live under ``tmp_path``, never the real ``/data``.
"""

from __future__ import annotations

import json
import threading

import pytest

from osprey.bridges.nextcloud_talk import OffsetStore


def _store(tmp_path, name: str = "offsets.json") -> OffsetStore:
    return OffsetStore(str(tmp_path / name))


def _on_disk(tmp_path, name: str = "offsets.json"):
    return json.loads((tmp_path / name).read_text(encoding="utf-8"))


# --- first start / resume ---------------------------------------------------
def test_unknown_room_reads_as_none(tmp_path):
    """No file at all: every room is a first-ever start."""
    store = _store(tmp_path)

    assert store.get("room-a") is None
    assert not (tmp_path / "offsets.json").exists(), "a pure read must not write"


def test_known_room_does_not_leak_its_offset_to_other_rooms(tmp_path):
    """Offsets are per-room: advancing one leaves the others at first-start."""
    store = _store(tmp_path)
    store.advance("room-a", 42)

    assert store.get("room-a") == 42
    assert store.get("room-b") is None


def test_offset_survives_a_restart(tmp_path):
    """A fresh instance over the same path resumes where the last one stopped —
    this is the whole point of persisting: messages posted while the bridge was
    down are fetched, not skipped."""
    first = _store(tmp_path)
    first.advance("room-a", 7)
    first.advance("room-b", 11)

    resumed = _store(tmp_path)

    assert resumed.get("room-a") == 7
    assert resumed.get("room-b") == 11


def test_first_advance_persists_the_file(tmp_path):
    store = _store(tmp_path)
    store.advance("room-a", 5)

    assert _on_disk(tmp_path) == {"room-a": 5}


def test_zero_is_a_real_offset_not_a_first_start(tmp_path):
    """``0`` is a stored value, distinct from ``None``. Conflating them would
    re-skip the room's history on every restart."""
    store = _store(tmp_path)
    store.advance("room-a", 0)

    assert store.get("room-a") == 0
    assert _store(tmp_path).get("room-a") == 0


# --- monotonicity ----------------------------------------------------------
def test_advance_moves_forward(tmp_path):
    store = _store(tmp_path)
    store.advance("room-a", 10)
    store.advance("room-a", 20)

    assert store.get("room-a") == 20
    assert _on_disk(tmp_path) == {"room-a": 20}


def test_lower_id_does_not_move_the_offset_back(tmp_path):
    """A re-fetched batch after a crash replays ids already passed; absorbing
    them silently is the contract — no exception, no regression, no write."""
    store = _store(tmp_path)
    store.advance("room-a", 100)
    before = (tmp_path / "offsets.json").read_bytes()

    store.advance("room-a", 99)
    store.advance("room-a", 1)

    assert store.get("room-a") == 100
    assert (tmp_path / "offsets.json").read_bytes() == before, "no-op must not rewrite"
    assert _store(tmp_path).get("room-a") == 100


def test_repeating_the_current_id_is_a_no_op(tmp_path):
    store = _store(tmp_path)
    store.advance("room-a", 100)
    before = (tmp_path / "offsets.json").read_bytes()

    store.advance("room-a", 100)

    assert store.get("room-a") == 100
    assert (tmp_path / "offsets.json").read_bytes() == before


def test_declined_advance_performs_no_write(tmp_path, monkeypatch):
    """A declined advance must not touch the file at all.

    Asserted by counting flushes rather than comparing bytes: rewriting the same
    offset produces an identical file, so only the flush count can tell a no-op
    apart from a redundant write. Every long-poll cycle after a crash re-offers
    ids already passed, and those must not each cost a rewrite of the volume.
    """
    store = _store(tmp_path)
    store.advance("room-a", 100)

    flushes = 0
    original = store._flush_locked

    def counting_flush() -> None:
        nonlocal flushes
        flushes += 1
        original()

    monkeypatch.setattr(store, "_flush_locked", counting_flush)

    store.advance("room-a", 100)  # equal
    store.advance("room-a", 42)  # behind
    assert flushes == 0

    store.advance("room-a", 101)  # ahead
    assert flushes == 1


def test_out_of_order_burst_settles_on_the_maximum(tmp_path):
    store = _store(tmp_path)
    for message_id in (5, 3, 9, 2, 7, 9, 4):
        store.advance("room-a", message_id)

    assert store.get("room-a") == 9
    assert _store(tmp_path).get("room-a") == 9


# --- concurrency -----------------------------------------------------------
def test_concurrent_advance_never_regresses_and_settles_on_the_maximum(tmp_path):
    """Two threads advancing the same room leave a consistent, correct file.

    Deterministic by construction rather than by timing: a ``Barrier`` makes the
    threads overlap, and the assertions are scheduling-independent invariants —
    every offset a thread observes after its own advance is >= what it saw before
    (monotonicity holds under ANY interleaving), and the settled value is the
    maximum id either thread offered. The on-disk value is asserted equal to the
    in-memory one, so a lost or half-applied write is caught too.
    """
    store = _store(tmp_path)
    ids = list(range(1, 201))
    # Interleaved halves: each thread offers ids that are sometimes ahead of and
    # sometimes behind the other thread's latest, so regressions are attempted
    # constantly and must all be declined.
    lanes = [ids[0::2], ids[1::2]]

    barrier = threading.Barrier(len(lanes))
    errors: list[BaseException] = []
    regressions: list[tuple[int, int]] = []
    probe = threading.Lock()

    def worker(lane: list[int]) -> None:
        try:
            barrier.wait(timeout=10)
            seen = -1
            for message_id in lane:
                store.advance("room-a", message_id)
                observed = store.get("room-a")
                assert observed is not None
                if observed < seen:
                    with probe:
                        regressions.append((seen, observed))
                seen = observed
        except BaseException as exc:  # surface in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(lane,)) for lane in lanes]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert not any(t.is_alive() for t in threads)
    assert regressions == [], f"offset moved backward: {regressions}"
    assert store.get("room-a") == max(ids)
    assert _on_disk(tmp_path) == {"room-a": max(ids)}
    assert _store(tmp_path).get("room-a") == max(ids)


def test_concurrent_advance_across_rooms_keeps_every_room(tmp_path):
    """One store is shared by all room threads; a write for one room must not
    drop another room's entry (the flush serializes the whole map)."""
    store = _store(tmp_path)
    rooms = [f"room-{i}" for i in range(8)]
    barrier = threading.Barrier(len(rooms))
    errors: list[BaseException] = []

    def worker(room: str) -> None:
        try:
            barrier.wait(timeout=10)
            for message_id in range(1, 51):
                store.advance(room, message_id)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(room,)) for room in rooms]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert _on_disk(tmp_path) == dict.fromkeys(rooms, 50)


# --- load-path robustness --------------------------------------------------
def test_missing_file_loads_empty(tmp_path):
    store = _store(tmp_path, "nope.json")

    assert store.get("room-a") is None


@pytest.mark.parametrize(
    "garbage",
    ["", "<<<not json", "null", "[1, 2, 3]", '"just a string"', "{"],
    ids=["empty", "not-json", "null", "list", "string", "truncated"],
)
def test_corrupt_file_loads_empty_instead_of_crashing(tmp_path, garbage):
    """A garbage offsets file must not take the bridge down at startup — the
    room falls back to first-start behaviour and stays serviceable."""
    (tmp_path / "offsets.json").write_text(garbage, encoding="utf-8")

    store = _store(tmp_path)

    assert store.get("room-a") is None
    # And it recovers: the next advance rewrites a clean file.
    store.advance("room-a", 3)
    assert _on_disk(tmp_path) == {"room-a": 3}


@pytest.mark.parametrize(
    "value",
    ["12", 1.5, None, True, False, {"nested": 1}, [1], -1],
    ids=["str", "float", "none", "true", "false", "dict", "list", "negative"],
)
def test_malformed_entries_are_dropped_not_repaired(tmp_path, value):
    """An offset we cannot trust reads as ``None`` (skip history) rather than
    being coerced into a number that could skip past unhandled messages.

    ``true``/``false`` matter specifically: ``bool`` is an ``int`` subclass, so
    an unguarded check would accept ``true`` as the offset ``1``.
    """
    (tmp_path / "offsets.json").write_text(
        json.dumps({"room-a": value, "room-b": 9}), encoding="utf-8"
    )

    store = _store(tmp_path)

    assert store.get("room-a") is None
    assert store.get("room-b") == 9, "one bad entry must not discard the good ones"


def test_unreadable_path_fails_loudly(tmp_path):
    """OSError is deliberately not swallowed. Starting with silently empty
    offsets would skip every room's history on a broken volume."""
    directory = tmp_path / "offsets.json"
    directory.mkdir()

    with pytest.raises(OSError):
        OffsetStore(str(directory))


def test_unwritable_path_fails_loudly_on_advance(tmp_path):
    """A flush that cannot land must raise, not be swallowed: an ``advance`` the
    poller believes is durable but is not would replay the batch after a restart.

    The store opens on a healthy directory, which is then replaced by a regular
    file — a stand-in for the volume going bad under a running bridge. The flush
    can no longer create its temp file alongside the target and must say so.
    """
    directory = tmp_path / "data"
    directory.mkdir()
    store = OffsetStore(str(directory / "offsets.json"))

    directory.rmdir()
    directory.write_text("no longer a directory", encoding="utf-8")

    with pytest.raises(OSError):
        store.advance("room-a", 1)
