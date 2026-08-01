"""Unit tests for the lock-guarded JSON file store in ``osprey.bridges.core.store``.

:class:`JsonFileStore` is the persistence substrate under the dedup, history and
retry-queue stores, and it has no direct test suite of its own upstream — it is
covered there only transitively. These tests pin its contract directly:

  * the load path's *asymmetric* error handling (missing/corrupt loads empty, an
    unreadable volume fails loudly),
  * the ``_coerce`` normalization hook,
  * the atomic tmp-then-``os.replace`` write, including tmp cleanup on failure,
  * the ``threading.Lock`` guard that makes a shared store safe under the
    bridges' thread-per-callback dispatch model,
  * and the **on-disk JSON shape**, asserted byte-for-byte. A live deployment's
    state files must stay readable across upgrades, so a format drift here is a
    compatibility break and must fail loudly rather than quietly re-serialize.

The store is exercised through a minimal concrete subclass, mirroring how the
real stores use it: mutate ``_data`` under ``_lock``, then ``_flush_locked``.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from typing import Any

import pytest

from osprey.bridges.core.store import JsonFileStore


class _Counter(JsonFileStore):
    """Smallest realistic subclass: read/modify/write under the inherited lock."""

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._flush_locked()

    def get(self, key: str) -> Any:
        with self._lock:
            return self._data.get(key)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)


class _Coercing(JsonFileStore):
    """Subclass that records and rewrites what ``_load`` hands to ``_coerce``."""

    def __init__(self, path: str):
        self.seen: list[dict] = []
        super().__init__(path)

    def _coerce(self, loaded: dict) -> dict:
        self.seen.append(dict(loaded))
        return {k: v for k, v in loaded.items() if not k.startswith("_")}


# --------------------------------------------------------------------------- load


def test_missing_file_loads_empty_and_creates_nothing(tmp_path):
    """A store pointed at a nonexistent path starts empty without touching disk."""
    path = tmp_path / "state.json"

    store = JsonFileStore(str(path))

    assert store._data == {}
    assert not path.exists()
    assert store.path == str(path)


def test_corrupt_json_loads_empty(tmp_path):
    """Unparseable content must load empty rather than crash the bridge at startup."""
    path = tmp_path / "state.json"
    path.write_text("{not json at all", encoding="utf-8")

    assert JsonFileStore(str(path))._data == {}


def test_non_utf8_bytes_load_empty(tmp_path):
    """UnicodeDecodeError is a ValueError — a mojibake file also loads empty."""
    path = tmp_path / "state.json"
    path.write_bytes(b'{"k": "\xff\xfe"}')

    assert JsonFileStore(str(path))._data == {}


@pytest.mark.parametrize("payload", ["[1, 2, 3]", "null", '"a string"', "42"])
def test_valid_json_that_is_not_an_object_leaves_data_untouched(tmp_path, payload):
    """Only a JSON *object* is adopted; anything else leaves ``_data`` as it was.

    On a fresh instance that means empty. Note this is NOT the same as the
    corrupt path: ``_load`` does not reset ``_data`` here, it simply declines to
    replace it — see :func:`test_reload_over_a_non_object_file_keeps_prior_data`.
    """
    path = tmp_path / "state.json"
    path.write_text(payload, encoding="utf-8")

    assert JsonFileStore(str(path))._data == {}


def test_reload_over_a_non_object_file_keeps_prior_data(tmp_path):
    """Pins the asymmetry: a non-object reload keeps state, a corrupt one clears it."""
    path = tmp_path / "state.json"
    store = _Counter(str(path))
    store.put("a", 1)

    path.write_text("[]", encoding="utf-8")
    store._load()
    assert store._data == {"a": 1}  # declined, not cleared

    path.write_text("<<<", encoding="utf-8")
    store._load()
    assert store._data == {}  # corrupt clears


def test_unreadable_path_raises_rather_than_loading_empty(tmp_path):
    """OSError is deliberately NOT swallowed: a broken volume must fail loudly.

    Silently starting with an empty dedup store would re-dispatch already
    answered messages, so only FileNotFoundError/ValueError load as empty.
    """
    directory = tmp_path / "state.json"
    directory.mkdir()

    with pytest.raises(OSError):
        JsonFileStore(str(directory))


def test_coerce_hook_normalizes_the_loaded_mapping(tmp_path):
    """``_coerce`` sees the raw mapping and its return value becomes the state."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"keep": 1, "_drop": 2}), encoding="utf-8")

    store = _Coercing(str(path))

    assert store.seen == [{"keep": 1, "_drop": 2}]
    assert store._data == {"keep": 1}


def test_coerce_hook_is_not_called_when_the_file_is_absent(tmp_path):
    """No file, no mapping to normalize — subclasses may rely on this."""
    store = _Coercing(str(tmp_path / "missing.json"))

    assert store.seen == []
    assert store._data == {}


# ------------------------------------------------------------------------ write


def test_write_then_read_round_trip_preserves_structure_and_types(tmp_path):
    """A fresh instance reading the same path recovers the exact nested value."""
    path = tmp_path / "state.json"
    payload = {
        "msg-1": {
            "status": "completed",
            "attempts": 3,
            "score": 1.5,
            "done": True,
            "run_id": None,
            "artifacts": ["a.png", "b.txt"],
            "nested": {"turns": [{"question": "q", "answer": "a"}]},
        }
    }

    store = _Counter(str(path))
    store.put("msg-1", payload["msg-1"])

    reloaded = _Counter(str(path))
    assert reloaded.snapshot() == payload
    entry = reloaded.get("msg-1")
    assert isinstance(entry["attempts"], int)
    assert isinstance(entry["score"], float)
    assert entry["done"] is True
    assert entry["run_id"] is None


def test_on_disk_artifact_is_plain_json_with_the_documented_shape(tmp_path):
    """Format guard: 2-space-indented, ASCII-escaped JSON object, no trailing newline.

    A deployment upgrading onto this engine must be able to read its existing
    state files, so any change to this serialization is a compatibility break.
    """
    path = tmp_path / "state.json"
    store = _Counter(str(path))
    store.put("msg-1", {"status": "completed", "note": "café"})

    raw = path.read_text(encoding="utf-8")

    assert json.loads(raw) == {"msg-1": {"status": "completed", "note": "café"}}
    assert raw.startswith("{\n")
    assert '\n  "msg-1": {\n' in raw  # indent=2, default separators
    assert "caf\\u00e9" in raw  # ensure_ascii default
    assert not raw.endswith("\n")


def test_empty_store_writes_an_empty_json_object(tmp_path):
    """The empty state is ``{}`` on disk, and reloads as empty."""
    path = tmp_path / "state.json"
    store = _Counter(str(path))

    with store._lock:
        store._flush_locked()

    assert path.read_text(encoding="utf-8") == "{}"
    assert _Counter(str(path))._data == {}


def test_flush_creates_missing_parent_directories(tmp_path):
    """First write on a fresh volume must not require the directory to exist."""
    path = tmp_path / "deep" / "nested" / "state.json"
    store = _Counter(str(path))

    store.put("k", "v")

    assert json.loads(path.read_text(encoding="utf-8")) == {"k": "v"}


def test_flush_handles_a_bare_filename_relative_to_cwd(tmp_path, monkeypatch):
    """``os.path.dirname`` is empty for a bare filename; the store falls back to "."."""
    monkeypatch.chdir(tmp_path)
    store = _Counter("state.json")

    store.put("k", "v")

    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8")) == {"k": "v"}


def test_successful_write_leaves_no_temporary_files(tmp_path):
    """The tmp file is renamed into place, never left behind."""
    path = tmp_path / "state.json"
    store = _Counter(str(path))

    for i in range(5):
        store.put(f"k{i}", i)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


def test_failed_dump_unlinks_the_tmp_file_and_reraises(tmp_path):
    """A non-serializable value must not leave debris or a truncated state file."""
    path = tmp_path / "state.json"
    store = _Counter(str(path))
    store.put("good", 1)
    before = path.read_text(encoding="utf-8")

    with pytest.raises(TypeError):
        store.put("bad", object())

    assert path.read_text(encoding="utf-8") == before  # previous state intact
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_tmp_file_is_created_in_the_target_directory(tmp_path, monkeypatch):
    """``os.replace`` is only atomic within one filesystem, so the tmp file must
    be a sibling of the target — not in the system temp dir."""
    path = tmp_path / "deep" / "state.json"
    store = _Counter(str(path))
    seen: list[str] = []

    real_mkstemp = tempfile.mkstemp

    def spy(*args, **kwargs):
        seen.append(kwargs.get("dir", ""))
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr("osprey.bridges.core.store.tempfile.mkstemp", spy)
    store.put("k", "v")

    assert seen == [str(tmp_path / "deep")]


# ------------------------------------------------------------------ concurrency


def test_lock_serializes_concurrent_read_modify_write(tmp_path):
    """The ``threading.Lock`` guard must prevent lost updates.

    This is the property the adapters' thread-per-room concurrency model depends
    on: many threads mutate one shared store. Each worker does a genuine
    read/modify/write with a yield in the middle of the critical section, so an
    unguarded implementation would interleave and lose increments. The store is
    also instrumented to assert mutual exclusion directly.
    """
    path = tmp_path / "state.json"
    store = _Counter(str(path))
    store.put("n", 0)

    workers, per_worker = 8, 40
    barrier = threading.Barrier(workers)
    inside = 0
    overlaps: list[int] = []
    probe = threading.Lock()
    errors: list[BaseException] = []

    def bump() -> None:
        nonlocal inside
        try:
            barrier.wait(timeout=10)
            for _ in range(per_worker):
                with store._lock:
                    with probe:
                        inside += 1
                        if inside > 1:
                            overlaps.append(inside)
                    current = store._data["n"]
                    time.sleep(0)  # yield the GIL mid-critical-section
                    store._data["n"] = current + 1
                    store._flush_locked()
                    with probe:
                        inside -= 1
        except BaseException as exc:  # surface in the main thread
            errors.append(exc)

    threads = [threading.Thread(target=bump) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert not any(t.is_alive() for t in threads)
    assert overlaps == [], "two threads were inside the guarded region at once"
    assert store.get("n") == workers * per_worker
    assert json.loads(path.read_text(encoding="utf-8")) == {"n": workers * per_worker}


def test_concurrent_writers_never_expose_a_partial_file(tmp_path):
    """A reader racing the writers always sees a complete, parseable JSON object.

    This is what the tmp-then-``os.replace`` write buys: readers never observe a
    half-written state file.
    """
    path = tmp_path / "state.json"
    store = _Counter(str(path))
    store.put("seed", 0)

    stop = threading.Event()
    reads = 0
    errors: list[BaseException] = []

    def write() -> None:
        try:
            for i in range(150):
                store.put(f"k{i}", {"i": i, "blob": "x" * 200})
        except BaseException as exc:
            errors.append(exc)
        finally:
            stop.set()

    def read() -> None:
        nonlocal reads
        try:
            while not stop.is_set():
                try:
                    parsed = json.loads(path.read_text(encoding="utf-8"))
                except FileNotFoundError:
                    continue
                assert isinstance(parsed, dict) and "seed" in parsed
                reads += 1
        except BaseException as exc:
            errors.append(exc)

    writer, reader = threading.Thread(target=write), threading.Thread(target=read)
    writer.start()
    reader.start()
    writer.join(timeout=30)
    reader.join(timeout=30)

    assert errors == []
    assert reads, "reader never observed the file"
    assert len(store.snapshot()) == 151
