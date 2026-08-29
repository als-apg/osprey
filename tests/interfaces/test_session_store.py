"""Unit tests for :class:`osprey.interfaces.web_auth.SessionStore`.

The store's whole contract is that it never costs anyone a login: a missing,
corrupt, or unwritable file must degrade to an in-memory-only process with at
most one warning, and a slow thread's stale snapshot must never overwrite a
newer one. These tests exercise those paths directly, constructing a store on a
``tmp_path`` rather than going through the process-wide credentials.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pytest

from osprey.interfaces.web_auth import SessionStore

LOGGER_NAME = "osprey.interfaces.web_auth"


@pytest.fixture
def store(tmp_path):
    """A store in a directory that does not exist yet, so ``__init__`` mkdirs it."""
    return SessionStore(tmp_path / "web_terminal", "8087")


def _warnings(caplog):
    """Every WARNING record captured, regardless of logger."""
    return [record for record in caplog.records if record.levelno == logging.WARNING]


def test_path_is_port_scoped_under_the_store_dir(tmp_path):
    """Two terminals on one host must not share a store file."""
    store = SessionStore(tmp_path / "web_terminal", "8087")
    assert store.path == tmp_path / "web_terminal" / "sessions-8087.json"
    assert store.path.parent.is_dir()


def test_path_is_portless_when_no_port_is_given(tmp_path):
    """An empty port yields the bare name rather than ``sessions-.json``."""
    store = SessionStore(tmp_path / "web_terminal", "")
    assert store.path == tmp_path / "web_terminal" / "sessions.json"


def test_roundtrip_returns_what_was_saved(store):
    """The ordinary path: save a map, read the same map back."""
    snapshot = {"a" * 64: 1_700_000_000.5, "b" * 64: 1_700_000_060.0}
    store.save(snapshot, 1)

    assert store.load() == snapshot
    assert json.loads(store.path.read_text()) == {"v": 1, "sessions": snapshot}


def test_missing_file_loads_empty_and_does_not_warn(store, caplog):
    """A fresh deployment has no store; that is the first start, not a failure."""
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)

    assert not store.path.exists()
    assert store.load() == {}
    assert _warnings(caplog) == []


def test_corrupt_file_loads_empty_with_exactly_one_warning(store, caplog):
    """A truncated store costs a re-login and one log line, never an exception."""
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    store.path.write_text('{"v": 1, "sessions": {"abc": 1700')

    assert store.load() == {}
    assert store.load() == {}  # the second read is silenced, not re-reported

    records = _warnings(caplog)
    assert len(records) == 1
    assert str(store.path) in records[0].getMessage()


def test_file_that_is_not_utf8_loads_empty(store, caplog):
    """Bytes that do not decode are an unusable file, not an exception.

    A decode failure is a ``UnicodeDecodeError`` — a ``ValueError``, not an
    ``OSError`` — so it is the one corruption that could slip past the read
    guard. It must not, because the caller is process population: a raise here
    would refuse every request the process would ever serve, over a file that
    holds nothing but deadlines.
    """
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    store.path.write_bytes(b"\xff\xfe not utf8")

    assert store.load() == {}
    assert len(_warnings(caplog)) == 1


def test_wrong_version_loads_empty(store, caplog):
    """A payload this code does not know how to read is discarded, not guessed at."""
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    store.path.write_text(json.dumps({"v": 2, "sessions": {"a" * 64: 1_700_000_000.0}}))

    assert store.load() == {}
    assert len(_warnings(caplog)) == 1


def test_non_dict_payload_loads_empty(store, caplog):
    """Valid JSON that is not an object is still not a store."""
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    store.path.write_text(json.dumps(["a" * 64, 1_700_000_000.0]))

    assert store.load() == {}
    assert len(_warnings(caplog)) == 1


def test_non_dict_sessions_loads_empty(store, caplog):
    """The ``sessions`` field must itself be an object."""
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    store.path.write_text(json.dumps({"v": 1, "sessions": ["a" * 64]}))

    assert store.load() == {}
    assert len(_warnings(caplog)) == 1


@pytest.mark.parametrize(
    "sessions",
    [
        pytest.param({"a" * 64: "soon"}, id="deadline-is-a-string"),
        pytest.param({"a" * 64: None}, id="deadline-is-null"),
        pytest.param({"a" * 64: True}, id="deadline-is-a-bool"),
    ],
)
def test_malformed_entry_discards_the_whole_map(store, sessions):
    """A store that has been corrupted has no claim to the entries that still parse."""
    store.path.write_text(json.dumps({"v": 1, "sessions": sessions}))

    assert store.load() == {}


def test_integer_deadlines_load_as_floats(store):
    """A hand-edited or externally written store may spell an epoch as an int."""
    store.path.write_text(json.dumps({"v": 1, "sessions": {"a" * 64: 1_700_000_000}}))

    loaded = store.load()
    assert loaded == {"a" * 64: 1_700_000_000.0}
    assert isinstance(loaded["a" * 64], float)


def test_stale_sequence_is_dropped(store):
    """A slower thread's older snapshot must not undo the newer one already on disk."""
    newer = {"a" * 64: 1_700_000_060.0}
    older = {"a" * 64: 1_700_000_000.0, "b" * 64: 1_700_000_000.0}

    store.save(newer, 2)
    store.save(older, 1)

    assert store.load() == newer


def test_equal_sequence_is_dropped(store):
    """The sequence guard is strict: a repeat of the last stamp is not newer."""
    store.save({"a" * 64: 1.0}, 2)
    store.save({"b" * 64: 2.0}, 2)

    assert store.load() == {"a" * 64: 1.0}


def test_higher_sequence_replaces_content_without_leaving_debris(store):
    """A later save wins, and the temporary file it wrote through is gone."""
    store.save({"a" * 64: 1_700_000_000.0}, 1)
    store.save({"b" * 64: 1_700_000_060.0}, 2)

    assert store.load() == {"b" * 64: 1_700_000_060.0}
    assert sorted(p.name for p in store.path.parent.iterdir()) == [store.path.name]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_saved_file_is_owner_only(store):
    """The store is world-unreadable even though what it holds authenticates nobody."""
    store.save({"a" * 64: 1_700_000_000.0}, 1)

    assert store.path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory modes")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory modes"
)
def test_unwritable_directory_warns_once_and_never_raises(tmp_path, caplog):
    """A full or read-only disk must cost a warning, not a refused login."""
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    store_dir = tmp_path / "web_terminal"
    store = SessionStore(store_dir, "8087")

    store_dir.chmod(0o500)
    try:
        store.save({"a" * 64: 1_700_000_000.0}, 1)
        store.save({"b" * 64: 1_700_000_060.0}, 2)
    finally:
        store_dir.chmod(0o700)

    records = _warnings(caplog)
    assert len(records) == 1
    assert str(store.path) in records[0].getMessage()
    assert not store.path.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory modes")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory modes"
)
def test_unwritable_directory_keeps_retrying(tmp_path, caplog):
    """Silenced is not given up on: a disk that frees up starts persisting again."""
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    store_dir = tmp_path / "web_terminal"
    store = SessionStore(store_dir, "8087")

    store_dir.chmod(0o500)
    try:
        store.save({"a" * 64: 1_700_000_000.0}, 1)
    finally:
        store_dir.chmod(0o700)
    store.save({"b" * 64: 1_700_000_060.0}, 2)

    assert store.load() == {"b" * 64: 1_700_000_060.0}
    assert len(_warnings(caplog)) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory modes")
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory modes"
)
def test_directory_that_cannot_be_created_warns_at_construction(tmp_path, caplog):
    """An mkdir the process is not allowed to do is the first write failure."""
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    parent = tmp_path / "agent_data"
    parent.mkdir()
    parent.chmod(0o500)
    try:
        store = SessionStore(parent / "web_terminal", "8087")
        store.save({"a" * 64: 1_700_000_000.0}, 1)
    finally:
        parent.chmod(0o700)

    assert len(_warnings(caplog)) == 1
    assert store.load() == {}


def test_unreadable_file_loads_empty_with_one_warning(store, caplog):
    """A store owned by another user reads as no store at all."""
    caplog.set_level(logging.WARNING, logger=LOGGER_NAME)
    store.path.mkdir()  # a directory where a file belongs: read_text raises OSError

    assert store.load() == {}
    assert len(_warnings(caplog)) == 1
