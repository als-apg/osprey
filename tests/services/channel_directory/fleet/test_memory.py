"""The fleet's persisted memory (D9): what was seen, per target and read path."""

from __future__ import annotations

import json
import logging

import pytest

pytest.importorskip("osprey_connectors.safety_file")

from osprey.services.channel_directory.fleet.memory import (
    MEMORY_FILE,
    FleetMemory,
    GoodListing,
    RememberedServer,
)

VA = ("va", "in_process")
LIVE = ("live", "wire")
DAY = 86_400.0


def _ioc(name: str = "ioc-a", *, at: float = 1_000.0, **overrides) -> RememberedServer:
    fields = {
        "name": name,
        "last_answered": at,
        "host": "host-1",
        "server_class": "default",
        "liveness_channel": f"{name}:HEARTBEAT",
        "health_channels": (f"{name}:UPTIME",),
        "counter": 42.0,
        "counter_read_at": at,
    }
    fields.update(overrides)
    return RememberedServer(**fields)


def _fresh(tmp_path, retention: float = 7 * DAY) -> FleetMemory:
    return FleetMemory.load(tmp_path, retention=retention)


# --- Loading -----------------------------------------------------------------


def test_missing_file_starts_empty_and_is_reported(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        memory = FleetMemory.load(tmp_path / "absent", retention=DAY)
    assert memory.load_problem is not None
    assert "no memory" in memory.load_problem
    assert caplog.records
    assert not memory.seen_before(*VA, "ioc-a")
    assert memory.last_cycle_time(*VA) is None
    assert memory.last_good_listing(*VA) is None


@pytest.mark.parametrize(
    "content",
    [
        b"{not json",
        b"\xff\xfe",
        b"[]",
        b'{"version": 99, "paths": []}',
        b'{"version": 1, "paths": [{"target": "va"}]}',
        b'{"version": 1, "paths": [{"target": "va", "read_path": "x", "last_cycle": null,'
        b' "listing": null, "servers": {"a": {"name": "a"}}}]}',
        b'{"version": 1, "paths": [{"target": "va", "read_path": "x", "last_cycle": "soon",'
        b' "listing": null, "servers": {}}]}',
    ],
    ids=[
        "truncated",
        "not-utf8",
        "wrong-top",
        "unknown-version",
        "missing-keys",
        "short-ioc",
        "bad-time",
    ],
)
def test_corrupt_file_starts_empty_and_is_reported(tmp_path, caplog, content):
    (tmp_path / MEMORY_FILE).write_bytes(content)
    with caplog.at_level(logging.WARNING):
        memory = FleetMemory.load(tmp_path, retention=DAY)
    assert memory.load_problem is not None
    assert "corrupt" in memory.load_problem
    assert caplog.records
    assert not memory.seen_before("va", "x", "a")
    assert memory.last_cycle_time("va", "x") is None


def test_unreadable_path_is_reported_not_fatal(tmp_path):
    (tmp_path / MEMORY_FILE).mkdir()
    memory = FleetMemory.load(tmp_path, retention=DAY)
    assert memory.load_problem is not None


def test_good_file_loads_without_a_problem(tmp_path):
    memory = _fresh(tmp_path)
    memory.save()
    again = _fresh(tmp_path)
    assert again.load_problem is None


def test_retention_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        FleetMemory.load(tmp_path, retention=0)


# --- Round trip --------------------------------------------------------------


def test_everything_survives_a_restart(tmp_path):
    memory = _fresh(tmp_path)
    memory.record_answer(*VA, _ioc("ioc-a", at=1_000.0))
    memory.record_answer(
        *VA,
        _ioc(
            "ioc-b",
            at=1_100.0,
            host=None,
            counter=None,
            counter_read_at=None,
            liveness_channel=None,
            health_channels=(),
        ),
    )
    memory.record_cycle(*VA, 1_200.0)
    listing = [{"name": "ioc-a", "host": "host-1"}]
    lookup = {"ioc-a": {"liveness": "ioc-a:HEARTBEAT"}}
    memory.record_good_listing(*VA, listing=listing, lookup=lookup, at=1_150.0)
    memory.save()

    again = _fresh(tmp_path)
    assert again.load_problem is None
    assert again.server(*VA, "ioc-a") == _ioc("ioc-a", at=1_000.0)
    assert again.server(*VA, "ioc-b").host is None
    assert again.server(*VA, "ioc-b").health_channels == ()
    assert again.last_cycle_time(*VA) == 1_200.0
    assert again.last_good_listing(*VA) == GoodListing(listing=listing, lookup=lookup, at=1_150.0)


def test_save_creates_the_directory_and_writes_one_file(tmp_path):
    state_dir = tmp_path / "state" / "fleet"
    memory = FleetMemory.load(state_dir, retention=DAY)
    memory.record_cycle(*VA, 5.0)
    memory.save()
    assert sorted(p.name for p in state_dir.iterdir()) == [MEMORY_FILE]
    json.loads((state_dir / MEMORY_FILE).read_text())


def test_save_is_atomic_on_failure(tmp_path, monkeypatch):
    memory = _fresh(tmp_path)
    memory.record_cycle(*VA, 5.0)
    memory.save()
    before = (tmp_path / MEMORY_FILE).read_bytes()

    import osprey_connectors.safety_file as safety_file

    def boom(*args, **kwargs):
        raise OSError("disk refused")

    monkeypatch.setattr(safety_file.os, "fsync", boom)
    memory.record_cycle(*VA, 6.0)
    with pytest.raises(OSError):
        memory.save()
    assert (tmp_path / MEMORY_FILE).read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [MEMORY_FILE]


# --- Questions ---------------------------------------------------------------


def test_seen_before_and_previous_reading(tmp_path):
    memory = _fresh(tmp_path)
    assert not memory.seen_before(*VA, "ioc-a")
    assert memory.previous_reading(*VA, "ioc-a") is None
    memory.record_answer(*VA, _ioc("ioc-a", counter=7.0, counter_read_at=900.0))
    assert memory.seen_before(*VA, "ioc-a")
    assert memory.previous_reading(*VA, "ioc-a") == (7.0, 900.0)


def test_no_previous_reading_without_a_counter(tmp_path):
    memory = _fresh(tmp_path)
    memory.record_answer(*VA, _ioc(counter=None, counter_read_at=None))
    assert memory.seen_before(*VA, "ioc-a")
    assert memory.previous_reading(*VA, "ioc-a") is None


def test_targets_and_read_paths_are_kept_apart(tmp_path):
    memory = _fresh(tmp_path)
    memory.record_answer(*VA, _ioc("ioc-a"))
    memory.record_cycle(*VA, 10.0)
    assert not memory.seen_before(*LIVE, "ioc-a")
    assert not memory.seen_before("va", "wire", "ioc-a")
    assert memory.last_cycle_time(*LIVE) is None
    memory.save()
    again = _fresh(tmp_path)
    assert again.seen_before(*VA, "ioc-a")
    assert not again.seen_before(*LIVE, "ioc-a")


def test_a_new_answer_replaces_the_old_one(tmp_path):
    memory = _fresh(tmp_path)
    memory.record_answer(*VA, _ioc(at=1.0, host="old"))
    memory.record_answer(*VA, _ioc(at=2.0, host="new"))
    assert memory.server(*VA, "ioc-a").host == "new"
    assert memory.server(*VA, "ioc-a").last_answered == 2.0


def test_last_good_listing_is_replaced(tmp_path):
    memory = _fresh(tmp_path)
    memory.record_good_listing(*VA, listing=["a"], lookup={}, at=1.0)
    memory.record_good_listing(*VA, listing=["b"], lookup={"b": 1}, at=2.0)
    assert memory.last_good_listing(*VA) == GoodListing(listing=["b"], lookup={"b": 1}, at=2.0)


def test_listing_must_be_json(tmp_path):
    memory = _fresh(tmp_path)
    with pytest.raises(TypeError):
        memory.record_good_listing(*VA, listing=[object()], lookup={}, at=1.0)
    assert memory.last_good_listing(*VA) is None


# --- No longer listed and retention ------------------------------------------


def test_no_longer_listed_names_the_addresses_to_read(tmp_path):
    memory = _fresh(tmp_path)
    memory.record_answer(*VA, _ioc("ioc-a"))
    memory.record_answer(*VA, _ioc("ioc-b"))
    memory.record_answer(*VA, _ioc("ioc-c"))
    gone = memory.no_longer_listed(*VA, listed={"ioc-b"})
    assert [r.name for r in gone] == ["ioc-a", "ioc-c"]
    assert gone[0].liveness_channel == "ioc-a:HEARTBEAT"
    assert gone[0].health_channels == ("ioc-a:UPTIME",)
    assert memory.no_longer_listed(*LIVE, listed=set()) == []


def test_quiet_unlisted_server_is_dropped_after_retention(tmp_path):
    memory = _fresh(tmp_path, retention=100.0)
    memory.record_answer(*VA, _ioc("old", at=1_000.0))
    memory.record_answer(*VA, _ioc("recent", at=1_050.0))
    # Exactly at the retention period it is still kept.
    assert memory.prune(*VA, listed=set(), now=1_100.0) == []
    assert memory.prune(*VA, listed=set(), now=1_100.5) == ["old"]
    assert not memory.seen_before(*VA, "old")
    assert memory.seen_before(*VA, "recent")
    memory.save()
    assert not _fresh(tmp_path, retention=100.0).seen_before(*VA, "old")


def test_listed_server_is_never_dropped(tmp_path):
    memory = _fresh(tmp_path, retention=10.0)
    memory.record_answer(*VA, _ioc("ioc-a", at=0.0))
    assert memory.prune(*VA, listed={"ioc-a"}, now=1_000_000.0) == []
    assert memory.seen_before(*VA, "ioc-a")


def test_prune_touches_only_its_own_target(tmp_path):
    memory = _fresh(tmp_path, retention=10.0)
    memory.record_answer(*VA, _ioc("ioc-a", at=0.0))
    memory.record_answer(*LIVE, _ioc("ioc-a", at=0.0))
    assert memory.prune(*VA, listed=set(), now=100.0) == ["ioc-a"]
    assert memory.seen_before(*LIVE, "ioc-a")
