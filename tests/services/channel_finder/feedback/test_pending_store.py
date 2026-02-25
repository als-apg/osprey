"""Tests for PendingReviewStore - file-backed pending review storage."""

import json
import threading

import pytest

from osprey.services.channel_finder.feedback.pending_store import (
    MAX_ITEMS,
    PendingReviewStore,
)


@pytest.fixture()
def store(tmp_path):
    """Create a PendingReviewStore backed by a temp file."""
    return PendingReviewStore(tmp_path / "pending.json")


# ------------------------------------------------------------------
# 1. Capture and retrieve
# ------------------------------------------------------------------


def test_capture_returns_uuid(store):
    item_id = store.capture({"query": "show me magnets", "facility": "ALS"})
    assert isinstance(item_id, str)
    assert len(item_id) == 36  # UUID format


def test_capture_and_get(store):
    item_id = store.capture(
        {
            "query": "show me magnets",
            "facility": "ALS",
            "tool_name": "mcp__channel-finder__build_channels",
            "channel_count": 42,
            "selections": {"system": "MAG"},
        }
    )

    item = store.get_item(item_id)
    assert item is not None
    assert item["id"] == item_id
    assert item["query"] == "show me magnets"
    assert item["facility"] == "ALS"
    assert item["tool_name"] == "mcp__channel-finder__build_channels"
    assert item["channel_count"] == 42
    assert item["selections"] == {"system": "MAG"}
    assert "captured_at" in item


def test_get_missing_returns_none(store):
    assert store.get_item("nonexistent") is None


# ------------------------------------------------------------------
# 2. List ordering (newest first)
# ------------------------------------------------------------------


def test_list_items_sorted_newest_first(store):
    ids = []
    for i in range(5):
        item_id = store.capture({"query": f"query_{i}", "facility": "ALS"})
        ids.append(item_id)

    items = store.list_items()
    assert len(items) == 5
    # Most recent first
    assert items[0]["query"] == "query_4"
    assert items[-1]["query"] == "query_0"


def test_list_empty_store(store):
    assert store.list_items() == []


# ------------------------------------------------------------------
# 3. Delete
# ------------------------------------------------------------------


def test_delete_existing_item(store):
    item_id = store.capture({"query": "magnets", "facility": "ALS"})
    assert store.delete(item_id) is True
    assert store.get_item(item_id) is None


def test_delete_missing_returns_false(store):
    assert store.delete("nonexistent") is False


# ------------------------------------------------------------------
# 4. Clear
# ------------------------------------------------------------------


def test_clear_removes_all(store):
    for i in range(3):
        store.capture({"query": f"q{i}", "facility": "ALS"})

    store.clear()
    assert store.list_items() == []

    # File should still be valid
    raw = json.loads(store._path.read_text())
    assert raw["items"] == {}
    assert raw["version"] == 1


# ------------------------------------------------------------------
# 5. Cap enforcement (MAX_ITEMS = 500)
# ------------------------------------------------------------------


def test_eviction_at_cap(store):
    """Items beyond MAX_ITEMS are evicted (oldest first)."""
    for i in range(MAX_ITEMS + 10):
        store.capture({"query": f"q{i}", "facility": "ALS"})

    items = store.list_items()
    assert len(items) == MAX_ITEMS

    # The oldest 10 (q0 through q9) should be evicted
    queries = {item["query"] for item in items}
    for i in range(10):
        assert f"q{i}" not in queries
    # The newest should still be present
    assert f"q{MAX_ITEMS + 9}" in queries


# ------------------------------------------------------------------
# 6. File persistence
# ------------------------------------------------------------------


def test_file_persistence(tmp_path):
    path = tmp_path / "pending.json"

    store1 = PendingReviewStore(path)
    item_id = store1.capture({"query": "magnets", "facility": "ALS", "channel_count": 42})
    del store1

    store2 = PendingReviewStore(path)
    item = store2.get_item(item_id)
    assert item is not None
    assert item["channel_count"] == 42


def test_file_format_on_disk(store):
    store.capture({"query": "magnets", "facility": "ALS", "channel_count": 7})

    raw = json.loads(store._path.read_text())
    assert raw["version"] == 1
    assert isinstance(raw["items"], dict)
    assert len(raw["items"]) == 1

    item = next(iter(raw["items"].values()))
    assert item["query"] == "magnets"
    assert item["channel_count"] == 7


# ------------------------------------------------------------------
# 7. Concurrent access (file locking)
# ------------------------------------------------------------------


def test_concurrent_captures(tmp_path):
    """Multiple threads capturing concurrently should not corrupt data."""
    path = tmp_path / "pending.json"
    errors = []

    def capture_items(thread_id, count):
        try:
            s = PendingReviewStore(path)
            for i in range(count):
                s.capture({"query": f"t{thread_id}_q{i}", "facility": "ALS"})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=capture_items, args=(t, 10)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent capture errors: {errors}"

    store = PendingReviewStore(path)
    items = store.list_items()
    assert len(items) == 40  # 4 threads × 10 items


# ------------------------------------------------------------------
# 8. Atomic writes
# ------------------------------------------------------------------


def test_atomic_write_no_partial_file(store):
    """After a capture, the file should be valid JSON."""
    store.capture({"query": "magnets", "facility": "ALS"})
    raw = json.loads(store._path.read_text())
    assert "items" in raw


# ------------------------------------------------------------------
# 9. Missing file bootstrap
# ------------------------------------------------------------------


def test_missing_file_bootstrap(tmp_path):
    """Store on a non-existent path creates it on first write."""
    path = tmp_path / "a" / "b" / "pending.json"
    store = PendingReviewStore(path)
    store.capture({"query": "test", "facility": "ALS"})

    assert path.exists()
    items = store.list_items()
    assert len(items) == 1


# ------------------------------------------------------------------
# 10. Default values for missing fields
# ------------------------------------------------------------------


def test_default_values_for_missing_fields(store):
    item_id = store.capture({})
    item = store.get_item(item_id)
    assert item["query"] == ""
    assert item["facility"] == ""
    assert item["tool_name"] == ""
    assert item["tool_response"] == ""
    assert item["channel_count"] == 0
    assert item["selections"] == {}
    assert item["session_id"] == ""
    assert item["transcript_path"] == ""
