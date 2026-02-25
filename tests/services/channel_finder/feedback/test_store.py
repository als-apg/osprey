"""Tests for FeedbackStore - file-backed feedback storage for channel finder."""

import json

import pytest

from osprey.services.channel_finder.feedback.store import FeedbackStore


@pytest.fixture()
def store(tmp_path):
    """Create a FeedbackStore backed by a temp file."""
    return FeedbackStore(tmp_path / "feedback.json")


# ------------------------------------------------------------------
# 1. Record and retrieve a success
# ------------------------------------------------------------------


def test_record_and_retrieve_success(store):
    store.record_success(
        query="show me magnets",
        facility="ALS",
        selections={"system": "MAG", "device": "QF1"},
        channel_count=42,
    )

    hints = store.get_hints("show me magnets", "ALS")
    assert len(hints) == 1
    assert hints[0]["selections"] == {"system": "MAG", "device": "QF1"}
    assert hints[0]["channel_count"] == 42


# ------------------------------------------------------------------
# 2. Different facilities are isolated
# ------------------------------------------------------------------


def test_different_facilities_are_isolated(store):
    store.record_success(
        query="show me magnets",
        facility="ALS",
        selections={"system": "MAG"},
        channel_count=10,
    )
    store.record_success(
        query="show me magnets",
        facility="LCLS",
        selections={"system": "QUAD"},
        channel_count=20,
    )

    als_hints = store.get_hints("show me magnets", "ALS")
    lcls_hints = store.get_hints("show me magnets", "LCLS")

    assert len(als_hints) == 1
    assert als_hints[0]["selections"] == {"system": "MAG"}

    assert len(lcls_hints) == 1
    assert lcls_hints[0]["selections"] == {"system": "QUAD"}


# ------------------------------------------------------------------
# 3. Query normalization
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant",
    [
        "Show Me Magnets",  # case
        "  show  me   magnets  ",  # whitespace
        "show me magnets!",  # punctuation
        "SHOW ME MAGNETS???",  # case + punctuation
        "show, me: magnets.",  # mixed punctuation
    ],
)
def test_query_normalization(store, variant):
    """All normalised variants of a query should resolve to the same key."""
    store.record_success(
        query="show me magnets",
        facility="ALS",
        selections={"system": "MAG"},
        channel_count=5,
    )

    hints = store.get_hints(variant, "ALS")
    assert len(hints) == 1
    assert hints[0]["selections"] == {"system": "MAG"}


def test_normalize_query_directly():
    """Verify _normalize_query behaviour in isolation."""
    assert FeedbackStore._normalize_query("  Hello,  World!  ") == "hello world"
    assert FeedbackStore._normalize_query("ALL-CAPS???") == "allcaps"
    assert FeedbackStore._normalize_query("already clean") == "already clean"


# ------------------------------------------------------------------
# 4. Deduplication of identical selections
# ------------------------------------------------------------------


def test_deduplication_of_identical_selections(store):
    for _ in range(5):
        store.record_success(
            query="magnets",
            facility="ALS",
            selections={"system": "MAG"},
            channel_count=10,
        )

    hints = store.get_hints("magnets", "ALS")
    assert len(hints) == 1


def test_deduplication_ignores_channel_count_difference(store):
    """Even if channel_count differs, identical selections are deduplicated."""
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=10
    )
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=20
    )

    hints = store.get_hints("magnets", "ALS")
    assert len(hints) == 1
    # The first recording is kept (dedup returns early on match)
    assert hints[0]["channel_count"] == 10


def test_different_selections_are_not_deduplicated(store):
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=10
    )
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "QUAD"}, channel_count=20
    )

    hints = store.get_hints("magnets", "ALS")
    assert len(hints) == 2


# ------------------------------------------------------------------
# 5. Cap enforcement (MAX_ENTRIES_PER_KEY = 10)
# ------------------------------------------------------------------


def test_successes_capped_at_max_entries(store):
    for i in range(15):
        store.record_success(
            query="magnets",
            facility="ALS",
            selections={"system": f"SYS_{i}"},
            channel_count=i,
        )

    # All 15 are unique so no dedup; should cap at 10 most recent
    hints = store.get_hints("magnets", "ALS", max_hints=20)
    assert len(hints) == 10

    # Most recent first: SYS_14 should be first, SYS_5 should be last
    assert hints[0]["selections"] == {"system": "SYS_14"}
    assert hints[-1]["selections"] == {"system": "SYS_5"}


def test_failures_capped_at_max_entries(store):
    for i in range(15):
        store.record_failure(
            query="magnets",
            facility="ALS",
            partial_selections={"system": f"SYS_{i}"},
            reason=f"reason_{i}",
        )

    # Read raw data to verify the cap on failures
    raw = json.loads(store._path.read_text())
    key = FeedbackStore._make_key("magnets", "ALS")
    assert len(raw["entries"][key]["failures"]) == 10
    # Most recent should be the last 10 (indices 5-14)
    assert raw["entries"][key]["failures"][-1]["reason"] == "reason_14"
    assert raw["entries"][key]["failures"][0]["reason"] == "reason_5"


# ------------------------------------------------------------------
# 6. File persistence (close and reopen)
# ------------------------------------------------------------------


def test_file_persistence(tmp_path):
    path = tmp_path / "feedback.json"

    store1 = FeedbackStore(path)
    store1.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=42
    )
    del store1  # Let GC drop it

    store2 = FeedbackStore(path)
    hints = store2.get_hints("magnets", "ALS")
    assert len(hints) == 1
    assert hints[0]["channel_count"] == 42


def test_file_format_on_disk(store):
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=7
    )

    raw = json.loads(store._path.read_text())
    assert raw["version"] == 2
    assert isinstance(raw["entries"], dict)

    key = FeedbackStore._make_key("magnets", "ALS")
    assert key in raw["entries"]
    assert len(raw["entries"][key]["successes"]) == 1
    assert raw["entries"][key]["successes"][0]["selections"] == {"system": "MAG"}


# ------------------------------------------------------------------
# 7. clear() wipes everything
# ------------------------------------------------------------------


def test_clear_wipes_all_data(store):
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=10
    )
    store.record_failure(query="bpms", facility="ALS", partial_selections={}, reason="not found")

    store.clear()

    assert store.get_hints("magnets", "ALS") == []
    assert store.get_hints("bpms", "ALS") == []

    # File should still exist but be empty of entries
    raw = json.loads(store._path.read_text())
    assert raw["entries"] == {}
    assert raw["version"] == 2


# ------------------------------------------------------------------
# 8. get_hints returns most recent first, capped at max_hints
# ------------------------------------------------------------------


def test_get_hints_most_recent_first(store):
    for i in range(5):
        store.record_success(
            query="magnets",
            facility="ALS",
            selections={"system": f"SYS_{i}"},
            channel_count=i,
        )

    hints = store.get_hints("magnets", "ALS", max_hints=3)
    assert len(hints) == 3
    # Most recent (SYS_4) first, then SYS_3, SYS_2
    assert hints[0]["selections"] == {"system": "SYS_4"}
    assert hints[1]["selections"] == {"system": "SYS_3"}
    assert hints[2]["selections"] == {"system": "SYS_2"}


def test_get_hints_default_max_is_three(store):
    for i in range(5):
        store.record_success(
            query="magnets",
            facility="ALS",
            selections={"system": f"SYS_{i}"},
            channel_count=i,
        )

    hints = store.get_hints("magnets", "ALS")
    assert len(hints) == 3  # default max_hints=3


# ------------------------------------------------------------------
# 9. record_failure stores failures
# ------------------------------------------------------------------


def test_record_failure(store):
    store.record_failure(
        query="broken query",
        facility="ALS",
        partial_selections={"system": "MAG"},
        reason="no options at family level",
    )

    # Failures are not returned by get_hints (which only returns successes)
    hints = store.get_hints("broken query", "ALS")
    assert hints == []

    # But they are stored in the raw data
    raw = json.loads(store._path.read_text())
    key = FeedbackStore._make_key("broken query", "ALS")
    failures = raw["entries"][key]["failures"]
    assert len(failures) == 1
    assert failures[0]["partial_selections"] == {"system": "MAG"}
    assert failures[0]["reason"] == "no options at family level"
    assert "timestamp" in failures[0]


def test_record_multiple_failures(store):
    for i in range(3):
        store.record_failure(
            query="bad query",
            facility="ALS",
            partial_selections={"step": str(i)},
            reason=f"fail reason {i}",
        )

    raw = json.loads(store._path.read_text())
    key = FeedbackStore._make_key("bad query", "ALS")
    assert len(raw["entries"][key]["failures"]) == 3


def test_failure_deduplication(store):
    """Identical (selections, reason) pairs should be deduplicated."""
    for _ in range(5):
        store.record_failure(
            query="bad query",
            facility="ALS",
            partial_selections={"system": "MAG"},
            reason="no options at family level",
        )

    raw = json.loads(store._path.read_text())
    key = FeedbackStore._make_key("bad query", "ALS")
    assert len(raw["entries"][key]["failures"]) == 1


def test_failure_different_reasons_not_deduplicated(store):
    """Same selections but different reasons are separate failures."""
    store.record_failure(
        query="bad query",
        facility="ALS",
        partial_selections={"system": "MAG"},
        reason="no options at family level",
    )
    store.record_failure(
        query="bad query",
        facility="ALS",
        partial_selections={"system": "MAG"},
        reason="timeout at device level",
    )

    raw = json.loads(store._path.read_text())
    key = FeedbackStore._make_key("bad query", "ALS")
    assert len(raw["entries"][key]["failures"]) == 2


# ------------------------------------------------------------------
# 10. Empty store returns empty hints
# ------------------------------------------------------------------


def test_empty_store_returns_empty_hints(store):
    assert store.get_hints("anything", "ALS") == []


def test_empty_store_for_unknown_facility(store):
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=5
    )
    assert store.get_hints("magnets", "UNKNOWN_FACILITY") == []


def test_empty_store_for_unknown_query(store):
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=5
    )
    assert store.get_hints("totally different query", "ALS") == []


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_list_selections_are_flattened(store):
    """Single-element lists in selections should be flattened to scalars."""
    store.record_success(
        query="magnets",
        facility="ALS",
        selections={"system": ["MAG"], "device": ["QF1", "QF2"]},
        channel_count=3,
    )

    hints = store.get_hints("magnets", "ALS")
    assert hints[0]["selections"]["system"] == "MAG"  # flattened
    assert hints[0]["selections"]["device"] == ["QF1", "QF2"]  # kept as list


def test_make_key_is_deterministic():
    key1 = FeedbackStore._make_key("show me magnets", "ALS")
    key2 = FeedbackStore._make_key("show me magnets", "ALS")
    assert key1 == key2
    assert len(key1) == 64  # SHA-256 hex digest length


def test_make_key_differs_for_different_inputs():
    key_a = FeedbackStore._make_key("query a", "ALS")
    key_b = FeedbackStore._make_key("query b", "ALS")
    assert key_a != key_b


def test_store_created_in_nonexistent_subdirectory(tmp_path):
    """Store should create parent directories as needed."""
    deep_path = tmp_path / "a" / "b" / "c" / "feedback.json"
    store = FeedbackStore(deep_path)
    store.record_success(query="test", facility="ALS", selections={"x": "y"}, channel_count=1)

    assert deep_path.exists()
    hints = store.get_hints("test", "ALS")
    assert len(hints) == 1


# ------------------------------------------------------------------
# 12. list_keys returns metadata
# ------------------------------------------------------------------


def test_list_keys_returns_metadata(store):
    store.record_success(
        query="show me magnets", facility="ALS", selections={"system": "MAG"}, channel_count=10
    )
    store.record_failure(
        query="show me magnets",
        facility="ALS",
        partial_selections={"system": "X"},
        reason="bad",
    )

    keys = store.list_keys()
    assert len(keys) == 1
    entry = keys[0]
    assert entry["query"] == "show me magnets"
    assert entry["facility"] == "ALS"
    assert entry["success_count"] == 1
    assert entry["failure_count"] == 1
    assert entry["last_activity"] != ""
    assert len(entry["key"]) == 64


# ------------------------------------------------------------------
# 13. get_entry / get_entry missing
# ------------------------------------------------------------------


def test_get_entry_returns_bucket(store):
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=5
    )
    key = FeedbackStore._make_key("magnets", "ALS")
    entry = store.get_entry(key)
    assert entry is not None
    assert len(entry["successes"]) == 1
    assert entry["_meta"]["query"] == "magnets"


def test_get_entry_missing_returns_none(store):
    assert store.get_entry("nonexistent_key") is None


# ------------------------------------------------------------------
# 14. delete_entry
# ------------------------------------------------------------------


def test_delete_entry_removes_bucket(store):
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=5
    )
    key = FeedbackStore._make_key("magnets", "ALS")
    assert store.delete_entry(key) is True
    assert store.get_entry(key) is None


def test_delete_entry_missing_returns_false(store):
    assert store.delete_entry("nonexistent") is False


# ------------------------------------------------------------------
# 15. delete_record with stale check
# ------------------------------------------------------------------


def test_delete_record_with_matching_timestamp(store):
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=5
    )
    key = FeedbackStore._make_key("magnets", "ALS")
    entry = store.get_entry(key)
    ts = entry["successes"][0]["timestamp"]

    assert store.delete_record(key, "successes", 0, ts) is True
    assert store.get_entry(key) is None  # Empty bucket cleaned up


def test_delete_record_stale_timestamp_raises(store):
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=5
    )
    key = FeedbackStore._make_key("magnets", "ALS")

    with pytest.raises(ValueError, match="Stale timestamp"):
        store.delete_record(key, "successes", 0, "wrong-timestamp")


# ------------------------------------------------------------------
# 16. update_record
# ------------------------------------------------------------------


def test_update_record_success(store):
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=5
    )
    key = FeedbackStore._make_key("magnets", "ALS")
    entry = store.get_entry(key)
    ts = entry["successes"][0]["timestamp"]

    store.update_record(key, "successes", 0, ts, selections={"system": "QUAD"}, channel_count=99)

    updated = store.get_entry(key)
    assert updated["successes"][0]["selections"] == {"system": "QUAD"}
    assert updated["successes"][0]["channel_count"] == 99
    assert updated["successes"][0]["timestamp"] != ts  # Timestamp updated


def test_update_record_failure(store):
    store.record_failure(
        query="bad", facility="ALS", partial_selections={"system": "MAG"}, reason="no options"
    )
    key = FeedbackStore._make_key("bad", "ALS")
    entry = store.get_entry(key)
    ts = entry["failures"][0]["timestamp"]

    store.update_record(
        key, "failures", 0, ts, partial_selections={"system": "QUAD"}, reason="timeout"
    )

    updated = store.get_entry(key)
    assert updated["failures"][0]["partial_selections"] == {"system": "QUAD"}
    assert updated["failures"][0]["reason"] == "timeout"


def test_update_record_stale_timestamp_raises(store):
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=5
    )
    key = FeedbackStore._make_key("magnets", "ALS")

    with pytest.raises(ValueError, match="Stale timestamp"):
        store.update_record(key, "successes", 0, "wrong", selections={"system": "X"})


# ------------------------------------------------------------------
# 17. add_manual_entry
# ------------------------------------------------------------------


def test_add_manual_entry_success(store):
    key = store.add_manual_entry(
        query="magnets",
        facility="ALS",
        entry_type="success",
        selections={"system": "MAG"},
        channel_count=42,
    )
    hints = store.get_hints("magnets", "ALS")
    assert len(hints) == 1
    assert hints[0]["channel_count"] == 42
    assert len(key) == 64


def test_add_manual_entry_failure(store):
    key = store.add_manual_entry(
        query="bad",
        facility="ALS",
        entry_type="failure",
        selections={"system": "MAG"},
        reason="not found",
    )
    entry = store.get_entry(key)
    assert len(entry["failures"]) == 1
    assert entry["failures"][0]["reason"] == "not found"


def test_add_manual_entry_dedup(store):
    """Manual add of duplicate is silently ignored (existing dedup)."""
    store.add_manual_entry(
        query="magnets",
        facility="ALS",
        entry_type="success",
        selections={"system": "MAG"},
        channel_count=10,
    )
    store.add_manual_entry(
        query="magnets",
        facility="ALS",
        entry_type="success",
        selections={"system": "MAG"},
        channel_count=20,
    )
    hints = store.get_hints("magnets", "ALS")
    assert len(hints) == 1  # Dedup kept only the first


# ------------------------------------------------------------------
# 18. export_data
# ------------------------------------------------------------------


def test_export_data_returns_complete_store(store):
    store.record_success(
        query="magnets", facility="ALS", selections={"system": "MAG"}, channel_count=5
    )
    store.record_failure(query="bad", facility="ALS", partial_selections={}, reason="fail")

    exported = store.export_data()
    assert exported["version"] == 2
    assert len(exported["entries"]) == 2

    # Verify it's a deep copy
    exported["entries"].clear()
    assert len(store.export_data()["entries"]) == 2


# ------------------------------------------------------------------
# 19. v1 to v2 migration
# ------------------------------------------------------------------


def test_v1_to_v2_migration(tmp_path):
    """v1 format data should get _meta backfilled on load."""
    path = tmp_path / "feedback.json"
    v1_data = {
        "version": 1,
        "entries": {
            "abc123": {
                "successes": [
                    {"selections": {"system": "MAG"}, "channel_count": 5, "timestamp": "2026-01-01"}
                ],
                "failures": [],
            }
        },
    }
    path.write_text(json.dumps(v1_data))

    store = FeedbackStore(path)
    entry = store.get_entry("abc123")
    assert entry is not None
    assert entry["_meta"] == {"query": "(unknown)", "facility": "(unknown)"}

    # Version should now be 2 in memory
    exported = store.export_data()
    assert exported["version"] == 2


# ------------------------------------------------------------------
# 20. _meta populated on new records
# ------------------------------------------------------------------


def test_meta_populated_on_new_records(store):
    """_meta is set after record_success() and record_failure()."""
    store.record_success(
        query="Show Me Magnets", facility="ALS", selections={"system": "MAG"}, channel_count=5
    )
    key = FeedbackStore._make_key("Show Me Magnets", "ALS")
    entry = store.get_entry(key)
    assert entry["_meta"]["query"] == "Show Me Magnets"
    assert entry["_meta"]["facility"] == "ALS"

    store.record_failure(query="bad query", facility="LCLS", partial_selections={}, reason="fail")
    key2 = FeedbackStore._make_key("bad query", "LCLS")
    entry2 = store.get_entry(key2)
    assert entry2["_meta"]["query"] == "bad query"
    assert entry2["_meta"]["facility"] == "LCLS"


# ------------------------------------------------------------------
# 21. search_by_keywords
# ------------------------------------------------------------------


def test_search_by_keywords_single_match(store):
    """A keyword like 'magnet' matches a query containing 'magnets' (after normalization)."""
    store.record_success(
        query="show me magnets", facility="ALS", selections={"system": "MAG"}, channel_count=42
    )
    store.record_success(
        query="find BPM positions", facility="ALS", selections={"system": "BPM"}, channel_count=10
    )

    results = store.search_by_keywords(["magnets"])
    assert len(results) == 1
    assert results[0]["query"] == "show me magnets"
    assert results[0]["score"] == 1
    assert len(results[0]["successes"]) == 1


def test_search_by_keywords_multi_overlap_ranked(store):
    """More keyword overlap scores higher."""
    store.record_success(
        query="horizontal corrector magnets",
        facility="ALS",
        selections={"system": "HCM"},
        channel_count=20,
    )
    store.record_success(
        query="show me magnets",
        facility="ALS",
        selections={"system": "MAG"},
        channel_count=42,
    )

    results = store.search_by_keywords(["corrector", "magnets"])
    assert len(results) == 2
    # "horizontal corrector magnets" has 2 overlapping tokens, should be first
    assert results[0]["query"] == "horizontal corrector magnets"
    assert results[0]["score"] == 2
    # "show me magnets" has 1 overlapping token
    assert results[1]["query"] == "show me magnets"
    assert results[1]["score"] == 1


def test_search_by_keywords_no_match(store):
    """Unrelated keywords return empty."""
    store.record_success(
        query="show me magnets", facility="ALS", selections={"system": "MAG"}, channel_count=42
    )

    results = store.search_by_keywords(["temperature", "cryogenics"])
    assert results == []


def test_search_by_keywords_max_results(store):
    """Respects max_results cap."""
    for i in range(10):
        store.record_success(
            query=f"query with magnets variant {i}",
            facility="ALS",
            selections={"system": f"SYS_{i}"},
            channel_count=i,
        )

    results = store.search_by_keywords(["magnets"], max_results=3)
    assert len(results) == 3
