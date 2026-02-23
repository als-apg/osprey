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
        "Show Me Magnets",        # case
        "  show  me   magnets  ", # whitespace
        "show me magnets!",       # punctuation
        "SHOW ME MAGNETS???",     # case + punctuation
        "show, me: magnets.",     # mixed punctuation
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
    assert raw["version"] == 1
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
    store.record_failure(
        query="bpms", facility="ALS", partial_selections={}, reason="not found"
    )

    store.clear()

    assert store.get_hints("magnets", "ALS") == []
    assert store.get_hints("bpms", "ALS") == []

    # File should still exist but be empty of entries
    raw = json.loads(store._path.read_text())
    assert raw["entries"] == {}
    assert raw["version"] == 1


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
    store.record_success(
        query="test", facility="ALS", selections={"x": "y"}, channel_count=1
    )

    assert deep_path.exists()
    hints = store.get_hints("test", "ALS")
    assert len(hints) == 1
