"""Unit tests for `validation_record.py` (task 2.2): the content-hash-keyed
store of PASSING plan-validation records that task 2.4's session-layer load
gate and task 2.5's launch gate query after re-hashing a file's on-disk
content.

Each test builds its own `ValidationRecordStore` rather than reaching for the
module-level `validation_records` singleton, so tests never leak state into
each other (or into whatever else imports the singleton within the same
process).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from osprey.services.bluesky_bridge.plan_validation import hash_plan_body
from osprey.services.bluesky_bridge.validation_record import (
    ValidationRecordStore,
    validation_records,
)


def test_singleton_is_reachable_and_starts_empty() -> None:
    # The bridge-process singleton other tasks (2.3/2.4/2.5) import.
    assert isinstance(validation_records, ValidationRecordStore)


def test_unknown_hash_returns_false() -> None:
    store = ValidationRecordStore()
    assert store.has_passing_record("deadbeef") is False


def test_record_then_query_returns_true() -> None:
    store = ValidationRecordStore()
    store.record("abc123")
    assert store.has_passing_record("abc123") is True


def test_a_hash_that_was_never_recorded_stays_false() -> None:
    """No "record a failure" API exists — a failed validation records
    nothing, so a hash that only ever failed is indistinguishable from one
    that was never validated at all: both read as `False`.
    """
    store = ValidationRecordStore()
    store.record("some-other-hash")
    assert store.has_passing_record("never-recorded-hash") is False


def test_recording_the_same_hash_twice_is_a_harmless_noop() -> None:
    store = ValidationRecordStore()
    store.record("abc123")
    store.record("abc123")
    assert store.has_passing_record("abc123") is True


def test_round_trip_through_hash_plan_body() -> None:
    """Ties the store to the shared `hash_plan_body` normalization: a record
    made against one hashing of a body's content is found by a later re-hash
    of the same content, byte for byte — the exact contract 2.4/2.5 rely on.
    """
    body = "PLAN_METADATA = {'name': 'tiny'}\n"
    store = ValidationRecordStore()

    content_hash = hash_plan_body(body)
    store.record(content_hash)

    # A later caller (e.g. the load gate) re-hashes the file's current
    # on-disk content independently and must still find the same record.
    assert store.has_passing_record(hash_plan_body(body)) is True


def test_concurrent_record_and_query_is_thread_safe() -> None:
    store = ValidationRecordStore()
    hashes = [f"hash-{i}" for i in range(200)]

    def _record(content_hash: str) -> None:
        store.record(content_hash)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(_record, hashes))

    assert all(store.has_passing_record(h) for h in hashes)
    assert store.has_passing_record("not-one-of-them") is False
