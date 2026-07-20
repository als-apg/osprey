"""Process-lifetime failure counters and boot nonce on the worker /health payload.

Verifies the counters are driven by _stamp (via the counter-hook seam), are
exposed on /health as provider_errors / infrastructure_errors, stay monotonic
and independent of the evictable _runs map, and that the boot nonce is stable
within a process. Existing /health fields are unchanged.
"""

from __future__ import annotations

import pytest

from osprey.mcp_server.dispatch_worker import counters, dispatch_api, failure_class


@pytest.fixture(autouse=True)
def _wired_counters():
    """Install the counter hook and isolate counter + run state per test."""
    counters.reset()
    counters.install()  # register_counter_hook(increment)
    dispatch_api._runs.clear()
    yield
    failure_class.register_counter_hook(None)
    counters.reset()
    dispatch_api._runs.clear()


def _stamp(cls: str, n: int) -> None:
    """Stamp ``n`` error results of class ``cls`` (each bumps the lifetime counter)."""
    for _ in range(n):
        failure_class._stamp({"status": "error"}, cls, None)


# ---------------------------------------------------------------------------
# counters module
# ---------------------------------------------------------------------------


def test_increment_is_wired_to_stamp():
    _stamp(failure_class.FAILURE_PROVIDER, 2)
    _stamp(failure_class.FAILURE_INFRASTRUCTURE, 1)
    _stamp(failure_class.FAILURE_RUN, 4)
    assert counters.get_counts() == {
        failure_class.FAILURE_PROVIDER: 2,
        failure_class.FAILURE_INFRASTRUCTURE: 1,
        failure_class.FAILURE_RUN: 4,
    }


def test_get_counts_returns_a_copy():
    snapshot = counters.get_counts()
    snapshot[failure_class.FAILURE_PROVIDER] = 999
    assert counters.get_counts()[failure_class.FAILURE_PROVIDER] == 0


def test_counters_are_monotonic():
    _stamp(failure_class.FAILURE_PROVIDER, 1)
    first = counters.get_counts()[failure_class.FAILURE_PROVIDER]
    _stamp(failure_class.FAILURE_PROVIDER, 1)
    second = counters.get_counts()[failure_class.FAILURE_PROVIDER]
    assert second == first + 1


def test_unknown_class_is_still_counted():
    # Defensive path: increment() must not drop an unexpected class.
    counters.increment("mystery")
    assert counters.get_counts()["mystery"] == 1


# ---------------------------------------------------------------------------
# /health payload
# ---------------------------------------------------------------------------


async def test_health_exposes_lifetime_counters():
    _stamp(failure_class.FAILURE_PROVIDER, 3)
    _stamp(failure_class.FAILURE_INFRASTRUCTURE, 2)
    payload = await dispatch_api.health()
    assert payload["provider_errors"] == 3
    assert payload["infrastructure_errors"] == 2


async def test_health_preserves_existing_fields():
    dispatch_api._runs["a"] = {"status": "completed"}
    dispatch_api._runs["b"] = {"status": "error"}
    dispatch_api._runs["c"] = {"status": "pending"}
    payload = await dispatch_api.health()
    assert payload["status"] == "ok"
    assert payload["completed_runs"] == 1
    assert payload["error_runs"] == 1
    assert payload["pending_runs"] == 1
    assert payload["total_runs"] == 3


async def test_counters_survive_runs_eviction():
    """The lifetime counters must NOT be derived from the evictable _runs map."""
    _stamp(failure_class.FAILURE_PROVIDER, 5)
    _stamp(failure_class.FAILURE_INFRASTRUCTURE, 4)

    before = await dispatch_api.health()
    assert (before["provider_errors"], before["infrastructure_errors"]) == (5, 4)

    # Simulate eviction / lexical persisted-run reload wiping the run store.
    dispatch_api._runs.clear()

    after = await dispatch_api.health()
    assert (after["provider_errors"], after["infrastructure_errors"]) == (5, 4)
    # The _runs-derived counts collapse, but the lifetime counters do not.
    assert after["error_runs"] == 0


async def test_boot_nonce_is_present_and_stable_within_process():
    first = await dispatch_api.health()
    second = await dispatch_api.health()
    assert first["boot_nonce"] == dispatch_api._BOOT_NONCE
    assert first["boot_nonce"] == second["boot_nonce"]
