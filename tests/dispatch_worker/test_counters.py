"""Tests for the dispatch-worker process-lifetime failure counters.

Contracts:
  - counters seed one key per known failure class so readers never KeyError;
  - ``increment`` is monotonic and tolerates an unrecognised class;
  - ``install`` wires ``increment`` into the failure_class stamp seam so a real
    stamp bumps the matching lifetime counter;
  - ``get_counts`` returns a snapshot copy (mutating it can't corrupt state).
"""

from __future__ import annotations

import pytest

from osprey.mcp_server.dispatch_worker import counters
from osprey.mcp_server.dispatch_worker import failure_class as fc


@pytest.fixture(autouse=True)
def _isolate_counters():
    """Zero counters and detach the stamp hook around every test."""
    counters.reset()
    yield
    fc.register_counter_hook(None)
    counters.reset()


@pytest.mark.unit
def test_counts_seeded_for_every_class():
    """A fresh snapshot has a zero entry for each known class."""
    counts = counters.get_counts()
    assert set(counts) >= fc.FAILURE_CLASSES
    assert all(counts[cls] == 0 for cls in fc.FAILURE_CLASSES)


@pytest.mark.unit
def test_increment_is_monotonic():
    counters.increment(fc.FAILURE_PROVIDER)
    counters.increment(fc.FAILURE_PROVIDER)
    counters.increment(fc.FAILURE_RUN)
    counts = counters.get_counts()
    assert counts[fc.FAILURE_PROVIDER] == 2
    assert counts[fc.FAILURE_RUN] == 1
    assert counts[fc.FAILURE_INFRASTRUCTURE] == 0


@pytest.mark.unit
def test_increment_tolerates_unknown_class():
    """An unexpected class is counted under its own key, never dropped."""
    counters.increment("mystery")
    assert counters.get_counts()["mystery"] == 1


@pytest.mark.unit
def test_get_counts_returns_a_copy():
    """Mutating the returned snapshot must not corrupt the live counters."""
    snap = counters.get_counts()
    snap[fc.FAILURE_PROVIDER] = 999
    assert counters.get_counts()[fc.FAILURE_PROVIDER] == 0


@pytest.mark.unit
def test_install_wires_stamp_to_counters():
    """After install(), a real _stamp bumps the lifetime counter for its class."""
    counters.install()
    fc._stamp({}, fc.FAILURE_PROVIDER, num_tool_calls=1)
    fc._stamp({}, fc.FAILURE_INFRASTRUCTURE, num_tool_calls=0)
    counts = counters.get_counts()
    assert counts[fc.FAILURE_PROVIDER] == 1
    assert counts[fc.FAILURE_INFRASTRUCTURE] == 1


@pytest.mark.unit
def test_reset_zeros_all():
    counters.increment(fc.FAILURE_RUN)
    counters.reset()
    assert all(v == 0 for v in counters.get_counts().values())
