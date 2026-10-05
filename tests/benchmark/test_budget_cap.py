"""The per-query dollar cap the e2e harness hands the Agent SDK."""

import math

import pytest
from tests.e2e.sdk_helpers import e2e_budget_cap, e2e_budget_scale


@pytest.mark.parametrize(
    ("raw", "base", "cap"),
    [(None, 2.0, 2.0), ("3", 2.0, 6.0), ("0", 2.0, 2.0), ("junk", 2.0, 2.0)],
)
def test_cap_scales_the_call_site_budget(monkeypatch, raw, base, cap):
    if raw is None:
        monkeypatch.delenv("OSPREY_E2E_BUDGET_SCALE", raising=False)
    else:
        monkeypatch.setenv("OSPREY_E2E_BUDGET_SCALE", raw)
    assert e2e_budget_cap(base) == cap


def test_an_infinite_scale_sends_no_cap_and_passes_every_cost_ceiling(monkeypatch):
    monkeypatch.setenv("OSPREY_E2E_BUDGET_SCALE", "inf")
    assert e2e_budget_cap(2.0) is None
    assert math.isinf(e2e_budget_scale())
    assert 1e6 < 0.5 * e2e_budget_scale()
