"""Latency guard: declaring the optics arrays costs a setpoint write nothing.

``PyATRingModel`` declares three read-only ND optics arrays (the tunes, and
beta and the true orbit at the BPMs) that the model computes on demand and
memoises per solve. The whole design rests on one claim: a routed setpoint
write -- the hot path, one per client put -- never pays for them. This module
holds that claim to account two ways.

The deterministic half is the spy: ``at.get_optics`` is the single optics
compute in the VA source (``model/pyat.py``, inside ``PyATRingModel._optics``),
and twenty writes must not call it once. That assertion carries the real
weight, because it cannot be swayed by a busy host.

The wall-clock half compares the typical per-write cost of a model that
declares the three arrays against one built without them. It is a
corroboration, not the primary evidence, and it is written to be robust
rather than tight: see ``TestSetpointLatency`` for what the budget is and why
it is that number.

No markers and no ``importorskip``: like ``test_pyat_ring_model.py``, this
module imports ``at`` at module scope because accelerator-toolbox is a hard
dependency of the package, and ``--strict-markers`` admits no marker that
would gate it.
"""

from __future__ import annotations

import gc
import statistics
import time
from typing import Any

import at
import pytest

from osprey.services.virtual_accelerator.ioc.physics_bridge import PhysicsBridge
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    build_manifest,
    pyat_coupled_setpoint_addresses,
)
from osprey.services.virtual_accelerator.model import pyat as pyat_module
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel
from osprey.services.virtual_accelerator.model.variables import PyATReadOnlyNDVariable

# The three read-only arrays whose cost this module denies.
OPTICS_NAMES = frozenset({"tunes", "beta_at_bpms", "orbit_at_bpms"})

# The routed setpoint every write goes through. One corrector, so every write
# does the same work: one calibration read, one solve, one BPM truth read, one
# fault read, and 72 `bpm_read` draws.
A_CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"

# Corrector currents in Amps, cycled over the writes. Small kicks well inside
# the ring's stable range, and a cycle rather than a ramp so a long run never
# drifts away from the closed orbit. Both models are handed the same sequence
# in the same order, so they solve identical lattices.
CURRENTS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)

WRITES = 20
WARMUP_WRITES = 5

# The two medians may differ by less than this, in milliseconds. The
# regression it guards is a write that computes the optics: `at.get_optics`
# over the 72 monitors costs ~12 ms on this ring, which is as much again as an
# entire write (~11 ms), so a write that paid for the optics would miss this
# budget by six times over. The budget is therefore loose against the defect
# and tight against nothing.
BUDGET_MS = 2.0

# Rounds of `WRITES` writes per model; the closest round has to clear the
# budget. A round is still a wall-clock measurement on a shared host, and a
# host that stalls for a whole round skews both of its medians unequally.
# Retrying is what keeps such a stall from reading as a regression: a real
# regression shifts *every* round by the optics cost, so no round would ever
# clear the budget.
ROUNDS = 3


class FakeRecord:
    """Minimal duck-typed stand-in for a softioc record: `.set()` and `.get()`.

    `value` is what the record currently holds. Duplicated here rather than
    shared, as in every other module in this suite -- tests/va/conftest.py
    carries no fixtures.
    """

    def __init__(self, value: float | None = None) -> None:
        self.value: float | None = value

    def set(self, value: float) -> None:
        self.value = value

    def get(self) -> float | None:
        return self.value


class RingModelWithoutOpticsVariables(PyATRingModel):
    """The control: the same ring model, declaring no optics arrays.

    `_optics_variables()` is a module-level factory called inline in
    `PyATRingModel.__init__`, so a subclass cannot override it -- replacing
    the module attribute for the duration of the construction is the only way
    to build this model without reimplementing the whole constructor, and is
    the same idiom `test_pyat_ring_model.py`'s `refusal_ring` uses to swap
    `build_ring`. The attribute is restored in a `finally`, so a failed
    construction cannot leave the module patched for the rest of the session.

    Everything else -- the 348 setpoints, the 144 BPM outputs, the 1,344 fault
    variables -- is identical, which is what makes the pair a controlled
    comparison rather than two different models.
    """

    def __init__(self) -> None:
        declared = pyat_module._optics_variables
        pyat_module._optics_variables = list
        try:
            super().__init__()
        finally:
            pyat_module._optics_variables = declared


def _pyat_coupled_addresses() -> list[str]:
    """Every pyat-coupled address, straight from the manifest."""
    return [
        channel["address"]
        for channel in build_manifest()["channels"]
        if channel["partition"] == PARTITION_PYAT_COUPLED
    ]


def _bound_bridge(model: PyATRingModel, records: dict[str, Any]) -> PhysicsBridge:
    """A bridge serving `model` through `records`, warmed and ready to time.

    Bound to the whole pyat-coupled partition, as `entrypoint` binds it, so a
    timed write pushes into all 144 BPM readback records exactly as a served
    write does. Seeded RNG: the reading noise must not vary between the two
    models, or the comparison is measuring the draws.
    """
    bridge = PhysicsBridge(model=model, rng_seed=20250909)
    bridge.bind(
        records,
        physics_setpoints=pyat_coupled_setpoint_addresses(build_manifest()["channels"]),
    )
    for index in range(WARMUP_WRITES):
        bridge.on_setpoint(A_CORRECTOR, CURRENTS[index % len(CURRENTS)])
    return bridge


def _timed_write(bridge: PhysicsBridge, value: float) -> float:
    """One routed setpoint write, in milliseconds."""
    start = time.perf_counter()
    bridge.on_setpoint(A_CORRECTOR, value)
    return (time.perf_counter() - start) * 1000.0


def _round_medians(declared: PhysicsBridge, skipped: PhysicsBridge) -> tuple[float, float]:
    """Median ms per write over `WRITES` writes on each bridge.

    The median, not the mean: samples range 8-40 ms around a median of 10, and
    one descheduled sample moves a 20-sample mean by more than the budget on
    its own, in whichever series the host happened to look away from. A
    per-write cost that really grew moves every sample, and so moves the
    median with them.

    The two are interleaved write by write so that a host that slows down
    partway through the round slows both series, rather than whichever one
    happened to be measured second. Garbage collection is off for the round
    and restored afterwards: a collection pause lands in one series only and
    is the one nondeterministic cost that can be taken out of the measurement
    outright.
    """
    declared_samples: list[float] = []
    skipped_samples: list[float] = []
    gc.collect()
    gc.disable()
    try:
        for index in range(WRITES):
            value = CURRENTS[index % len(CURRENTS)]
            declared_samples.append(_timed_write(declared, value))
            skipped_samples.append(_timed_write(skipped, value))
    finally:
        gc.enable()
    return statistics.median(declared_samples), statistics.median(skipped_samples)


@pytest.fixture(scope="module")
def records() -> dict[str, Any]:
    """One fake record per pyat-coupled address, shared by both bridges.

    Module-scoped and shared on purpose: the bridges only push readings into
    these, and both must be handed the same number of records or one would be
    doing less work than the other.
    """
    return {address: FakeRecord(0.0) for address in _pyat_coupled_addresses()}


@pytest.fixture(scope="module")
def bridges(records) -> tuple[PhysicsBridge, PhysicsBridge]:
    """The pair under comparison: optics declared, and optics skipped.

    Module-scoped -- building the ring twice is the expensive part, and a
    setpoint write leaves nothing behind that the next test cares about.
    """
    return (
        _bound_bridge(PyATRingModel(), records),
        _bound_bridge(RingModelWithoutOpticsVariables(), records),
    )


@pytest.fixture
def optics_calls(monkeypatch) -> list[tuple]:
    """Record every ``at.get_optics`` call, passing each one through."""
    calls: list[tuple] = []
    real = at.get_optics

    def recording(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(at, "get_optics", recording)
    return calls


class TestSetpointLatency:
    """Twenty routed setpoint writes cost the same whether or not the model
    declares the read-only optics arrays, and never compute them."""

    def test_the_pair_differs_in_the_optics_declarations_and_nothing_else(self, bridges):
        """Without this the comparison could be a model against itself.

        `RingModelWithoutOpticsVariables` patches a private module attribute;
        if that factory were renamed the patch would silently do nothing and
        both halves of the timing test would measure the same model.
        """
        declared, skipped = (bridge._model.supported_variables for bridge in bridges)

        assert OPTICS_NAMES <= set(declared)
        assert all(isinstance(declared[name], PyATReadOnlyNDVariable) for name in OPTICS_NAMES)
        assert OPTICS_NAMES.isdisjoint(skipped)
        assert set(declared) - OPTICS_NAMES == set(skipped)

    def test_a_routed_setpoint_write_never_computes_the_optics(self, bridges, optics_calls):
        """The deterministic half of the claim, on both models.

        `PhysicsBridge.on_setpoint` is what a routed put reaches --
        `SetpointRoutedModel._set` calls it once per routed address -- and
        `at.get_optics` is the one optics compute in the VA source. Twenty
        writes must not reach it, whatever the model declares.
        """
        for index in range(WRITES):
            value = CURRENTS[index % len(CURRENTS)]
            for bridge in bridges:
                bridge.on_setpoint(A_CORRECTOR, value)

        assert optics_calls == []

    def test_declaring_the_optics_does_not_slow_a_setpoint_write(self, bridges):
        """The wall-clock corroboration: the two medians agree within the budget.

        Reported as the median over the twenty writes rather than the slowest
        one or their mean, because the slowest samples on a shared host
        measure the scheduler, not the write path.
        """
        declared, skipped = bridges
        rounds = [_round_medians(declared, skipped) for _ in range(ROUNDS)]

        closest = min(abs(with_optics - without) for with_optics, without in rounds)
        assert closest < BUDGET_MS, rounds
