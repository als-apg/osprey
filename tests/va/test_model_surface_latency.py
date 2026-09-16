"""Cost guard: declaring the optics arrays costs a setpoint write nothing.

``PyATRingModel`` declares three read-only ND optics arrays (the tunes, and
beta and the true orbit at the BPMs) that the model computes on demand and
memoises per solve. The whole design rests on one claim: a routed setpoint
write -- the hot path, one per client put -- never pays for them. This module
holds that claim to account structurally, never by the clock.

Two assertions carry it, both over a matched pair of models -- one declaring
the arrays, one built without them -- driven through identical write
sequences:

- **No optics entry point is reached.** ``at.get_optics`` is the single optics
  compute in the VA source (``model/pyat.py``, inside
  ``PyATRingModel._optics``). Twenty writes call neither it, nor the per-solve
  memo in front of it, nor an optics variable's own ``_get``.
- **The write does the same work either way.** The Python call profile of one
  routed write is compared frame by frame between the two models, and has to
  match exactly bar the one type test per declared array that keeps the arrays
  out of the readout. So the declarations add no call to the write path, and
  make no call already on it happen more often.

The second assertion replaced a wall-clock comparison of the two models'
median write time against a millisecond budget. That measured the host as
much as it measured the write: on a loaded shared CI runner both medians
tripled, unequally, and the comparison failed while the property it guards
held perfectly. A call profile counts the same quantity the budget was a
proxy for -- work per write -- instead of timing it, so a busy host cannot
move it.

No markers and no ``importorskip``: like ``test_pyat_ring_model.py``, this
module imports ``at`` at module scope because accelerator-toolbox is a hard
dependency of the package, and ``--strict-markers`` admits no marker that
would gate it.
"""

from __future__ import annotations

import gc
import sys
from collections import Counter
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
# in the same order, so they solve identical lattices -- which is also what
# makes their call profiles comparable, since the closed-orbit solver iterates
# from the previous solution and a different kick can cost it another pass.
CURRENTS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)

WRITES = 20
WARMUP_WRITES = 5

# Frames with this name are `isinstance` against an abstract base class. The
# only trace the declarations leave on a write is one such test per declared
# array, where `_read_outputs` filters the arrays out of the per-BPM readings.
# The bound is "no more than one per array", not "exactly one per array", so
# an interpreter that answers such a test without a Python frame still holds.
TYPE_CHECK = "__instancecheck__"


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
    """A bridge serving `model` through `records`, warmed and ready to measure.

    Bound to the whole pyat-coupled partition, as `entrypoint` binds it, so a
    measured write pushes into all 144 BPM readback records exactly as a
    served write does. Seeded RNG: the reading noise must not vary between the
    two models, or the comparison is measuring the draws. The warm-up writes
    settle everything a first write pays once -- lazy imports, caches, the
    first closed-orbit solution the next solve starts from.
    """
    bridge = PhysicsBridge(model=model, rng_seed=20250909)
    bridge.bind(
        records,
        physics_setpoints=pyat_coupled_setpoint_addresses(build_manifest()["channels"]),
    )
    for index in range(WARMUP_WRITES):
        bridge.on_setpoint(A_CORRECTOR, CURRENTS[index % len(CURRENTS)])
    return bridge


def _write_call_profile(bridge: PhysicsBridge, value: float) -> Counter[tuple[str, str]]:
    """Every Python call one routed setpoint write makes, counted per function.

    Keyed by code object -- (file, function name) -- which is finer than the
    module boundary and needs no list of which packages count as work: a call
    the write makes is a call the write pays for, wherever it lands.

    Garbage collection is off for the measurement and restored afterwards, so
    a collection that happened to fall inside one of the two writes cannot
    contribute a finalizer's frames to it. Any profile function already
    installed is restored rather than dropped.
    """
    counts: Counter[tuple[str, str]] = Counter()

    def record(frame: Any, event: str, _arg: Any) -> None:
        if event == "call":
            code = frame.f_code
            counts[(code.co_filename, code.co_name)] += 1

    previous = sys.getprofile()
    gc.collect()
    gc.disable()
    sys.setprofile(record)
    try:
        bridge.on_setpoint(A_CORRECTOR, value)
    finally:
        sys.setprofile(previous)
        gc.enable()
    return counts


def _work(profile: Counter[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """The profile without its type tests -- the calls that do the write."""
    return {frame: count for frame, count in profile.items() if frame[1] != TYPE_CHECK}


def _type_checks(profile: Counter[tuple[str, str]]) -> int:
    """How many abstract-base-class type tests the write made."""
    return sum(count for frame, count in profile.items() if frame[1] == TYPE_CHECK)


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
    Every test in this module writes the same values to both bridges in the
    same order, so the two lattices stay in step whatever order the tests run
    in, and a write means the same amount of physics on each.
    """
    return (
        _bound_bridge(PyATRingModel(), records),
        _bound_bridge(RingModelWithoutOpticsVariables(), records),
    )


@pytest.fixture
def optics_calls(monkeypatch) -> list[str]:
    """Record every call to an optics entry point, passing each one through.

    All three of them: the compute itself, the per-solve memo that fronts it
    (so a write that reaches a *warm* memo, costing nothing, is still caught
    as a write that asked for the optics), and the read of an optics variable.
    """
    calls: list[str] = []

    def spy(owner: Any, attribute: str, label: str) -> None:
        real = getattr(owner, attribute)

        def recording(*args: Any, **kwargs: Any) -> Any:
            calls.append(label)
            return real(*args, **kwargs)

        monkeypatch.setattr(owner, attribute, recording)

    spy(at, "get_optics", "at.get_optics")
    spy(PyATRingModel, "_optics", "PyATRingModel._optics")
    spy(PyATReadOnlyNDVariable, "_get", "PyATReadOnlyNDVariable._get")
    return calls


class TestSetpointLatency:
    """A routed setpoint write does the same work, and so costs the same,
    whether or not the model declares the read-only optics arrays."""

    def test_the_pair_differs_in_the_optics_declarations_and_nothing_else(self, bridges):
        """Without this the comparison could be a model against itself.

        `RingModelWithoutOpticsVariables` patches a private module attribute;
        if that factory were renamed the patch would silently do nothing and
        both halves of the comparison would measure the same model.
        """
        declared, skipped = (bridge._model.supported_variables for bridge in bridges)

        assert OPTICS_NAMES <= set(declared)
        assert all(isinstance(declared[name], PyATReadOnlyNDVariable) for name in OPTICS_NAMES)
        assert OPTICS_NAMES.isdisjoint(skipped)
        assert set(declared) - OPTICS_NAMES == set(skipped)

    def test_a_routed_setpoint_write_never_computes_the_optics(self, bridges, optics_calls):
        """The optics are off the write path entirely, on both models.

        `PhysicsBridge.on_setpoint` is what a routed put reaches --
        `SetpointRoutedModel._set` calls it once per routed address -- and the
        three spied entry points are the whole of how a value of one of these
        arrays can be arrived at. Twenty writes must reach none of them,
        whatever the model declares.
        """
        for index in range(WRITES):
            value = CURRENTS[index % len(CURRENTS)]
            for bridge in bridges:
                bridge.on_setpoint(A_CORRECTOR, value)

        assert optics_calls == []

    def test_declaring_the_optics_does_not_slow_a_setpoint_write(self, bridges):
        """The two models make the very same calls to serve the very same write.

        Compared per write rather than in bulk, because the claim is about one
        write: a cost the declarations added would show up as a function the
        declaring model calls and the control does not, or calls more often.
        Every current in the cycle is measured, so a difference that only
        appears at one working point cannot hide behind another.

        The only admitted difference is the type test per declared array in
        `PyATRingModel._read_outputs`, which is what keeps the arrays out of
        the per-BPM readings. That is the price of declaring them, it is
        bounded by the number of declarations, and it is paid once per write
        rather than per BPM or per variable.
        """
        declared, skipped = bridges

        for value in CURRENTS:
            with_optics = _write_call_profile(declared, value)
            without_optics = _write_call_profile(skipped, value)

            assert any(name == "on_setpoint" for _file, name in with_optics), (
                "no write was profiled -- the measurement, not the write path, is broken"
            )

            done, control = _work(with_optics), _work(without_optics)
            difference = {
                frame: (done.get(frame, 0), control.get(frame, 0))
                for frame in done.keys() | control.keys()
                if done.get(frame, 0) != control.get(frame, 0)
            }
            assert difference == {}, (
                f"a write of {value} A does different work when the optics are declared; "
                f"(declared, control) call counts: {difference}"
            )

            extra = _type_checks(with_optics) - _type_checks(without_optics)
            assert 0 <= extra <= len(OPTICS_NAMES), (
                f"a write of {value} A made {extra} more type tests with the optics "
                f"declared; at most one per declared array is expected"
            )
