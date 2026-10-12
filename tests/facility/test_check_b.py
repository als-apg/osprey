"""Check B's comparison: a served measurement against pyAT's exact physics at the same steps.

Check B holds a ``pyaml_measure`` measurement taken on the virtual
accelerator to pyAT's exact physics on the same lattice, at the same
operating point, taken at the tool's own steps
(``tests/facility/_model_reference.py``). The served lane
(``tests/va/e2e/test_check_b_served.py``) gets the exact side by running the
same measurement, with the same arguments, on a replica of the served machine
in its own process: the composite over the view the container serves, driven
through :class:`ReplicaRuntime`. Both sides take the same steps through the
same view, so the finite-step error is on both and the only difference left
is numerical.

This module keeps what is pure about that comparison -- the bands, the
quantity each kind is compared on, the comparison and the replica's runtime
-- and holds each to recorded inputs. No virtual accelerator boots here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from tests.facility._model_reference import ORBIT_ATOL_M

#: The precision of a published tune against pyAT's on the replica. Both are
#: tunes of one lattice setting from two builds of pyAT's integrators, the
#: container's and this host's; each rests on a closed orbit solved to the
#: Newton threshold, and off momentum their rounding differs by a few 1e-11.
TUNE_ATOL = 1e-10

#: How many readings a response entry differences: the reading at the step and
#: the one before it.
ARMS = 2.0

#: The kinds whose saved measurement is a response matrix.
MATRIX_KINDS = ("orm", "dispersion", "trm", "crm")

#: Every kind check B covers.
KINDS = (*MATRIX_KINDS, "chromaticity_monitor")


def _smallest_step(result: Mapping[str, Any]) -> float:
    steps = [abs(float(step)) for step in (result.get("deltas") or {}).values()]
    assert steps and min(steps) > 0.0, f"the result states no step: {result.get('deltas')}"
    return min(steps)


def band(kind: str, result: Mapping[str, Any]) -> float:
    """The absolute band one entry of ``kind``'s measured quantity is held to.

    Each is a reading's numerical floor over the step it differences across:
    a closed orbit is good to :data:`ORBIT_ATOL_M`, a tune to
    :data:`TUNE_ATOL`, and a chromaticity is a tune difference over the
    momentum step of the monitor (``e_delta_used``).

    Args:
        kind: The measurement kind.
        result: The tool's answer for the served measurement: its steps.
    """
    if kind not in KINDS:
        raise ValueError(f"check B covers no kind {kind!r}")
    if kind == "orm":
        return ARMS * ORBIT_ATOL_M / _smallest_step(result)
    if kind == "dispersion":
        return ARMS * ORBIT_ATOL_M / abs(float(result["rf_span_hz"]))
    if kind == "trm":
        return ARMS * TUNE_ATOL / _smallest_step(result)
    chromaticity = ARMS * TUNE_ATOL / abs(float(result["e_delta_used"]))
    if kind == "crm":
        return ARMS * chromaticity / _smallest_step(result)
    return chromaticity


def quantity(kind: str, document: Mapping[str, Any], result: Mapping[str, Any]) -> np.ndarray:
    """What check B compares for ``kind``: the saved matrix, or the chromaticity answered.

    Args:
        kind: The measurement kind.
        document: The measurement file the tool saved.
        result: The tool's answer.
    """
    if kind in MATRIX_KINDS:
        return np.asarray(document["matrix"], dtype=float)
    if kind == "chromaticity_monitor":
        return np.asarray(result["chromaticity"], dtype=float)
    raise ValueError(f"check B covers no kind {kind!r}")


def labels(document: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """The names a saved matrix gives its rows and columns."""
    return (
        [str(name) for name in document.get("observable_names") or ()],
        [str(name) for name in document.get("variable_names") or ()],
    )


def disagreements(
    measured: np.ndarray, exact: np.ndarray, atol: float, rows: Sequence[str] = ()
) -> list[str]:
    """One line per entry of ``measured`` farther than ``atol`` from ``exact``; empty if none.

    A shape difference, or an entry that is not finite on either side, is a
    disagreement however small the band.
    """
    measured = np.atleast_2d(np.asarray(measured, dtype=float))
    exact = np.atleast_2d(np.asarray(exact, dtype=float))
    if measured.shape != exact.shape:
        return [f"shape {measured.shape} against {exact.shape}"]
    error = np.abs(measured - exact)
    bad = np.argwhere(~(error <= atol))
    return [
        f"{rows[i] if i < len(rows) else i}[{j}]: {measured[i, j]!r} against {exact[i, j]!r}"
        for i, j in bad
    ]


def assert_meets(
    kind: str,
    measured: tuple[Mapping[str, Any], Mapping[str, Any]],
    exact: tuple[Mapping[str, Any], Mapping[str, Any]],
) -> None:
    """Fail unless the served measurement meets the replica's within ``kind``'s band.

    Args:
        kind: The measurement kind.
        measured: The served measurement's saved file and the tool's answer.
        exact: The replica's saved file and answer, taken at the same steps.
    """
    (measured_document, measured_result), (exact_document, exact_result) = measured, exact
    assert measured_result.get("deltas") == exact_result.get("deltas"), (
        f"{kind}: the replica stepped {exact_result.get('deltas')}, "
        f"the served run {measured_result.get('deltas')}"
    )
    if kind in MATRIX_KINDS:
        assert labels(measured_document) == labels(exact_document), f"{kind}: labels differ"
    atol = band(kind, measured_result)
    rows = labels(measured_document)[0]
    found = disagreements(
        quantity(kind, measured_document, measured_result),
        quantity(kind, exact_document, exact_result),
        atol,
        rows,
    )
    assert not found, f"{kind}: {len(found)} entries outside {atol:.3g}: {found[:10]}"


class ReplicaRuntime:
    """``osprey.runtime``'s channel functions over a composite in this process.

    Args:
        composite: Anything with the composite's ``get(list) -> dict`` and
            ``set(dict)``.
        limits: The limits each address is checked against, by address.
    """

    def __init__(self, composite: Any, limits: Mapping[str, Any]) -> None:
        self.composite = composite
        self.limits = dict(limits)
        self.writes: list[tuple[str, Any]] = []

    def read_channel(self, channel_address: str, **_kwargs: Any) -> Any:
        return self.composite.get([channel_address])[channel_address]

    def read_channels(self, addresses: Sequence[str], **_kwargs: Any) -> list[Any]:
        values = self.composite.get(list(addresses))
        return [values[address] for address in addresses]

    def write_channel(self, channel_address: str, value: Any, **_kwargs: Any) -> None:
        self.writes.append((channel_address, value))
        self.composite.set({channel_address: value})

    def write_channels(self, channel_values: Mapping[str, Any], **_kwargs: Any) -> None:
        for address, value in channel_values.items():
            self.write_channel(address, value)

    def channel_limits(self, address: str) -> Any:
        return self.limits.get(address)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> ReplicaRuntime:
        """Replace ``osprey.runtime``'s channel functions with this replica's."""
        import osprey.runtime

        for name in (
            "read_channel",
            "read_channels",
            "write_channel",
            "write_channels",
            "channel_limits",
        ):
            monkeypatch.setattr(osprey.runtime, name, getattr(self, name))
        return self


# ---------------------------------------------------------------------------
# The bands
# ---------------------------------------------------------------------------


def test_an_orbit_response_band_is_the_orbit_floor_over_the_smallest_step() -> None:
    result = {"deltas": {"H1": 2.0e-5, "H2": -1.0e-5}}

    assert band("orm", result) == pytest.approx(ARMS * ORBIT_ATOL_M / 1.0e-5)


def test_a_dispersion_band_is_over_the_rf_span() -> None:
    assert band("dispersion", {"rf_span_hz": -50.0}) == pytest.approx(ARMS * ORBIT_ATOL_M / 50.0)


def test_a_tune_response_band_is_the_tune_floor_over_the_group_step() -> None:
    assert band("trm", {"deltas": {"QF": 1.0e-3}}) == pytest.approx(ARMS * TUNE_ATOL / 1.0e-3)


def test_a_chromaticity_band_is_over_the_monitors_momentum_step() -> None:
    monitor = {"e_delta_used": 1.0e-3}

    assert band("chromaticity_monitor", monitor) == pytest.approx(ARMS * TUNE_ATOL / 1.0e-3)
    assert band("crm", {**monitor, "deltas": {"SF": 1.0e-2}}) == pytest.approx(
        ARMS * (ARMS * TUNE_ATOL / 1.0e-3) / 1.0e-2
    )


def test_a_result_without_a_step_has_no_band() -> None:
    with pytest.raises(AssertionError, match="states no step"):
        band("orm", {"deltas": {}})
    with pytest.raises(ValueError, match="no kind"):
        band("bba", {})


# ---------------------------------------------------------------------------
# The comparison, on recorded measurements
# ---------------------------------------------------------------------------

#: A recorded unipolar orbit response of two monitor planes to two correctors.
RECORDED_ORM = {
    "type": "pyaml_cs_osprey.measure.step_measurement",
    "matrix": [[12.5, -3.25], [0.75, 8.0]],
    "observable_names": ["BPM1.x", "BPM1.y"],
    "variable_names": ["H1", "V1"],
}
RECORDED_ORM_RESULT = {"kind": "orm", "deltas": {"H1": 1.0e-5, "V1": 1.0e-5}}


def _shifted(document: Mapping[str, Any], by: float) -> dict[str, Any]:
    matrix = np.asarray(document["matrix"], dtype=float)
    matrix[0, 1] += by
    return {**document, "matrix": matrix.tolist()}


def test_a_measurement_within_the_band_meets_the_replica() -> None:
    atol = band("orm", RECORDED_ORM_RESULT)

    assert_meets(
        "orm",
        (_shifted(RECORDED_ORM, 0.5 * atol), RECORDED_ORM_RESULT),
        (RECORDED_ORM, RECORDED_ORM_RESULT),
    )


def test_an_entry_outside_the_band_fails_naming_its_row() -> None:
    atol = band("orm", RECORDED_ORM_RESULT)

    with pytest.raises(AssertionError, match=r"BPM1\.x\[1\]"):
        assert_meets(
            "orm",
            (_shifted(RECORDED_ORM, 2.0 * atol), RECORDED_ORM_RESULT),
            (RECORDED_ORM, RECORDED_ORM_RESULT),
        )


def test_a_replica_that_stepped_otherwise_is_no_comparison() -> None:
    other = {**RECORDED_ORM_RESULT, "deltas": {"H1": 2.0e-5, "V1": 1.0e-5}}

    with pytest.raises(AssertionError, match="the replica stepped"):
        assert_meets("orm", (RECORDED_ORM, RECORDED_ORM_RESULT), (RECORDED_ORM, other))


def test_relabelled_rows_fail() -> None:
    relabelled = {**RECORDED_ORM, "observable_names": ["BPM2.x", "BPM2.y"]}

    with pytest.raises(AssertionError, match="labels differ"):
        assert_meets(
            "orm",
            (relabelled, RECORDED_ORM_RESULT),
            (RECORDED_ORM, RECORDED_ORM_RESULT),
        )


def test_a_shape_or_a_non_finite_entry_is_a_disagreement() -> None:
    exact = np.asarray(RECORDED_ORM["matrix"])

    assert disagreements(exact[:, :1], exact, 1.0) == ["shape (2, 1) against (2, 2)"]
    poisoned = exact.copy()
    poisoned[1, 0] = math.nan
    assert len(disagreements(poisoned, exact, 1.0)) == 1


def test_the_chromaticity_monitor_is_compared_on_the_chromaticity_it_answered() -> None:
    result = {"kind": "chromaticity_monitor", "chromaticity": [1.25, -0.5], "e_delta_used": 1e-3}
    nudged = {**result, "chromaticity": [1.25 + 0.5 * band("chromaticity_monitor", result), -0.5]}

    assert_meets("chromaticity_monitor", ({}, nudged), ({}, result))
    with pytest.raises(AssertionError, match="outside"):
        assert_meets(
            "chromaticity_monitor", ({}, {**result, "chromaticity": [2.0, -0.5]}), ({}, result)
        )


# ---------------------------------------------------------------------------
# The replica's runtime
# ---------------------------------------------------------------------------


class _Composite:
    def __init__(self, values: dict[str, float]) -> None:
        self.values = dict(values)

    def get(self, names: list[str]) -> dict[str, float]:
        return {name: self.values[name] for name in names}

    def set(self, values: dict[str, float]) -> None:
        self.values.update(values)


def test_the_replica_reads_and_writes_the_composite_and_answers_its_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osprey.runtime

    composite = _Composite({"H1:SP": 0.0, "BPM1:X": 1.5e-4})
    replica = ReplicaRuntime(composite, {"H1:SP": "limits"}).install(monkeypatch)

    osprey.runtime.write_channel("H1:SP", 2.0)
    osprey.runtime.write_channels({"H1:SP": 3.0})

    assert osprey.runtime.read_channels(["H1:SP", "BPM1:X"]) == [3.0, 1.5e-4]
    assert osprey.runtime.read_channel("BPM1:X") == 1.5e-4
    assert osprey.runtime.channel_limits("H1:SP") == "limits"
    assert osprey.runtime.channel_limits("BPM1:X") is None
    assert replica.writes == [("H1:SP", 2.0), ("H1:SP", 3.0)]
