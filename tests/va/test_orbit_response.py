"""One device's orbit response, measured through the served model.

:func:`~osprey.services.virtual_accelerator.lattice.response.orbit_response`
is the oracle a facility's exported response matrix is checked against. It
writes a bipolar sweep in hardware units through the model's own write path,
reads the monitors back in hardware units, and converts both sides to physics
through the calibrations the bindings document carries -- so what it returns is
in the units the export states, and nothing about it is derived from a family
name.

The ring here is synthetic: a small stable FODO lattice with one cavity, eight
cells, unique names on every bound element, and no nonlinear element beyond the
bends' own curvature. Two properties of it are load-bearing. Its closed-orbit
response to a corrector kick is linear to well within the tolerances below, so
a response measured over one sweep is the response measured over any other;
and a horizontal kick moves the horizontal orbit alone, so the two planes can
be told apart. Nothing here needs a particular accelerator, and no address
carries a facility's vocabulary: the bindings document is the only thing that
pairs an address with an element.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, NamedTuple

import at
import pytest

from osprey.services.virtual_accelerator.bindings import Binding, load_bindings
from osprey.services.virtual_accelerator.lattice.calibration import to_physics
from osprey.services.virtual_accelerator.lattice.response import orbit_response
from osprey.services.virtual_accelerator.manifest import PARTITION_PYAT_COUPLED
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

C_LIGHT = 299792458.0

#: The deck these fixture trees are exported at, in GeV.
DECK_ENERGY_GEV = 3.0

#: Cells of the fixture ring; one monitor and one corrector each.
CELLS = 8

#: The focusing strength the ring is built with, and its harmonic number.
QUAD_K = 1.1
HARMONIC = 88

#: Radians of kick per amp, as a corrector's exported calibration states it.
KICK_GAIN = 1.0e-6

#: A monitor's two exported curves: metres per millimetre on the way to
#: physics, millimetres per metre on the way back. They are consistent here,
#: which is what makes a reading that went out through one and back through
#: the other the metres the solve produced.
BPM_TO_PHYSICS = 1.0e-3
BPM_TO_HARDWARE = 1.0e3

#: The sweep width the tests drive, in amps, and the corrector band that
#: admits it.
DELTA_A = 2.0
CORRECTOR_BAND = (-10.0, 10.0)

#: The addresses the fixture documents bind. Nothing relates them to the
#: element names below; the bindings document does that and only it.
H_CORR = "R1:PWR:CORH:01:CUR:SP"
V_CORR = "R1:PWR:CORV:02:CUR:SP"
SPLIT_CORR = "R1:PWR:CORS:03:CUR:SP"
FIRST_PIECE = "R1:PWR:CORP:03:CUR:SP"
SECOND_PIECE = "R1:PWR:CORP:04:CUR:SP"


def _monitor_address(cell: int, axis: str) -> str:
    """The address one monitor's reading on one axis is published on."""
    return f"R1:DIA:BPM:{cell:02d}:POS:{axis.upper()}"


def _corrector(cell: int) -> str:
    """The deck's own name for the corrector of ``cell``."""
    return f"HC{cell}"


def _monitor_element(cell: int) -> str:
    """The deck's own name for the monitor of ``cell``."""
    return f"BPM{cell}"


# -- the synthetic emitted tree ----------------------------------------------


def _ring(cells: int = CELLS, kq: float = QUAD_K) -> at.Lattice:
    """A small stable ring with one cavity and uniquely named magnets.

    Eight FODO cells, each carrying a monitor and a corrector, closed by
    sixteen dipoles. Returned 4D, the way a ring saved out of a facility's
    simulator model arrives; ``build_ring`` is what enables the cavity.
    """
    angle = 2 * math.pi / (2 * cells)
    elements: list[Any] = []
    for cell in range(1, cells + 1):
        elements += [
            at.Quadrupole(f"QF{cell}", 0.3, kq),
            at.Drift("DR", 1.0),
            at.Dipole(f"BD{cell}A", 1.0, angle),
            at.Monitor(_monitor_element(cell)),
            at.Drift("DR", 1.0),
            at.Quadrupole(f"QD{cell}", 0.3, -kq),
            at.Drift("DR", 1.0),
            at.Corrector(_corrector(cell), 0.0, [0.0, 0.0]),
            at.Dipole(f"BD{cell}B", 1.0, angle),
            at.Drift("DR", 1.0),
        ]
    ring = at.Lattice(elements, name="fixture", energy=DECK_ENERGY_GEV * 1.0e9, periodicity=1)
    frequency = HARMONIC * C_LIGHT / ring.circumference
    ring.append(at.RFCavity("RFC", 0.0, 1.0e6, frequency, HARMONIC, ring.energy))
    ring.disable_6d()
    return ring


def _linear(gain: float, offset: float = 0.0) -> dict:
    return {"kind": "linear", "gain": gain, "offset": offset}


def _kick(
    address: str,
    *,
    family: str,
    slices: list[tuple[str, float]],
    index: int = 0,
    calibration: dict | None = None,
) -> dict:
    """One corrector setpoint, over the pieces it is bound to."""
    return {
        "kind": "kick",
        "family": family,
        "setpoint_address": address,
        "readback_address": address.replace(":SP", ":RB"),
        "readback": "identity",
        "element": slices[0][0],
        "attribute": "KickAngle",
        "index": index,
        "slices": [{"element": name, "weight": weight} for name, weight in slices],
        "owner": family,
        "calibration": _linear(KICK_GAIN) if calibration is None else calibration,
        "monitor_inverse": None,
        "nominal": 0.0,
        "energy_scaling": "none",
        "energy_table": None,
    }


def _monitor(cell: int, axis: str, *, calibration: dict | None = None) -> dict:
    """One orbit reading: metres on the ring, millimetres on the wire."""
    return {
        "kind": "monitor",
        "family": f"bpm{axis}",
        "setpoint_address": _monitor_address(cell, axis),
        "readback_address": None,
        "readback": "inverse",
        "element": _monitor_element(cell),
        "attribute": axis,
        "index": None,
        "slices": [{"element": _monitor_element(cell), "weight": 1.0}],
        "owner": "bpm",
        "calibration": _linear(BPM_TO_PHYSICS) if calibration is None else calibration,
        "monitor_inverse": _linear(BPM_TO_HARDWARE),
        "nominal": None,
        "energy_scaling": "none",
        "energy_table": None,
    }


def _bindings(
    *,
    kick_calibration: dict | None = None,
    axes: tuple[str, ...] = ("x", "y"),
    monitor_calibrations: dict[tuple[int, str], dict] | None = None,
) -> list[dict]:
    """The standard document: three correctors and every monitor of the ring.

    One corrector per plane on its own element, plus one whose kick is shared
    over two pieces -- the three shapes an actuator binding comes in.
    """
    overrides = {} if monitor_calibrations is None else monitor_calibrations
    return [
        _kick(
            H_CORR,
            family="corh",
            slices=[(_corrector(1), 1.0)],
            calibration=kick_calibration,
        ),
        _kick(
            V_CORR,
            family="corv",
            slices=[(_corrector(2), 1.0)],
            index=1,
            calibration=kick_calibration,
        ),
        _kick(
            SPLIT_CORR,
            family="cors",
            slices=[(_corrector(3), 0.5), (_corrector(4), 0.5)],
            calibration=kick_calibration,
        ),
        *(
            _monitor(cell, axis, calibration=overrides.get((cell, axis)))
            for cell in range(1, CELLS + 1)
            for axis in axes
        ),
    ]


def _piece_bindings() -> list[dict]:
    """A document binding the split corrector's two pieces separately.

    The same two elements the split corrector writes, one address each and the
    whole kick on each -- which is what makes the shared kick comparable with
    its pieces. One document cannot hold both: the schema refuses two bindings
    writing one element field.
    """
    return [
        _kick(FIRST_PIECE, family="corp", slices=[(_corrector(3), 1.0)]),
        _kick(SECOND_PIECE, family="corp", slices=[(_corrector(4), 1.0)]),
        *(_monitor(cell, axis) for cell in range(1, CELLS + 1) for axis in ("x", "y")),
    ]


def _channel(address: str) -> dict:
    """One manifest channel, in the per-channel schema the manifest carries."""
    ring, system, family, device, field, subfield = address.split(":")
    return {
        "address": address,
        "ring": ring,
        "system": system,
        "family": family,
        "device": device,
        "field": field,
        "subfield": subfield,
        "partition": PARTITION_PYAT_COUPLED,
        "record_type": "ai",
        "noise": False,
    }


def _manifest(bindings: list[dict]) -> list[dict]:
    """The channel list a deployment of this tree resolved.

    Every bound address and nothing else: a coupled channel the document does
    not bind refuses the boot, and an address the manifest omits reaches no
    variable at all.
    """
    return [_channel(body["setpoint_address"]) for body in bindings]


def _machine(bindings: list[dict]) -> dict:
    """Every nominal and unit the served ``machine.json`` declares."""
    return {
        body["setpoint_address"]: (
            {"value": 0.0, "units": "mm"}
            if body["kind"] == "monitor"
            else {"value": body["nominal"], "units": "A"}
        )
        for body in bindings
    }


def _limits(bindings: list[dict]) -> dict:
    """The write bands the served ``channel_limits.json`` ships."""
    bands: dict[str, Any] = {"_version": "1.0", "defaults": {"writable": True, "confirm": True}}
    for body in bindings:
        if body["kind"] != "monitor":
            bands[body["setpoint_address"]] = {
                "min_value": CORRECTOR_BAND[0],
                "max_value": CORRECTOR_BAND[1],
            }
    return bands


class _Served(NamedTuple):
    """A written tree, and what a caller needs to drive the model it serves."""

    data_dir: Path
    manifest: list[dict]
    bindings: dict[str, Binding]

    def boot(self) -> PyATRingModel:
        """A fresh model on this tree; every model owns its own ring."""
        return PyATRingModel(self.data_dir, self.manifest)

    def monitors(self) -> list[Binding]:
        """Every monitor binding of the document, in document order."""
        return [binding for binding in self.bindings.values() if binding.kind == "monitor"]


def _serve(root: Path, bindings: list[dict]) -> _Served:
    """Write a served tree and return the handles onto it.

    Args:
        root: the data root to write. ``simulation/`` holds the lattice and
            the bindings; the write bands sit beside it, the way
            :class:`ManifestPaths` resolves a facility tree.
        bindings: the document's bindings, as JSON bodies.
    """
    paths = ManifestPaths(data_root=root)
    paths.lattice_json.parent.mkdir(parents=True, exist_ok=True)
    at.save_lattice(_ring(), paths.lattice_json)
    document = {
        "system": "StorageRing",
        "energy_gev": DECK_ENERGY_GEV,
        "lattice_sha256": hashlib.sha256(paths.lattice_json.read_bytes()).hexdigest(),
        "bindings": bindings,
    }
    paths.va_bindings.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    paths.machine_json.write_text(
        json.dumps({"name": "fixture", "channels": _machine(bindings)}), encoding="utf-8"
    )
    paths.channel_limits.write_text(json.dumps(_limits(bindings)), encoding="utf-8")
    parsed = load_bindings(paths.va_bindings)
    return _Served(
        data_dir=root,
        manifest=_manifest(bindings),
        bindings={binding.setpoint_address: binding for binding in parsed.bindings},
    )


@pytest.fixture(scope="module")
def served(tmp_path_factory: pytest.TempPathFactory) -> _Served:
    """The standard tree: consistent monitor curves, both planes bound."""
    return _serve(tmp_path_factory.mktemp("standard") / "data", _bindings())


@pytest.fixture(scope="module")
def booted(served: _Served) -> PyATRingModel:
    """One boot, shared by the tests that leave the model where they found it.

    Every :func:`orbit_response` call restores the setpoint it swept, so a
    model is reusable across those calls by construction -- which is one of
    the things the tests below pin.
    """
    return served.boot()


def _response(
    model: PyATRingModel,
    served: _Served,
    address: str,
    delta: float = DELTA_A,
) -> dict[str, tuple[float, float]]:
    """The response of one bound corrector, read at every bound monitor."""
    return orbit_response(model, served.bindings[address], delta, monitors=served.monitors())


def _axis(response: dict[str, tuple[float, float]], cell: int, axis: str) -> float:
    """One entry of a response, by the monitor element and the plane."""
    return response[_monitor_element(cell)][0 if axis == "x" else 1]


# -- what the returned number is ---------------------------------------------


class TestTheResponseIsPhysicsOverPhysics:
    """Both sides go through the document's own curves, and nothing else."""

    def test_it_is_the_monitor_secant_over_the_actuator_secant(
        self, served: _Served, booted: PyATRingModel
    ) -> None:
        """The definition, recomputed through the same public write path.

        Hardware in, hardware out, each mapped to physics by the binding that
        carries it, and the quotient of the two spans -- not the slope of
        either curve at a point, and not the orbit the lattice happens to
        hold in native units.
        """
        binding = served.bindings[H_CORR]
        monitors = served.monitors()

        response = _response(booted, served, H_CORR)

        center = booted.get(H_CORR)
        booted.set({H_CORR: center + DELTA_A / 2})
        plus = {
            monitor.setpoint_address: booted.get(monitor.setpoint_address) for monitor in monitors
        }
        booted.set({H_CORR: center - DELTA_A / 2})
        minus = {
            monitor.setpoint_address: booted.get(monitor.setpoint_address) for monitor in monitors
        }
        booted.set({H_CORR: center})
        span = float(to_physics(binding.calibration, center + DELTA_A / 2)) - float(
            to_physics(binding.calibration, center - DELTA_A / 2)
        )
        for monitor in monitors:
            address = monitor.setpoint_address
            expected = (
                float(to_physics(monitor.calibration, plus[address]))
                - float(to_physics(monitor.calibration, minus[address]))
            ) / span
            assert _axis(response, int(address.split(":")[3]), monitor.attribute) == pytest.approx(
                expected, rel=1e-12, abs=1e-18
            )

    def test_the_exported_hardware_gain_cancels(
        self, tmp_path_factory: pytest.TempPathFactory, served: _Served, booted: PyATRingModel
    ) -> None:
        """A response in physics units cannot depend on the units written.

        The same corrector on a tree whose exported calibration is twice as
        steep receives twice the kick and divides by twice the span, so the
        quotient is the one the facility measured either way.
        """
        steep = _serve(
            tmp_path_factory.mktemp("steep") / "data",
            _bindings(kick_calibration=_linear(2 * KICK_GAIN)),
        )

        response = _response(booted, served, H_CORR)
        steeper = _response(steep.boot(), steep, H_CORR)

        for cell in range(1, CELLS + 1):
            assert _axis(steeper, cell, "x") == pytest.approx(
                _axis(response, cell, "x"), rel=1e-6, abs=1e-12
            )

    def test_a_monitor_whose_two_curves_disagree_scales_its_own_entry(
        self, tmp_path_factory: pytest.TempPathFactory, served: _Served, booted: PyATRingModel
    ) -> None:
        """The reading goes out through the inverse and back through the
        calibration, so the two exported curves are both exercised.

        A monitor whose forward curve is twice its inverse's reciprocal
        reports twice the physics reading -- which is the disagreement a
        response check exists to surface. Reading metres off the solve
        instead would hide it, and every other monitor is unmoved.
        """
        mismatched = _serve(
            tmp_path_factory.mktemp("mismatched") / "data",
            _bindings(monitor_calibrations={(1, "x"): _linear(2 * BPM_TO_PHYSICS)}),
        )

        response = _response(booted, served, H_CORR)
        skewed = _response(mismatched.boot(), mismatched, H_CORR)

        assert _axis(skewed, 1, "x") == pytest.approx(2 * _axis(response, 1, "x"), rel=1e-9)
        assert _axis(skewed, 2, "x") == pytest.approx(_axis(response, 2, "x"), rel=1e-9)

    def test_the_sweep_width_does_not_change_the_answer(
        self, served: _Served, booted: PyATRingModel
    ) -> None:
        """A response is a slope: on this ring it is the same slope at any
        width, which is what makes the file's own ``ActuatorDelta`` usable."""
        narrow = _response(booted, served, H_CORR, delta=DELTA_A)
        wide = _response(booted, served, H_CORR, delta=4 * DELTA_A)

        for cell in range(1, CELLS + 1):
            assert _axis(wide, cell, "x") == pytest.approx(
                _axis(narrow, cell, "x"), rel=1e-6, abs=1e-12
            )

    def test_a_sampled_calibration_is_divided_by_its_secant(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """Through the two arms, not along the curve at the operating point.

        The table below has a kink at the centre of the sweep: its slope on
        either arm is one gain or the other, and only the secant between the
        two arms is their mean. A tree whose calibration is that mean as a
        straight line therefore measures the same response -- to the ring's
        own linearity, which is what limits the tolerance here: the kinked
        tree kicks the beam asymmetrically (three halves of the mean one way,
        one half the other) and the straight one symmetrically, so the two
        sweeps sample the bends' curvature at different amplitudes and the
        slope they share is shared to parts in a million, not exactly. Either
        of the table's own arm slopes as the denominator would be out by a
        third.
        """
        low, high = 0.5 * KICK_GAIN, 1.5 * KICK_GAIN
        kinked = _serve(
            tmp_path_factory.mktemp("kinked") / "data",
            _bindings(
                kick_calibration={
                    "kind": "table",
                    "grid": [-CORRECTOR_BAND[1], 0.0, CORRECTOR_BAND[1]],
                    "values": [-CORRECTOR_BAND[1] * low, 0.0, CORRECTOR_BAND[1] * high],
                }
            ),
        )
        secant = _serve(
            tmp_path_factory.mktemp("secant") / "data",
            _bindings(kick_calibration=_linear(0.5 * (low + high))),
        )

        through_the_table = _response(kinked.boot(), kinked, H_CORR)
        through_the_line = _response(secant.boot(), secant, H_CORR)

        for cell in range(1, CELLS + 1):
            assert _axis(through_the_table, cell, "x") == pytest.approx(
                _axis(through_the_line, cell, "x"), rel=1e-4, abs=1e-12
            )


# -- the shape of the result -------------------------------------------------


class TestTheResultIsKeyedByMonitorElement:
    """One entry per monitor element, carrying both of its planes."""

    def test_every_bound_monitor_element_appears_once(
        self, served: _Served, booted: PyATRingModel
    ) -> None:
        response = _response(booted, served, H_CORR)

        assert set(response) == {_monitor_element(cell) for cell in range(1, CELLS + 1)}

    def test_a_horizontal_kick_moves_the_horizontal_plane_alone(
        self, served: _Served, booted: PyATRingModel
    ) -> None:
        response = _response(booted, served, H_CORR)

        assert any(abs(_axis(response, cell, "x")) > 1.0 for cell in range(1, CELLS + 1))
        for cell in range(1, CELLS + 1):
            assert _axis(response, cell, "y") == pytest.approx(0.0, abs=1e-9)

    def test_a_vertical_kick_moves_the_vertical_plane_alone(
        self, served: _Served, booted: PyATRingModel
    ) -> None:
        """The plane comes from the binding's own component, so the two
        correctors differ in nothing but the index they write."""
        response = _response(booted, served, V_CORR)

        assert any(abs(_axis(response, cell, "y")) > 1.0 for cell in range(1, CELLS + 1))
        for cell in range(1, CELLS + 1):
            assert _axis(response, cell, "x") == pytest.approx(0.0, abs=1e-9)

    def test_an_unbound_plane_is_not_a_number(
        self, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        """A tree that publishes one plane has no reading for the other, and
        a zero there would read as a monitor that saw nothing move."""
        horizontal = _serve(tmp_path_factory.mktemp("horizontal") / "data", _bindings(axes=("x",)))

        response = _response(horizontal.boot(), horizontal, H_CORR)

        for cell in range(1, CELLS + 1):
            assert math.isfinite(_axis(response, cell, "x"))
            assert math.isnan(_axis(response, cell, "y"))

    def test_a_shared_kick_responds_as_the_mean_of_its_pieces(
        self, tmp_path_factory: pytest.TempPathFactory, served: _Served, booted: PyATRingModel
    ) -> None:
        """A sliced device is one setpoint over several elements, each taking
        its share -- so its response is the mean of what its pieces do at the
        same setpoint, which is only true if the write went through the
        model's own slice handling."""
        pieces = _serve(tmp_path_factory.mktemp("pieces") / "data", _piece_bindings())
        piece_model = pieces.boot()

        shared = _response(booted, served, SPLIT_CORR)
        first = _response(piece_model, pieces, FIRST_PIECE)
        second = _response(piece_model, pieces, SECOND_PIECE)

        for cell in range(1, CELLS + 1):
            mean = 0.5 * (_axis(first, cell, "x") + _axis(second, cell, "x"))
            assert _axis(shared, cell, "x") == pytest.approx(mean, rel=1e-6, abs=1e-12)


# -- the sweep, and what it leaves behind ------------------------------------


class TestTheSweepIsBipolarAndRolledBack:
    """Two arms about the value the model is holding, and then that value."""

    def test_the_arms_straddle_the_held_setpoint(self, served: _Served) -> None:
        """The operating point is the model's own retained value, so a caller
        positions the sweep by writing the setpoint it wants swept about."""
        model = served.boot()
        model.set({H_CORR: 3.0})
        written: list[float] = []
        original = model.set

        def record(values: dict) -> None:
            written.extend(values.values())
            original(values)

        model.set = record  # type: ignore[method-assign]

        _response(model, served, H_CORR)

        assert written == [3.0 + DELTA_A / 2, 3.0 - DELTA_A / 2, 3.0]

    def test_the_model_is_left_where_the_sweep_found_it(self, served: _Served) -> None:
        model = served.boot()
        before = model.get(H_CORR)
        kick_before = model.lattice[model.element_index(_corrector(1))].KickAngle[0]
        reading_before = model.get(_monitor_address(1, "x"))

        _response(model, served, H_CORR)

        assert model.get(H_CORR) == before
        assert model.lattice[model.element_index(_corrector(1))].KickAngle[0] == pytest.approx(
            kick_before
        )
        assert model.get(_monitor_address(1, "x")) == pytest.approx(reading_before, abs=1e-12)

    def test_a_failed_reading_still_restores_the_setpoint(self, served: _Served) -> None:
        """The sweep is not an experiment a caller can be left in the middle
        of: a monitor read that raises leaves the lattice as it was found."""
        model = served.boot()
        kick_before = model.lattice[model.element_index(_corrector(1))].KickAngle[0]
        original = model.get

        monitor_addresses = {monitor.setpoint_address for monitor in served.monitors()}

        def fail_on_monitors(names: list[str] | str) -> Any:
            asked = {names} if isinstance(names, str) else set(names)
            if asked & monitor_addresses:
                raise RuntimeError("the monitor read failed")
            return original(names)

        model.get = fail_on_monitors  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="the monitor read failed"):
            _response(model, served, H_CORR)

        model.get = original  # type: ignore[method-assign]
        assert model.lattice[model.element_index(_corrector(1))].KickAngle[0] == pytest.approx(
            kick_before
        )
        assert model.get(H_CORR) == pytest.approx(0.0)

    def test_consecutive_sweeps_are_independent(
        self, served: _Served, booted: PyATRingModel
    ) -> None:
        """Nothing accumulates: the same corrector swept twice, and a second
        corrector swept in between, give one answer."""
        first = _response(booted, served, H_CORR)
        _response(booted, served, V_CORR)
        again = _response(booted, served, H_CORR)

        for cell in range(1, CELLS + 1):
            assert _axis(again, cell, "x") == pytest.approx(_axis(first, cell, "x"), rel=1e-12)


# -- refusals ----------------------------------------------------------------


class TestWhatItRefuses:
    """Every refusal names the address it is about."""

    def test_a_monitor_is_not_an_actuator(self, served: _Served, booted: PyATRingModel) -> None:
        monitor = served.bindings[_monitor_address(1, "x")]

        with pytest.raises(ValueError, match="is read only"):
            orbit_response(booted, monitor, DELTA_A, monitors=served.monitors())

    def test_an_actuator_with_no_calibration_is_refused(
        self, served: _Served, booted: PyATRingModel
    ) -> None:
        """The energy knob is the binding this describes: it converts through
        its energy table, so it has no physics span to divide by."""
        uncalibrated = dataclasses.replace(served.bindings[H_CORR], calibration=None)

        with pytest.raises(ValueError, match="no calibration"):
            orbit_response(booted, uncalibrated, DELTA_A, monitors=served.monitors())

    @pytest.mark.parametrize("delta", [0.0, math.nan, math.inf])
    def test_a_sweep_of_no_width_is_refused(
        self, served: _Served, booted: PyATRingModel, delta: float
    ) -> None:
        with pytest.raises(ValueError, match="finite and nonzero"):
            _response(booted, served, H_CORR, delta=delta)

    def test_an_actuator_the_model_does_not_serve_is_refused(
        self, served: _Served, booted: PyATRingModel
    ) -> None:
        elsewhere = dataclasses.replace(
            served.bindings[H_CORR], setpoint_address="R1:PWR:CORH:99:CUR:SP"
        )

        with pytest.raises(ValueError, match="99"):
            orbit_response(booted, elsewhere, DELTA_A, monitors=served.monitors())

    def test_a_monitor_the_model_does_not_serve_is_refused(
        self, served: _Served, booted: PyATRingModel
    ) -> None:
        """Skipping it would drop a row of the response matrix silently."""
        elsewhere = dataclasses.replace(
            served.bindings[_monitor_address(1, "x")],
            setpoint_address=_monitor_address(99, "x"),
        )

        with pytest.raises(ValueError, match="99"):
            orbit_response(booted, served.bindings[H_CORR], DELTA_A, monitors=[elsewhere])

    def test_a_writable_cannot_stand_in_for_a_monitor(
        self, served: _Served, booted: PyATRingModel
    ) -> None:
        with pytest.raises(ValueError, match="not a monitor"):
            orbit_response(
                booted,
                served.bindings[H_CORR],
                DELTA_A,
                monitors=[served.bindings[V_CORR]],
            )

    def test_a_sweep_with_nothing_to_read_is_refused(
        self, served: _Served, booted: PyATRingModel
    ) -> None:
        with pytest.raises(ValueError, match="at least one monitor"):
            orbit_response(booted, served.bindings[H_CORR], DELTA_A, monitors=[])

    def test_one_plane_of_one_monitor_is_read_once(
        self, served: _Served, booted: PyATRingModel
    ) -> None:
        """Two bindings on one element and one axis would put two readings in
        one entry, the second silently winning."""
        monitor = served.bindings[_monitor_address(1, "x")]

        with pytest.raises(ValueError, match="twice"):
            orbit_response(booted, served.bindings[H_CORR], DELTA_A, monitors=[monitor, monitor])
