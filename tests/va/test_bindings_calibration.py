"""Every calibration a served tree exports is that deck's own physics.

A binding's calibration is the one number that decides what a hardware value
means: the served ``va_bindings.json`` states a curve per channel, and the
model puts a written value through it before anything reaches the lattice.
Nothing downstream can tell a right curve from a wrong one -- a gain off by a
factor still produces a plausible orbit -- so the check has to be against the
deck itself.

The identity this module pins is the one a facility's export is built to
satisfy, in both directions:

* a setpoint's declared nominal, put through its own calibration, is exactly
  the quantity the deck's element carries at boot. That is what makes "the
  machine is at its nominal" true of the served model rather than merely
  declared by it;
* a monitor's published reading, put through its own calibration, is exactly
  the closed-orbit coordinate the solve produced at the element it reads --
  the hardware-to-physics direction, never the inverse curve run backwards
  (:mod:`~osprey.services.virtual_accelerator.lattice.calibration` forbids
  deriving one from the other, and a facility may export a pair that does not
  invert).

The tree is the demo facility's own, which is generated rather than hand
written, so what is checked here is a real export of ~500 bindings rather than
a handful of fixture curves. Every binding is reached through the locator its
own document carries -- element name, attribute and component -- so no family,
count or strength constant appears below; a tree with other families and other
curves satisfies exactly the same assertions.
"""

from __future__ import annotations

import at
import pytest

from osprey.services.virtual_accelerator.bindings import Binding, BindingsDocument, load_bindings
from osprey.services.virtual_accelerator.lattice.calibration import to_physics
from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

#: Which row of a pyAT orbit vector a monitor's transverse axis reads.
_ORBIT_ROW = {"x": 0, "y": 2}

#: How far a step test moves a setpoint, as a fraction of the value it holds.
#: Small enough that the ring keeps its closed orbit under every footprint
#: moving at once, large enough to sit far above the solve's own noise.
_STEP_FRACTION = 1.0e-4

#: The hardware step used where a setpoint holds zero, so the step has a width
#: at all. In amps for a magnet, but nothing here depends on the unit.
_ZERO_STEP = 1.0

#: The calibration is exported at finite precision, so the line through it
#: reproduces the deck's baked number to round-off rather than bit for bit --
#: a millionth of a per cent is many orders below any curve error that
#: matters and many orders above the last bit of a double.
_ROUNDOFF = 1.0e-11


@pytest.fixture(scope="module")
def document() -> BindingsDocument:
    """The bindings the served tree ships."""
    return load_bindings(PACKAGE_PATHS.va_bindings)


@pytest.fixture(scope="module")
def channels() -> list[dict]:
    """The channel namespace a deployment of that tree resolves."""
    return build_manifest()["channels"]


@pytest.fixture(scope="module")
def booted(channels: list[dict]) -> PyATRingModel:
    """One model on the served tree, for the tests that only read it."""
    return PyATRingModel(PACKAGE_PATHS.data_root, channels)


@pytest.fixture
def model(channels: list[dict]) -> PyATRingModel:
    """A fresh model, for the test that writes to the lattice."""
    return PyATRingModel(PACKAGE_PATHS.data_root, channels)


def _writable(document: BindingsDocument) -> list[Binding]:
    return [binding for binding in document.bindings if binding.is_writable]


def _monitors(document: BindingsDocument) -> list[Binding]:
    return [binding for binding in document.bindings if binding.kind == "monitor"]


def _component(model: PyATRingModel, binding: Binding) -> float:
    """The quantity of the deck a setpoint binding drives.

    The binding's own locator and nothing else: the element it names, the
    attribute it names, and the component of that attribute it names. A
    binding with no component drives the attribute whole.
    """
    stored = getattr(model.lattice[model.element_index(binding.element)], binding.attribute)
    return float(stored if binding.index is None else stored[binding.index])


def _footprints(bindings: list[Binding]) -> dict[tuple[str, str, int | None], Binding]:
    """One binding per distinct place in an element's storage a write lands.

    A document repeats each footprint once per device, and a calibration that
    misses its mark misses it the same way for every device sharing the
    footprint. One representative each keeps the stepping test to a single
    solve while still covering every shape of write the document contains.
    """
    chosen: dict[tuple[str, str, int | None], Binding] = {}
    for binding in bindings:
        chosen.setdefault((binding.kind, binding.attribute, binding.index), binding)
    return chosen


class TestASetpointsNominalIsTheStrengthItsElementCarries:
    """The boot identity: declared nominal in, baked deck quantity out."""

    def test_every_setpoint_lands_on_the_quantity_its_element_holds(
        self, document: BindingsDocument, booted: PyATRingModel
    ) -> None:
        divergent = [
            (binding.setpoint_address, _component(booted, binding), through)
            for binding in _writable(document)
            if _component(booted, binding)
            != pytest.approx(
                through := float(to_physics(binding.calibration, binding.nominal)),
                rel=_ROUNDOFF,
                abs=_ROUNDOFF,
            )
        ]
        assert divergent == []

    def test_the_model_serves_each_setpoint_from_that_same_nominal(
        self, document: BindingsDocument, booted: PyATRingModel
    ) -> None:
        """Both ends of the identity are the tree's: the value the model hands
        back is the nominal the check above put through the curve, so the two
        cannot drift apart while each still looks right on its own."""
        assert {
            binding.setpoint_address: float(booted.get(binding.setpoint_address))
            for binding in _writable(document)
        } == {binding.setpoint_address: binding.nominal for binding in _writable(document)}

    def test_no_curve_is_flat(self, document: BindingsDocument) -> None:
        """Guards the identity from holding trivially.

        A curve that answers the same physics value at every hardware value
        satisfies the identity above for any device parked where it maps to,
        and it would leave that device unreachable from the control system.
        Where the deck quantity itself is zero at nominal -- a trim coil about
        a nominal field is -- that is the deck's own state, and it is the
        curve rather than the value that has to carry the device's range.
        """
        writable = _writable(document)
        assert writable
        flat = [
            binding.setpoint_address
            for binding in writable
            if float(to_physics(binding.calibration, binding.nominal + _ZERO_STEP))
            == float(to_physics(binding.calibration, binding.nominal))
        ]
        assert flat == []

    def test_some_setpoint_holds_the_deck_away_from_zero(
        self, document: BindingsDocument, booted: PyATRingModel
    ) -> None:
        """And the ring the curves are checked against is an excited one: a
        deck whose every bound quantity were zero would match any curve that
        passes through the origin."""
        assert any(_component(booted, binding) != 0.0 for binding in _writable(document))

    def test_the_whole_writable_document_was_covered(
        self, document: BindingsDocument, booted: PyATRingModel
    ) -> None:
        """Every writable binding reaches an element of the served ring, so
        the identity above ran on all of them rather than on a subset a
        lookup quietly dropped."""
        writable = _writable(document)
        assert writable
        assert all(binding.element for binding in writable)
        assert len({binding.setpoint_address for binding in writable}) == len(writable)


class TestAStepMovesTheDeckByTheCurvesOwnSpan:
    def test_one_representative_of_every_footprint_moves_by_its_own_gain(
        self, document: BindingsDocument, model: PyATRingModel
    ) -> None:
        """The curve is the whole conversion, not just its value at nominal.

        An offset absorbed into a gain reproduces the nominal exactly and is
        wrong everywhere else, so what is checked here is the span the curve
        states between two hardware values against the span the deck actually
        moves. All footprints step in one batch, which is one solve and one
        rollback boundary.
        """
        footprints = _footprints(_writable(document))
        assert footprints, "the document writes nothing"

        held = {
            binding.setpoint_address: float(model.get(binding.setpoint_address))
            for binding in footprints.values()
        }
        stepped = {
            address: value + (abs(value) * _STEP_FRACTION or _ZERO_STEP)
            for address, value in held.items()
        }
        before = {key: _component(model, binding) for key, binding in footprints.items()}

        model.set(stepped)

        for key, binding in footprints.items():
            address = binding.setpoint_address
            span = float(to_physics(binding.calibration, stepped[address])) - float(
                to_physics(binding.calibration, held[address])
            )
            moved = _component(model, binding) - before[key]
            assert moved == pytest.approx(span, rel=_ROUNDOFF, abs=_ROUNDOFF), key


class TestAMonitorPublishesTheOrbitTheSolveProduced:
    def test_every_reading_is_its_elements_closed_orbit_through_its_own_curve(
        self, document: BindingsDocument, booted: PyATRingModel
    ) -> None:
        """The reading direction of the same identity.

        The monitor's calibration is the hardware-to-physics curve; running it
        over the number the model published has to give back the coordinate
        the solve found at that element. Read in one ``find_orbit`` pass over
        the ring the model is holding, so both sides describe one orbit.
        """
        monitors = _monitors(document)
        assert monitors, "the tree publishes no reading to check a curve against"
        index = {binding.element: booted.element_index(binding.element) for binding in monitors}
        refpts = sorted(set(index.values()))
        _closed, orbit = at.find_orbit(booted.lattice, refpts=refpts)
        solved = dict(zip(refpts, orbit, strict=True))

        divergent = [
            binding.setpoint_address
            for binding in monitors
            if float(to_physics(binding.calibration, booted.get(binding.setpoint_address)))
            != pytest.approx(
                float(solved[index[binding.element]][_ORBIT_ROW[binding.attribute]]),
                rel=_ROUNDOFF,
                abs=_ROUNDOFF,
            )
        ]
        assert divergent == []

    def test_a_monitor_reads_a_plane_the_orbit_vector_carries(
        self, document: BindingsDocument
    ) -> None:
        """The axes a monitor may name are the transverse ones, which is what
        makes the row lookup above total rather than a lucky hit."""
        assert {binding.attribute for binding in _monitors(document)} <= set(_ORBIT_ROW)

    def test_the_reading_moves_off_axis_somewhere_on_the_ring(
        self, document: BindingsDocument, booted: PyATRingModel
    ) -> None:
        """Guards the reading identity from holding on an all-zero orbit,
        which any curve through the origin would satisfy."""
        assert any(
            float(booted.get(binding.setpoint_address)) != 0.0 for binding in _monitors(document)
        )
