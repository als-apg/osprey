"""The demo tree's generated model: its lattice, its bindings, and the physics
they state.

``osprey.simulation.lattice.build`` is the demo facility's export lane. Where a
real facility exports a deck and its calibrations out of the Middle Layer, the
demo's are generated from the ring it is hand-ported from, so the two committed
artifacts under ``data/simulation/`` are build outputs and never hand edits.
Three things are checked here:

* **They regenerate byte for byte.** ``--check`` is the drift gate CI runs, and
  it compares bytes rather than meaning, so a value that moved is caught even
  when the document still parses.
* **They describe the facility spec's inventory.** Every magnet and corrector
  device carries one binding and every monitor two, which is what makes the
  manifest's ``pyat-coupled`` partition derivable from the spec end to end.
* **Their calibrations are the ring's own physics.** Each binding's straight
  line is checked against the strengths baked into the built ring and the
  nominal current the scenario seed states for that device, so a calibration
  can only be right by being the physics rather than by matching a pinned
  number.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter

import at
import pytest

from osprey.services.virtual_accelerator.bindings import (
    PROVENANCE_KEY,
    Linear,
    load_bindings,
    parse_bindings,
)
from osprey.services.virtual_accelerator.lattice.ring import build_ring as build_served_ring
from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS
from osprey.simulation.facility_spec import ALS_U_AR
from osprey.simulation.lattice import build as demo_build
from osprey.simulation.lattice.artifact import canonical_mat_path
from osprey.simulation.lattice.ring import build_ring

#: How close a generated calibration has to sit to the physics recomputed here.
#: Both sides are the same double arithmetic in a different order, so the only
#: gap is the order; a real disagreement is orders of magnitude wider.
_TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def ring():
    return build_ring()


@pytest.fixture(scope="module")
def elements(ring) -> dict:
    return {element.FamName: element for element in ring}


@pytest.fixture(scope="module")
def document():
    return load_bindings(PACKAGE_PATHS.va_bindings)


@pytest.fixture(scope="module")
def by_setpoint(document) -> dict:
    return {binding.setpoint_address: binding for binding in document.bindings}


@pytest.fixture(scope="module")
def machine_channels() -> dict:
    return json.loads(PACKAGE_PATHS.machine_json.read_text(encoding="utf-8"))["channels"]


class TestTheCommittedArtifactsAreGenerated:
    """The two files under ``data/simulation/`` are what the generator writes."""

    def test_the_tree_carries_both_artifacts(self):
        missing = [
            str(path)
            for path in (PACKAGE_PATHS.lattice_json, PACKAGE_PATHS.va_bindings)
            if not path.is_file()
        ]
        assert not missing, f"the demo tree is missing its generated model: {missing}"

    def test_check_passes_on_the_committed_tree(self):
        result = subprocess.run(
            [sys.executable, "-m", "osprey.simulation.lattice.build", "--check"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    def test_check_leaves_the_tree_untouched(self):
        before = {
            path: path.read_bytes()
            for path in (PACKAGE_PATHS.lattice_json, PACKAGE_PATHS.va_bindings)
        }
        demo_build.main(["--check"])
        after = {path: path.read_bytes() for path in before}
        assert after == before

    def test_the_committed_bytes_are_the_rendered_bytes(self, ring):
        lattice_text, bindings_text = demo_build.render_demo_model(ring)
        assert PACKAGE_PATHS.lattice_json.read_bytes() == lattice_text.encode("utf-8")
        assert PACKAGE_PATHS.va_bindings.read_bytes() == bindings_text.encode("utf-8")

    def test_rendering_twice_gives_the_same_bytes(self, ring):
        first = demo_build.render_demo_model(ring)
        second = demo_build.render_demo_model(build_ring())
        assert first == second


class TestTheTwoFilesDescribeOneAccelerator:
    """The bindings point into the lattice beside them, and the service agrees."""

    def test_the_bindings_stamp_the_committed_lattice(self, document):
        assert document.lattice_sha256 == demo_build.digest_of(PACKAGE_PATHS.lattice_json)

    def test_the_served_ring_builds_from_the_tree(self):
        served = build_served_ring(PACKAGE_PATHS)
        assert len(served) > 0

    def test_every_bound_element_is_named_once_in_the_ring(self, document, ring):
        census = Counter(element.FamName for element in ring)
        shared = sorted(
            {
                slice_.element
                for binding in document.bindings
                for slice_ in binding.slices
                if census[slice_.element] != 1
            }
        )
        assert not shared, (
            "the bindings bind elements by name, so a name the deck carries "
            f"none or several of says nothing about which element a channel writes: {shared}"
        )

    def test_the_deck_is_never_indexed_by_position(self):
        """A position taken against the ``.mat`` does not index the saved deck.

        The canonical ``.mat`` carries a ``RingParam`` row that the JSON deck
        folds into its properties, so the two files are one element out of
        step and a one-based position read off either one indexes the other
        wrongly. Bindings therefore resolve elements by name.
        """
        saved = json.loads(PACKAGE_PATHS.lattice_json.read_text(encoding="utf-8"))
        rows = at.load_mat(str(canonical_mat_path()), use="RING", keep_all=True)
        assert len(saved["elements"]) == len(rows) - 1


class TestProvenance:
    """Which file says who wrote it, and which deliberately does not."""

    def test_the_bindings_stamp_their_provenance_first(self):
        text = PACKAGE_PATHS.va_bindings.read_text(encoding="utf-8")
        assert json.loads(text)[PROVENANCE_KEY]
        first_key = next(iter(json.loads(text)))
        assert first_key == PROVENANCE_KEY

    def test_the_lattice_carries_no_provenance(self):
        """It stays a pure pyAT document; its identity is the bindings' digest."""
        saved = json.loads(PACKAGE_PATHS.lattice_json.read_text(encoding="utf-8"))
        assert PROVENANCE_KEY not in saved


class TestThePopulationIsTheFacilitySpec:
    """One binding per magnet and corrector device, two per monitor."""

    def test_the_document_describes_the_ring_it_was_built_for(self, document):
        assert document.energy_gev == pytest.approx(ALS_U_AR.energy_ev / 1e9)
        assert document.system

    def test_every_spec_device_carries_its_bindings(self, document):
        expected = Counter()
        for family in ALS_U_AR.families:
            expected[family.name] = family.count * (2 if family.kind == "monitor" else 1)
        assert Counter(binding.family for binding in document.bindings) == expected

    def test_the_claimed_addresses_are_the_coupled_namespace(self, document):
        claimed = set()
        for binding in document.bindings:
            claimed.add(binding.setpoint_address)
            if binding.readback_address is not None:
                claimed.add(binding.readback_address)
        magnets = sum(f.count for f in ALS_U_AR.families if f.kind != "monitor")
        monitors = sum(f.count for f in ALS_U_AR.families if f.kind == "monitor")
        assert len(claimed) == magnets * 2 + monitors * 2

    def test_every_claimed_address_is_seeded_by_the_scenario(self, document, machine_channels):
        unseeded = sorted(
            address
            for binding in document.bindings
            for address in (binding.setpoint_address, binding.readback_address)
            if address is not None and address not in machine_channels
        )
        assert not unseeded, (
            "a bound address the scenario seed does not state is a channel the "
            f"mock engine cannot serve: {unseeded[:10]}"
        )

    def test_the_demo_binds_no_energy_or_rf_knob(self, document):
        """The demo ring has neither: its dipoles are trim coils, its cavity fixed."""
        assert {binding.kind for binding in document.bindings} == {
            "strength",
            "kick",
            "monitor",
        }


class TestTheCalibrationsAreTheRingsPhysics:
    """Each straight line is the ring's own physics, stated as data."""

    def _nominal(self, machine_channels, address: str) -> float:
        return float(machine_channels[address]["value"])

    @pytest.mark.parametrize("family_name", ["QF", "QD", "QFA"])
    def test_a_quadrupole_scales_its_baked_gradient(
        self, family_name, by_setpoint, elements, machine_channels
    ):
        for device in range(1, ALS_U_AR.family(family_name).count + 1):
            address = f"SR:MAG:{family_name}:{device:02d}:CURRENT:SP"
            binding = by_setpoint[address]
            element = elements[f"{family_name}{device:02d}"]
            nominal = self._nominal(machine_channels, address)
            assert binding.attribute == "PolynomB"
            assert binding.index == 1
            assert isinstance(binding.calibration, Linear)
            assert binding.calibration.gain == pytest.approx(
                float(element.PolynomB[1]) / nominal, rel=_TOLERANCE
            )
            assert binding.calibration.offset == 0.0
            assert binding.nominal == pytest.approx(nominal)

    @pytest.mark.parametrize("family_name", ["SF", "SD", "SHF", "SHD"])
    def test_a_sextupole_scales_its_baked_strength(
        self, family_name, by_setpoint, elements, machine_channels
    ):
        for device in range(1, ALS_U_AR.family(family_name).count + 1):
            address = f"SR:MAG:{family_name}:{device:02d}:CURRENT:SP"
            binding = by_setpoint[address]
            element = elements[f"{family_name}{device:02d}"]
            nominal = self._nominal(machine_channels, address)
            assert binding.attribute == "PolynomB"
            assert binding.index == 2
            assert binding.calibration.gain == pytest.approx(
                float(element.PolynomB[2]) / nominal, rel=_TOLERANCE
            )
            assert binding.calibration.offset == 0.0

    def test_a_dipole_is_a_trim_coil_about_its_nominal(
        self, by_setpoint, elements, machine_channels
    ):
        """At nominal current the field error is exactly zero, by construction."""
        for device in range(1, ALS_U_AR.family("DIPOLE").count + 1):
            address = f"SR:MAG:DIPOLE:{device:02d}:CURRENT:SP"
            binding = by_setpoint[address]
            element = elements[f"DIPOLE{device:02d}"]
            nominal = self._nominal(machine_channels, address)
            slope = float(element.BendingAngle) / float(element.Length)
            assert binding.attribute == "PolynomB"
            assert binding.index == 0
            assert binding.calibration.gain == pytest.approx(slope / nominal, rel=_TOLERANCE)
            assert binding.calibration.offset == pytest.approx(-slope, rel=_TOLERANCE)
            at_nominal = binding.calibration.gain * nominal + binding.calibration.offset
            assert at_nominal == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize(("family_name", "plane"), [("HCM", 0), ("VCM", 1)])
    def test_a_corrector_kicks_its_own_plane(self, family_name, plane, by_setpoint):
        for device in range(1, ALS_U_AR.family(family_name).count + 1):
            binding = by_setpoint[f"SR:MAG:{family_name}:{device:02d}:CURRENT:SP"]
            assert binding.kind == "kick"
            assert binding.attribute == "KickAngle"
            assert binding.index == plane
            assert binding.calibration.gain == pytest.approx(1.0 / demo_build.AMPS_PER_RADIAN_KICK)
            assert binding.calibration.offset == 0.0

    def test_a_monitor_reads_its_axis_through_an_inverse(self, by_setpoint):
        for device in range(1, ALS_U_AR.family("BPM").count + 1):
            for axis in ("x", "y"):
                address = f"SR:DIAG:BPM:{device:02d}:POSITION:{axis.upper()}"
                binding = by_setpoint[address]
                assert binding.kind == "monitor"
                assert binding.attribute == axis
                assert binding.readback == "inverse"
                assert binding.readback_address is None
                assert binding.monitor_inverse is not None
                assert binding.nominal is None

    def test_a_written_binding_echoes_its_setpoint_on_a_readback(self, by_setpoint):
        for address, binding in by_setpoint.items():
            if binding.kind == "monitor":
                continue
            assert binding.readback == "identity"
            assert binding.readback_address == address.replace(":SP", ":RB")
            assert binding.monitor_inverse is None

    def test_every_device_writes_one_element_whole(self, by_setpoint):
        """The demo has no split device: one slice, carrying the whole value."""
        for binding in by_setpoint.values():
            assert len(binding.slices) == 1
            assert binding.slices[0].weight == 1.0
            assert binding.slices[0].element == binding.element
            assert binding.owner == binding.family

    def test_nothing_rescales_with_rigidity(self, by_setpoint):
        """The demo tree binds no energy knob, so no physics value ever moves."""
        assert {binding.energy_scaling for binding in by_setpoint.values()} == {"none"}


class TestTheGeneratorRefusesWhatItCannotState:
    def test_a_device_the_seed_states_no_nominal_for_is_refused(self, ring):
        with pytest.raises(demo_build.DemoModelError, match="nominal"):
            demo_build.build_demo_bindings(ring, "0" * 64, {})

    def test_a_name_the_ring_carries_twice_is_refused_as_ambiguous(self, ring, machine_channels):
        """A shared name is refused for being shared, not for being absent.

        The two want opposite repairs -- rename an element, or add one -- so a
        deck carrying a device's name twice may not be reported as carrying it
        none.
        """
        target = ALS_U_AR.device_name("", "QF", 1)
        doubled = [*ring, next(element for element in ring if element.FamName == target)]
        with pytest.raises(demo_build.DemoModelError, match=rf"2 elements named '{target}'"):
            demo_build.build_demo_bindings(doubled, "0" * 64, machine_channels)

    def test_the_document_it_builds_is_the_document_it_writes(self, ring, machine_channels):
        rebuilt = demo_build.build_demo_bindings(
            ring, demo_build.digest_of(PACKAGE_PATHS.lattice_json), machine_channels
        )
        text = PACKAGE_PATHS.va_bindings.read_text(encoding="utf-8")
        assert parse_bindings(json.loads(text)) == rebuilt
