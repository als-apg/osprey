"""Tests for the served lattice loader (``lattice/ring.py``).

``build_ring`` is the one way the virtual accelerator acquires its lattice: it
reads the emitted tree it is handed -- a saved pyAT ring beside the bindings
document derived against it -- and hands back a ring the model can drive. The
tests below pin what that costs the tree rather than the implementation:

* the tree the emit lane writes boots, and the ring comes back with
  longitudinal motion on for the cavity alone -- radiation stays off, the way
  MML's own read path runs;
* a lattice the bindings were *not* derived against is refused, because every
  element name, index and nominal in the document was read off one particular
  ring;
* a bound element the lattice does not name exactly once is refused, because
  lume-pyat addresses elements by ``FamName`` and cannot say which of two
  ``BPM1`` elements a channel means.

The rings here are synthetic: eight elements with unique names, one cavity,
saved 4D. That is the shape a real emitted tree has in the two respects this
module cares about, and nothing here needs a facility's physics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import at
import pytest

from osprey.services.virtual_accelerator.bindings import BindingsError
from osprey.services.virtual_accelerator.lattice import build_ring
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths

# -- the synthetic emitted tree ----------------------------------------------


def _ring(
    monitor_name: str = "BPM1",
    corrector_names: tuple[str, str] = ("HCM1", "HCM2"),
) -> at.Lattice:
    """A small 4D ring with unique element names and one cavity.

    Args:
        monitor_name: FamName of the second monitor, so a caller can collide
            it with the first and make the lattice repeat a bound name.
        corrector_names: FamNames of the two corrector pieces a sliced kick
            binding writes.

    Returns:
        A lattice with longitudinal motion disabled, as a ring saved from
        MML's simulator model arrives.
    """
    first, second = corrector_names
    elements = [
        at.Drift("D1", 2.0),
        at.Quadrupole("QUAD1", 0.5, 1.2),
        at.Dipole("BEND1", 1.0, 0.1),
        at.Corrector(first, 0.0, [0.0, 0.0]),
        at.Corrector(second, 0.0, [0.0, 0.0]),
        at.Monitor("BPM0"),
        at.Monitor(monitor_name),
        at.RFCavity("RF1", 0.0, 1.0e6, 500.0e6, 32, 3.0e9),
    ]
    ring = at.Lattice(elements, name="probe", energy=3.0e9, periodicity=1)
    ring.disable_6d()
    return ring


def _linear(gain: float = 2.0, offset: float = 0.0) -> dict:
    return {"kind": "linear", "gain": gain, "offset": offset}


def _binding(**overrides) -> dict:
    """A strength binding on ``QUAD1``, overridable key by key."""
    body = {
        "kind": "strength",
        "family": "QUAD",
        "setpoint_address": "quad:sp",
        "readback_address": "quad:rb",
        "readback": "identity",
        "element": "QUAD1",
        "attribute": "PolynomB",
        "index": 1,
        "slices": [{"element": "QUAD1", "weight": 1.0}],
        "owner": "QUAD",
        "calibration": _linear(),
        "monitor_inverse": None,
        "nominal": 12.5,
        "energy_scaling": "brho",
        "energy_table": None,
    }
    body.update(overrides)
    return body


def _kick(**overrides) -> dict:
    """A kick shared over the two corrector pieces."""
    body = _binding(
        kind="kick",
        family="HCM",
        setpoint_address="hcm:sp",
        readback_address="hcm:rb",
        readback="inverse",
        element="HCM1",
        attribute="KickAngle",
        index=0,
        slices=[{"element": "HCM1", "weight": 0.5}, {"element": "HCM2", "weight": 0.5}],
        owner="HCM",
        monitor_inverse=_linear(gain=0.5),
    )
    body.update(overrides)
    return body


def _monitor(**overrides) -> dict:
    body = _binding(
        kind="monitor",
        family="BPM",
        setpoint_address="bpm:x",
        readback_address=None,
        readback="inverse",
        element="BPM1",
        attribute="x",
        index=None,
        slices=[{"element": "BPM1", "weight": 1.0}],
        owner="BPM",
        calibration=_linear(gain=1.0e-3),
        monitor_inverse=_linear(gain=1.0e3),
        nominal=None,
        energy_scaling="none",
    )
    body.update(overrides)
    return body


def _rf(**overrides) -> dict:
    body = _binding(
        kind="rf",
        family="RF",
        setpoint_address="rf:sp",
        readback_address=None,
        readback="same_as_setpoint",
        element="RF1",
        attribute="Frequency",
        index=None,
        slices=[{"element": "RF1", "weight": 1.0}],
        owner="RF",
        calibration=_linear(gain=1.0e6),
        nominal=499.64,
        energy_scaling="none",
    )
    body.update(overrides)
    return body


def _energy(**overrides) -> dict:
    """The lattice-level knob: it binds no element at all."""
    body = _binding(
        kind="energy",
        family="BEND",
        setpoint_address="bend:sp",
        readback_address="bend:rb",
        readback="identity",
        element=None,
        attribute=None,
        index=None,
        slices=[],
        owner=None,
        calibration=None,
        monitor_inverse=None,
        nominal=300.0,
        energy_scaling="none",
        energy_table={"kind": "table", "grid": [280.0, 300.0], "values": [2.8, 3.0]},
    )
    body.update(overrides)
    return body


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(
    root: Path,
    ring: at.Lattice | None = None,
    bindings: list[dict] | None = None,
    digest: str | None = None,
) -> ManifestPaths:
    """Write a served tree and return the paths that address it.

    Args:
        root: The data root to write ``simulation/`` under.
        ring: The lattice to save; a default synthetic ring when omitted.
        bindings: The document's bindings; one of each kind when omitted.
        digest: The digest to stamp, overriding the saved lattice's own --
            how a tree whose lattice and bindings disagree is made.

    Returns:
        The paths of the written tree.
    """
    paths = ManifestPaths(data_root=root)
    paths.lattice_json.parent.mkdir(parents=True, exist_ok=True)
    at.save_lattice(_ring() if ring is None else ring, paths.lattice_json)
    document = {
        "system": "StorageRing",
        "energy_gev": 3.0,
        "lattice_sha256": _sha256(paths.lattice_json) if digest is None else digest,
        "bindings": [_binding(), _kick(), _monitor(), _rf(), _energy()]
        if bindings is None
        else bindings,
    }
    paths.va_bindings.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return paths


def _refused(paths: ManifestPaths) -> BindingsError:
    with pytest.raises(BindingsError) as excinfo:
        build_ring(paths)
    return excinfo.value


# -- the boot path -----------------------------------------------------------


class TestBoot:
    """What the emitted tree gives the model when everything agrees."""

    def test_boots_the_emitted_tree(self, tmp_path):
        ring = build_ring(_tree(tmp_path))
        assert isinstance(ring, at.Lattice)
        assert [element.FamName for element in ring] == [
            "D1",
            "QUAD1",
            "BEND1",
            "HCM1",
            "HCM2",
            "BPM0",
            "BPM1",
            "RF1",
        ]
        assert ring.energy == pytest.approx(3.0e9)

    def test_turns_longitudinal_motion_on_for_the_cavity_alone(self, tmp_path):
        paths = _tree(tmp_path)
        assert not at.load_lattice(paths.lattice_json).is_6d, "the saved ring is 4D"

        ring = build_ring(paths)

        assert ring.is_6d
        cavities = [element for element in ring if isinstance(element, at.RFCavity)]
        assert [element.PassMethod for element in cavities] == ["RFCavityPass"]
        radiating = [element.FamName for element in ring if element.PassMethod.endswith("RadPass")]
        assert radiating == [], "radiation stays off, as MML's read path runs it"

    def test_builds_a_fresh_ring_each_call(self, tmp_path):
        paths = _tree(tmp_path)
        first = build_ring(paths)
        first[1].PolynomB[1] = 99.0
        assert build_ring(paths)[1].PolynomB[1] != 99.0

    def test_the_energy_knob_needs_no_element(self, tmp_path):
        """It is the one kind that binds nothing, so nothing is looked up."""
        ring = build_ring(_tree(tmp_path, bindings=[_energy()]))
        assert isinstance(ring, at.Lattice)
        assert ring.is_6d


# -- the lattice the bindings were derived against ---------------------------


class TestDigest:
    """Every name, index and nominal in the document was read off one ring."""

    def test_refuses_a_lattice_the_bindings_were_not_derived_against(self, tmp_path):
        paths = _tree(tmp_path, digest="b" * 64)
        error = _refused(paths)
        assert error.key == "lattice_sha256"
        assert "b" * 64 in str(error)
        assert str(paths.lattice_json) in str(error)

    def test_refuses_a_lattice_edited_after_the_bindings_were_emitted(self, tmp_path):
        paths = _tree(tmp_path)
        edited = _ring()
        edited[1].PolynomB[1] = 4.2
        at.save_lattice(edited, paths.lattice_json)
        assert _refused(paths).key == "lattice_sha256"

    def test_names_the_lattice_file_that_is_absent(self, tmp_path):
        paths = _tree(tmp_path)
        paths.lattice_json.unlink()
        with pytest.raises(FileNotFoundError) as excinfo:
            build_ring(paths)
        assert str(paths.lattice_json) in str(excinfo.value)

    def test_names_the_bindings_file_that_is_absent(self, tmp_path):
        paths = _tree(tmp_path)
        paths.va_bindings.unlink()
        with pytest.raises(FileNotFoundError) as excinfo:
            build_ring(paths)
        assert str(paths.va_bindings) in str(excinfo.value)


# -- one element per bound name ----------------------------------------------


class TestBoundElements:
    """lume-pyat addresses elements by FamName, so the name must be unique."""

    def test_refuses_a_bound_name_the_lattice_repeats(self, tmp_path):
        paths = _tree(
            tmp_path,
            ring=_ring(monitor_name="BPM0"),
            bindings=[_monitor(element="BPM0", slices=[{"element": "BPM0", "weight": 1.0}])],
        )
        error = _refused(paths)
        assert error.key == "bindings[0].slices[0].element"
        assert "BPM0" in error.message
        assert "2" in error.message
        assert "BPM" in str(error), "the refusal names the family"

    def test_refuses_a_bound_name_the_lattice_does_not_carry(self, tmp_path):
        paths = _tree(
            tmp_path,
            bindings=[_binding(element="QUAD9", slices=[{"element": "QUAD9", "weight": 1.0}])],
        )
        error = _refused(paths)
        assert error.key == "bindings[0].slices[0].element"
        assert "QUAD9" in error.message

    def test_checks_every_slice_not_just_the_one_read_back(self, tmp_path):
        paths = _tree(
            tmp_path,
            bindings=[
                _kick(
                    slices=[{"element": "HCM1", "weight": 0.5}, {"element": "HCM9", "weight": 0.5}]
                )
            ],
        )
        assert _refused(paths).key == "bindings[0].slices[1].element"

    def test_names_the_first_offender_in_document_order(self, tmp_path):
        paths = _tree(
            tmp_path,
            bindings=[
                _monitor(),
                _binding(element="QUAD9", slices=[{"element": "QUAD9", "weight": 1.0}]),
                _rf(element="RF9", slices=[{"element": "RF9", "weight": 1.0}]),
            ],
        )
        assert _refused(paths).key == "bindings[1].slices[0].element"

    def test_a_document_its_own_schema_refuses_never_reaches_the_census(self, tmp_path):
        """The schema runs first, so its key is the one a stale document reports."""
        paths = _tree(tmp_path, bindings=[_binding(index=-1)])
        assert _refused(paths).key == "bindings[0].index"
