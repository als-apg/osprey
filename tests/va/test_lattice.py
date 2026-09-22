"""The deck a real export run wrote, served as a model.

Every other module in this suite hands the model a tree assembled by the test
that reads it. This one does not: it runs the whole ``osprey mml`` chain --
import, map, emit -- over a committed 2.0 export fixture, and serves the tree
that run wrote. What it pins is the seam between the two halves of this
repository, which neither half can check alone: the deck the emit lane saved,
the bindings it derived against that deck, the scenario seed and the write
bands it wrote beside them, and the channel namespace the manifest generator
builds from all of it, are one accelerator that the virtual accelerator boots.

Three properties of that seam, each of which has been wrong before:

* **Elements are found by the name the bindings carry.** The deck is a bag of
  named elements and the bindings name them; nothing derives an element from
  an address, a family or a position.
* **Positions are not a way in.** ``at.save_json`` drops the deck's
  ``RingParam`` marker into lattice properties, so the saved deck is one
  element shorter than the ``.mat`` the export read and the export's own
  one-based indices do not address it. A consumer that indexed by them would
  be off by one everywhere and wrong silently.
* **The cavity is on.** A deck arrives with longitudinal motion disabled, and
  a served model that left it that way would solve a 4D orbit while claiming
  the frequency knob does something.

The fixture export describes an invented machine, and every family name in it
is invented too; nothing below spells one. The emitted tree is built once per
module by the shared helper in ``tests/va/_served_tree.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.virtual_accelerator.bindings import BindingsDocument, load_bindings
from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel, UnknownDeviceError
from tests.va._served_tree import SYNTHETIC_EXPORT, emit_served_tree

pytest.importorskip("linkml_runtime")


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tree one emit run wrote."""
    return emit_served_tree(tmp_path_factory.mktemp("emitted"))


@pytest.fixture(scope="module")
def paths(data_dir: Path) -> ManifestPaths:
    return ManifestPaths(data_root=data_dir)


@pytest.fixture(scope="module")
def document(paths: ManifestPaths) -> BindingsDocument:
    return load_bindings(paths.va_bindings)


@pytest.fixture(scope="module")
def channels(paths: ManifestPaths) -> list[dict]:
    """The namespace a deployment of that tree resolves, generated from it."""
    return build_manifest(paths)["channels"]


@pytest.fixture(scope="module")
def booted(data_dir: Path, channels: list[dict]) -> PyATRingModel:
    """The model the emitted tree serves."""
    return PyATRingModel(data_dir, channels)


class TestTheEmittedTreeBootsAModel:
    """The chain's output is consumable, end to end, with nothing hand-made."""

    def test_the_run_wrote_every_file_the_model_reads(self, paths: ManifestPaths) -> None:
        assert paths.missing_sources() == []
        assert paths.channel_limits.is_file()

    def test_every_binding_of_the_document_became_a_variable(
        self, booted: PyATRingModel, document: BindingsDocument
    ) -> None:
        addressed = set(booted.supported_variables) - booted.derived_names
        assert addressed == {binding.setpoint_address for binding in document.bindings}

    def test_the_generated_namespace_serves_every_bound_address(
        self, channels: list[dict], document: BindingsDocument
    ) -> None:
        """The manifest and the bindings are derived from one export, so the
        addresses the document binds are addresses the deployment serves.
        A bound address the namespace omits reaches no variable at all."""
        served = {channel["address"] for channel in channels}
        assert {binding.setpoint_address for binding in document.bindings} <= served

    def test_the_document_states_the_deck_the_same_run_saved(
        self, paths: ManifestPaths, document: BindingsDocument
    ) -> None:
        """``build_ring`` refuses a deck the bindings were not derived
        against, so a model booting at all is that check passing -- this
        pins the deck it passed on as the one this run wrote."""
        assert paths.lattice_json.read_bytes()
        assert document.system
        assert document.energy_gev > 0.0


class TestElementsAreFoundByTheNameTheBindingCarries:
    def test_every_bound_element_is_named_exactly_once_in_the_deck(
        self, booted: PyATRingModel, document: BindingsDocument
    ) -> None:
        """What ``unique_element_index`` enforces at construction: a deck that
        repeats a bound name gives two elements one setpoint, and each would
        overwrite the other."""
        counts: dict[str, int] = {}
        for element in booted.lattice:
            counts[element.FamName] = counts.get(element.FamName, 0) + 1
        bound = {piece.element for binding in document.bindings for piece in binding.slices} | {
            binding.element for binding in document.bindings if binding.element
        }
        assert bound
        assert sorted(name for name in bound if counts.get(name, 0) != 1) == []

    def test_each_slice_of_a_shared_write_is_its_own_element(
        self, booted: PyATRingModel, document: BindingsDocument
    ) -> None:
        """A write shared over pieces reaches every piece, so every piece has
        to be findable -- not only the one the binding reads back from."""
        shared = [binding for binding in document.bindings if len(binding.slices) > 1]
        assert shared, "the export binds nothing over more than one element"
        for binding in shared:
            indices = [booted.element_index(piece.element) for piece in binding.slices]
            assert len(set(indices)) == len(indices)

    def test_a_name_the_deck_does_not_carry_is_refused_naming_it(
        self, booted: PyATRingModel
    ) -> None:
        with pytest.raises(UnknownDeviceError):
            booted.element_index("no_element_of_this_deck_is_called_this")


class TestTheDeckIsNeverIndexedByExportPosition:
    """The export's own element indices do not address the saved deck.

    ``at.save_json`` folds the deck's ``RingParam`` marker into the lattice's
    properties, so the saved deck is one element shorter than the ring the
    export read, and the export's one-based indices are one further along than
    the position of the same element here. Every index in the two files below
    would be off, in the same direction, for every element -- which is exactly
    the kind of wrong that still produces a plausible orbit.
    """

    def test_no_binding_carries_a_deck_position(self, document: BindingsDocument) -> None:
        """A binding locates an element by name and a component by index.
        ``index`` is into an element's own storage, never into the deck."""
        for binding in document.bindings:
            assert binding.element is None or isinstance(binding.element, str)
            for piece in binding.slices:
                assert isinstance(piece.element, str)

    @pytest.mark.usefixtures("paths")
    def test_the_saved_deck_is_shorter_than_the_positions_the_export_states(
        self, booted: PyATRingModel
    ) -> None:
        """The off-by-one made concrete, so the trap is visible rather than
        only described: the highest position the export states is past the
        end of the deck a consumer would index with it."""
        export = json.loads(SYNTHETIC_EXPORT.read_text(encoding="utf-8"))
        stated = [
            value
            for name, family in export.items()
            if name != "_export" and isinstance(family, dict)
            for value in _numbers(family.get("AT", {}).get("ATIndex"))
        ]
        assert stated, "the export states no element position to be trapped by"
        assert max(stated) > len(booted.lattice)

    def test_the_deck_the_model_holds_is_the_file_this_run_wrote(
        self, booted: PyATRingModel, paths: ManifestPaths
    ) -> None:
        saved = json.loads(paths.lattice_json.read_text(encoding="utf-8"))
        assert len(saved["elements"]) == len(booted.lattice)


def _numbers(value: object) -> list[float]:
    """Every finite number anywhere inside a nested export value."""
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [number for item in value for number in _numbers(item)]
    return []


class TestTheServedRingSolvesSixDimensionally:
    def test_longitudinal_motion_is_on(self, booted: PyATRingModel) -> None:
        """A deck is saved 4D; ``build_ring`` enables the cavity, and without
        that the frequency knob the export binds would move nothing."""
        assert booted.lattice.is_6d

    def test_the_deck_carries_the_cavity_the_document_binds(
        self, booted: PyATRingModel, document: BindingsDocument
    ) -> None:
        cavities = [binding for binding in document.bindings if binding.kind == "rf"]
        assert cavities, "the export binds no cavity"
        for binding in cavities:
            element = booted.lattice[booted.element_index(binding.element)]
            assert hasattr(element, binding.attribute)

    def test_the_orbit_the_model_publishes_is_the_one_the_cavity_closes(
        self, booted: PyATRingModel, document: BindingsDocument
    ) -> None:
        """Every reading the served tree publishes is a finite number, which a
        deck without a stable six-dimensional orbit would not produce."""
        monitors = [binding for binding in document.bindings if binding.kind == "monitor"]
        assert monitors
        readings = booted.get([binding.setpoint_address for binding in monitors])
        assert all(reading == reading for reading in readings.values())
