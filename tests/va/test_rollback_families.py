"""A write the ring cannot take leaves the deck exactly where it was.

The model applies a batch, solves for the closed orbit, and -- if that orbit
is gone -- rolls the lattice back and raises. Rollback is the only thing
standing between an operator's bad number and a served accelerator that is
quietly no longer the machine it claims to be: a partial restore leaves every
later reading wrong while every value still looks plausible, and nothing
downstream can tell.

What makes that worth a test of its own rather than one case in the model's
own suite is the *footprint* of a write. A binding does not own an element; it
owns a component of one attribute of one element, and a shared write owns the
same component of several. So the question each check below asks is the same
one per binding kind the served document contains: after a batch this kind
took part in is rejected, is every piece of storage it reached back to the bit
it started at -- the component it wrote, the neighbours it did not, and each
slice of a write that was shared.

The tree is the one a real ``osprey mml`` run writes (built by the shared
helper in ``tests/va/_served_tree.py``): five binding kinds,
a kick shared over two elements, and an energy knob that rescales the
setpoints it has adopted, so the footprints here are an export's own rather
than a fixture author's guess. Nothing below names a family: the kinds, the
elements and the components all come out of the served document.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from osprey.services.virtual_accelerator.bindings import Binding, BindingsDocument, load_bindings
from osprey.services.virtual_accelerator.lattice.solve import OrbitSolveError
from osprey.services.virtual_accelerator.manifest import build_manifest
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths
from osprey.services.virtual_accelerator.model.pyat import PyATRingModel
from tests.va._served_tree import emit_served_tree

pytest.importorskip("linkml_runtime")

#: How far a probe escalates a setpoint looking for a value the closed orbit
#: cannot survive, and how far it is allowed to go before giving up. The
#: factor is searched rather than pinned: which number breaks a ring is the
#: ring's business, and a constant chosen against one deck says nothing about
#: another.
_ESCALATION = 10.0
_ESCALATION_CEILING = 1.0e12

#: A step small enough that any ring still solves, used to show the model is
#: still usable once a write has been rejected.
_GENTLE_STEP = 1.0e-6


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One emit run's served tree, shared by every test in the module."""
    return emit_served_tree(tmp_path_factory.mktemp("rollback"))


@pytest.fixture(scope="module")
def document(data_dir: Path) -> BindingsDocument:
    return load_bindings(ManifestPaths(data_root=data_dir).va_bindings)


@pytest.fixture(scope="module")
def channels(data_dir: Path) -> list[dict]:
    return build_manifest(ManifestPaths(data_root=data_dir))["channels"]


@pytest.fixture
def model(data_dir: Path, channels: list[dict]) -> PyATRingModel:
    """A fresh model per test: every test here deliberately breaks one."""
    return PyATRingModel(data_dir, channels)


def _writable_kinds(document: BindingsDocument) -> dict[str, Binding]:
    """One binding per kind of write the served document contains.

    Rollback restores what a write declared, and what a write declares is
    decided by its kind, so one representative of each is the whole surface.
    """
    chosen: dict[str, Binding] = {}
    for binding in document.bindings:
        if binding.is_writable:
            chosen.setdefault(binding.kind, binding)
    return chosen


def _touched_elements(document: BindingsDocument) -> list[str]:
    """Every element name any binding of the document reaches.

    Wider than the binding under test on purpose: a knob that rescales its
    adopted setpoints writes elements its own binding never names, and a
    rollback that restored only the batch's nominal footprint would leave
    those behind.
    """
    return sorted(
        {piece.element for binding in document.bindings for piece in binding.slices}
        | {binding.element for binding in document.bindings if binding.element}
    )


def _deck_state(model: PyATRingModel, names: list[str]) -> dict[str, dict[str, Any]]:
    """A deep copy of every attribute a binding could have written.

    Copied rather than referenced: the arrays are the lattice's own storage,
    and a reference would follow the write it is meant to be compared against.
    """
    state: dict[str, dict[str, Any]] = {}
    for name in names:
        element = model.lattice[model.element_index(name)]
        state[name] = {
            attribute: copy.deepcopy(getattr(element, attribute))
            for attribute in ("PolynomA", "PolynomB", "KickAngle", "Frequency", "Voltage")
            if hasattr(element, attribute)
        }
    return state


def _differences(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[str]:
    """Which element attributes moved, named so a failure says where."""
    moved = []
    for name, attributes in before.items():
        for attribute, value in attributes.items():
            other = after[name][attribute]
            same = value == other
            if not (same if isinstance(same, bool) else bool(same.all())):
                moved.append(f"{name}.{attribute}")
    return moved


def _reject(model: PyATRingModel, binding: Binding, names: list[str]) -> tuple[float, dict]:
    """Drive one setpoint until the ring loses its closed orbit.

    Escalates the value until a batch is refused, snapshotting before each
    attempt -- so what comes back describes the state the *refused* write
    found, not the state the search started from. The writes that succeed on
    the way there are accepted writes and are meant to stick; comparing
    against the pre-search deck would call those a rollback failure.

    Args:
        model: The model to drive. It is left where the refused write found
            it, which is the thing under test.
        binding: The setpoint to escalate.
        names: The element names to snapshot.

    Returns:
        The value the model was holding when the refused write arrived, and
        the deck state at that moment.

    Raises:
        AssertionError: The ceiling was reached with the orbit still closed,
            so no rollback was exercised and any "nothing moved" assertion
            after it would be vacuous.
    """
    address = binding.setpoint_address
    value = float(model.get(address)) or 1.0
    while abs(value) < _ESCALATION_CEILING:
        held = float(model.get(address))
        before = _deck_state(model, names)
        value *= _ESCALATION
        try:
            model.set({address: value})
        except OrbitSolveError:
            return held, before
    raise AssertionError(f"{address}: no value up to {_ESCALATION_CEILING} cost the closed orbit")


class TestARejectedWriteRestoresTheDeck:
    def test_every_writable_kind_leaves_the_whole_deck_where_it_was(
        self, document: BindingsDocument, data_dir: Path, channels: list[dict]
    ) -> None:
        """The main property, once per kind of write the document contains.

        Every attribute of every bound element is compared, not just the
        component the binding names: a restore that put the component back and
        left a neighbour of it moved would be just as wrong and just as
        invisible.
        """
        names = _touched_elements(document)
        for kind, binding in sorted(_writable_kinds(document).items()):
            model = PyATRingModel(data_dir, channels)

            _held, before = _reject(model, binding, names)

            assert _differences(before, _deck_state(model, names)) == [], kind

    def test_a_write_shared_over_several_elements_restores_each_of_them(
        self, document: BindingsDocument, model: PyATRingModel
    ) -> None:
        """A shared write reaches elements the binding does not read back
        from, and those are the ones a rollback keyed to the read-back element
        alone would leave behind."""
        shared = next(
            (
                binding
                for binding in document.bindings
                if binding.is_writable and len(binding.slices) > 1
            ),
            None,
        )
        assert shared is not None, "the export shares no write over several elements"
        names = [piece.element for piece in shared.slices]
        assert len(names) > 1

        _held, before = _reject(model, shared, names)

        assert _differences(before, _deck_state(model, names)) == []

    def test_the_model_serves_the_value_it_was_holding(
        self, document: BindingsDocument, data_dir: Path, channels: list[dict]
    ) -> None:
        """The refused value must not be retained either: a setpoint that kept
        it would report a machine state the lattice does not have."""
        for kind, binding in sorted(_writable_kinds(document).items()):
            model = PyATRingModel(data_dir, channels)
            held, _before = _reject(model, binding, [])
            assert float(model.get(binding.setpoint_address)) == held, kind


class TestTheModelSurvivesTheRefusal:
    def test_a_later_write_still_reaches_the_deck(
        self, document: BindingsDocument, model: PyATRingModel
    ) -> None:
        """A rejected batch is not a broken model: the next write solves and
        lands, which is what makes rollback a recovery rather than a stop."""
        binding = next(iter(sorted(_writable_kinds(document).items())))[1]
        _reject(model, binding, [])

        address = binding.setpoint_address
        held = float(model.get(address))
        target = held + (abs(held) * _GENTLE_STEP or _GENTLE_STEP)
        model.set({address: target})

        assert float(model.get(address)) == target

    def test_a_monitor_still_publishes_a_reading(
        self, document: BindingsDocument, model: PyATRingModel
    ) -> None:
        monitors = [binding for binding in document.bindings if binding.kind == "monitor"]
        assert monitors
        _reject(model, next(iter(sorted(_writable_kinds(document).items())))[1], [])

        readings = model.get([binding.setpoint_address for binding in monitors])

        assert all(reading == reading for reading in readings.values())


class TestTheCheckedSurfaceIsTheDocumentsOwn:
    def test_every_writable_binding_of_the_document_has_its_kind_covered(
        self, document: BindingsDocument
    ) -> None:
        """The per-kind loops above are total over the document rather than
        over a list restated here, so an export carrying a kind this one does
        not is covered the day it arrives."""
        covered = set(_writable_kinds(document))
        assert covered
        assert {binding.kind for binding in document.bindings if binding.is_writable} == covered

    def test_the_document_carries_more_than_one_kind_of_write(
        self, document: BindingsDocument
    ) -> None:
        """Guards the loops from being one case wearing a loop's clothes."""
        assert len(_writable_kinds(document)) > 1
