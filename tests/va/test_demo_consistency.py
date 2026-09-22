"""The demo ring and every Middle Layer export, held to one bindings contract.

Four trees serve a model in this repository: the demo ring the control
assistant ships, whose bindings a generator writes from a lattice it also
writes, and one per committed 2.0 Middle Layer export, whose bindings the emit
lane derives from what a facility stated about itself. They are meant to be
the same kind of document -- the served process reads all four with the same
loader and drives them through the same model layer -- so a difference between
them is either a fact about a machine or a place the generator and the emitter
drifted apart.

This module says which. It reads each tree's bindings into one *shape*: which
binding kinds are present, what shape of calibration each kind carries, how
each is read back, how many elements each drives and with what weights, which
halves of a device the manifest's pyat-coupled partition would claim, and
which setpoints the bands leave writable. The shapes are printed as a report
(``pytest -s``), and the contract every tree keeps is asserted.

What is asserted is only what a bindings document owes whatever wrote it: a
vocabulary that closes, a monitor that is read back through its own inverse
and never written, a driven setpoint that is banded, an energy knob that binds
no element, and a slice weight that is a usable factor. Everything else is
printed and not judged, because a demo ring genuinely has no dipole ramp, no
series supply and no facility conversion table, and a test that called those
differences failures would be asserting that a toy is a storage ring.

The report is the deliverable: read the differences section against the demo
generator and decide, per line, whether it is a fact about the demo ring or
something the generator owes the emitter.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from osprey.services.virtual_accelerator.bindings import (
    BINDING_KINDS,
    ENERGY_SCALINGS,
    READBACK_RULES,
    Linear,
    Table,
)
from tests.templates.test_channel_limits_va import (
    FIXTURES,
    Tree,
    _demo_tree,
    emit_export_tree,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from osprey.services.virtual_accelerator.bindings import Binding

#: The name the demo tree is reported under.
DEMO = "demo"

#: The shape keys the report prints, in the order it prints them.
SHAPE_KEYS = (
    "kinds",
    "calibration",
    "readback",
    "energy_scaling",
    "readout",
    "slices",
    "weights",
    "halves",
    "bands",
    "nominals",
)

#: What a calibration may be. ``None`` is the energy knob, which converts
#: through its own table rather than through a calibration.
CALIBRATION_SHAPES = (Linear, Table)


def _shape_of(value: object) -> str:
    """The name of a calibration's shape, or ``none`` where there is none."""
    return "none" if value is None else type(value).__name__.lower()


def _half(binding: Binding) -> str:
    """Which halves of a device this binding claims in the coupled partition.

    The rule is the manifest's own: a monitor claims one address under the
    transverse axis it reads, a binding serving both halves on one address
    claims that address as the setpoint, and everything else claims a setpoint
    and a readback.
    """
    if binding.kind == "monitor":
        return (binding.attribute or "?").upper()
    return "SP" if binding.readback_address is None else "SP+RB"


def _readout_keys(binding: Binding) -> tuple[str, ...]:
    """Which readout calibrations a monitor states, by name."""
    if binding.readout is None:
        return ()
    return tuple(sorted(name for name, value in vars(binding.readout).items() if value is not None))


@dataclass(frozen=True)
class Shape:
    """One tree's bindings, counted by the facts a reader compares trees on."""

    name: str
    system: str
    energy_gev: float
    bindings: int
    counts: dict[str, Counter[tuple[str, ...]]]

    def kinds(self) -> set[str]:
        return {key[0] for key in self.counts["kinds"]}

    def lines(self) -> Iterator[str]:
        yield f"{self.name}: system {self.system}, {self.energy_gev} GeV, {self.bindings} bindings"
        for key in SHAPE_KEYS:
            for entry, count in sorted(self.counts[key].items()):
                yield f"    {key:<14} {' '.join(entry):<34} {count}"


def _shape(name: str, tree: Tree) -> Shape:
    counts: dict[str, Counter[tuple[str, ...]]] = {key: Counter() for key in SHAPE_KEYS}
    for binding in tree.document.bindings:
        kind = binding.kind
        counts["kinds"][(kind,)] += 1
        counts["calibration"][
            (
                kind,
                f"setpoint={_shape_of(binding.calibration)}",
                f"inverse={_shape_of(binding.monitor_inverse)}",
            )
        ] += 1
        counts["readback"][(kind, binding.readback)] += 1
        counts["energy_scaling"][(kind, binding.energy_scaling)] += 1
        counts["readout"][(kind, ",".join(_readout_keys(binding)) or "none")] += 1
        counts["slices"][(kind, f"{len(binding.slices)} elements")] += 1
        counts["weights"][
            (kind, "unit" if all(s.weight == 1.0 for s in binding.slices) else "scaled")
        ] += 1
        counts["halves"][(kind, _half(binding))] += 1
        low, high = tree.band(binding.setpoint_address)
        counts["bands"][
            (
                kind,
                "banded" if low is not None and high is not None else "unbanded",
                "writable" if tree.writable(binding.setpoint_address) else "read-only",
            )
        ] += 1
        counts["nominals"][(kind, "stated" if binding.nominal is not None else "none")] += 1
    return Shape(
        name=name,
        system=tree.document.system,
        energy_gev=tree.document.energy_gev,
        bindings=len(tree.document.bindings),
        counts=counts,
    )


@pytest.fixture(scope="module")
def shapes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Shape]:
    """The demo tree and one tree per committed 2.0 export, shaped once.

    Every export is emitted, not just one per fixture directory, so a facility
    committing a transfer line beside its ring is reported as the two machines
    it is. Emitting a storage ring is seconds of work, so the trees are built
    once for the module and read by every case.
    """
    root = tmp_path_factory.mktemp("consistency")
    built = {DEMO: _shape(DEMO, _demo_tree())}
    for directory in sorted(FIXTURES.iterdir()):
        if not directory.is_dir():
            continue
        for export in sorted(directory.glob("*.va.json")):
            stem = export.name[: -len(".va.json")]
            out = root / stem
            out.mkdir(parents=True)
            built[stem] = _shape(stem, emit_export_tree(directory.name, directory, stem, out))
    return built


@pytest.fixture(scope="module")
def trees(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Tree]:
    """The same trees, kept whole for the cases that read a binding itself."""
    root = tmp_path_factory.mktemp("consistency-trees")
    built = {DEMO: _demo_tree()}
    for directory in sorted(FIXTURES.iterdir()):
        if not directory.is_dir():
            continue
        for export in sorted(directory.glob("*.va.json")):
            stem = export.name[: -len(".va.json")]
            out = root / stem
            out.mkdir(parents=True)
            built[stem] = emit_export_tree(directory.name, directory, stem, out)
    return built


def _derived(shapes: dict[str, Shape]) -> list[Shape]:
    return [shape for name, shape in sorted(shapes.items()) if name != DEMO]


# ===================================================================
# The report
# ===================================================================


def test_the_consistency_report(shapes: dict[str, Shape]) -> None:
    """Print every tree's shape and the lines on which the demo stands alone.

    The report is printed and the difference count is asserted to be zero: the
    demo's generator states what every export states, so the demo is the same
    shape as a facility and a reader has one document to learn, not two.
    """
    demo = shapes[DEMO]
    derived = _derived(shapes)
    assert derived, "no committed export produced a tree to compare the demo with"

    print("\n=== bindings shape, tree by tree ===")
    for shape in (demo, *derived):
        for line in shape.lines():
            print(line)

    print("\n=== where the demo stands alone ===")
    alone: list[str] = []
    for key in SHAPE_KEYS:
        demo_entries = set(demo.counts[key])
        shared = set.intersection(*(set(shape.counts[key]) for shape in derived))
        every = set.union(*(set(shape.counts[key]) for shape in derived))
        alone += [
            f"{key}: every export states {' '.join(entry)}; the demo states none"
            for entry in sorted(shared - demo_entries)
        ]
        alone += [
            f"{key}: the demo states {' '.join(entry)}; no export does"
            for entry in sorted(demo_entries - every)
        ]
    for line in alone:
        print(f"    {line}")
    if not alone:
        print("    nothing: the demo's shape is inside the range the exports cover")

    print("\n=== facts, not drift ===")
    for shape in derived:
        missing = shape.kinds() - demo.kinds()
        extra = demo.kinds() - shape.kinds()
        print(f"    {shape.name}: binds {sorted(missing) or 'nothing'} the demo does not", end="")
        print(f"; the demo binds {sorted(extra) or 'nothing'} it does not")

    assert alone == [], "\n".join(["the demo's shape differs from every export:", *alone])


def test_the_demo_states_what_an_export_states_about_each_binding(
    trees: dict[str, Tree],
) -> None:
    """The two facts the demo's generator used to leave out, named directly.

    They were the report's whole difference list, so failing on the report
    alone would say "something moved" where these say what. Asserted of the
    demo, not of the exports: a facility states what its own magnets do, and
    a skew quadrupole setting an angle is rightly not rigidity-scaled.
    """
    demo = trees[DEMO]

    monitors = [b for b in demo.document.bindings if b.kind == "monitor"]
    strengths = [b for b in demo.document.bindings if b.kind == "strength"]
    assert monitors and strengths

    for binding in monitors:
        assert binding.nominal is not None, (
            f"{binding.family} {binding.setpoint_address} states no reading to start at"
        )
    for binding in strengths:
        assert binding.energy_scaling == "brho", f"{binding.family} {binding.setpoint_address}"


# ===================================================================
# The contract every tree keeps
# ===================================================================


def test_every_tree_spells_its_vocabulary_the_same_way(shapes: dict[str, Shape]) -> None:
    """Kind, readback rule and energy scaling come from the closed sets."""
    for name, shape in sorted(shapes.items()):
        for kind, *_ in shape.counts["kinds"]:
            assert kind in BINDING_KINDS, f"{name}: {kind}"
        for kind, rule in shape.counts["readback"]:
            assert rule in READBACK_RULES, f"{name}: {kind} reads back as {rule}"
        for kind, scaling in shape.counts["energy_scaling"]:
            assert scaling in ENERGY_SCALINGS, f"{name}: {kind} scales as {scaling}"


def test_a_monitor_is_read_through_its_own_inverse_and_never_written(
    trees: dict[str, Tree],
) -> None:
    """A monitor measures, so it carries an inverse and no band to write into.

    A tree may serve none at all: a transfer line whose beam monitors bind no
    lattice element is served with those elements as plain markers, and reads
    nothing back through the model.
    """
    for name, tree in sorted(trees.items()):
        for binding in tree.document.bindings:
            if binding.kind != "monitor":
                continue
            where = f"{name}: {binding.family} {binding.setpoint_address}"
            assert binding.readback == "inverse", where
            assert isinstance(binding.monitor_inverse, CALIBRATION_SHAPES), where
            assert not tree.writable(binding.setpoint_address), where
            assert tree.band(binding.setpoint_address) == (None, None), where
            assert (binding.attribute or "").upper() in ("X", "Y"), where


def test_a_driven_setpoint_is_writable_and_banded(trees: dict[str, Tree]) -> None:
    """Whatever a tree drives, a write to it is bounded by both edges."""
    for name, tree in sorted(trees.items()):
        for binding in tree.document.bindings:
            if binding.kind == "monitor":
                continue
            where = f"{name}: {binding.family} {binding.setpoint_address}"
            low, high = tree.band(binding.setpoint_address)
            assert tree.writable(binding.setpoint_address), where
            assert low is not None and high is not None, where
            assert low < high, f"{where}: band [{low}, {high}] is not an interval"


def test_the_energy_knob_binds_no_element_and_everything_else_binds_one(
    trees: dict[str, Tree],
) -> None:
    """Energy is the lattice-level knob; every other kind writes an element."""
    for name, tree in sorted(trees.items()):
        for binding in tree.document.bindings:
            where = f"{name}: {binding.family} {binding.setpoint_address}"
            if binding.kind == "energy":
                assert binding.slices == (), where
                assert binding.calibration is None, where
                assert binding.energy_table is not None, where
            else:
                assert binding.slices, where
                assert binding.energy_table is None, where
                assert isinstance(binding.calibration, CALIBRATION_SHAPES), where


def test_every_slice_weight_is_a_factor_a_write_can_be_scaled_by(
    trees: dict[str, Tree],
) -> None:
    """A weight is any finite non-zero factor, one supply's string included."""
    for name, tree in sorted(trees.items()):
        for binding in tree.document.bindings:
            for position, part in enumerate(binding.slices):
                where = f"{name}: {binding.family} {binding.setpoint_address}[{position}]"
                assert math.isfinite(part.weight), where
                assert part.weight != 0.0, where


def test_a_tree_serves_each_address_once(trees: dict[str, Tree]) -> None:
    """One binding per setpoint address, a series supply's string included."""
    for name, tree in sorted(trees.items()):
        seen = Counter(binding.setpoint_address for binding in tree.document.bindings)
        assert [address for address, count in seen.items() if count > 1] == [], name


def test_the_fixture_trees_cover_more_than_the_demo(shapes: dict[str, Shape]) -> None:
    """The comparison is worth running: some export binds what the demo cannot.

    A demo ring with one magnet per supply and no dipole ramp is the narrow
    case by construction. If this ever stops holding, the exports stopped
    carrying the machinery the emit lane was written for.
    """
    demo = shapes[DEMO]
    covered: set[str] = set()
    for shape in _derived(shapes):
        covered |= shape.kinds()
    assert covered - demo.kinds(), "no export binds a kind the demo does not"


def _fixture_directories() -> list[Path]:
    return [path for path in sorted(FIXTURES.iterdir()) if path.is_dir()]


def test_every_committed_export_is_in_the_report(shapes: dict[str, Shape]) -> None:
    """Every ``*.va.json`` in the fixtures is one tree of the report."""
    expected = {
        export.name[: -len(".va.json")]
        for directory in _fixture_directories()
        for export in directory.glob("*.va.json")
    }
    assert expected, "no fixture export carries a *.va.json sibling"
    assert set(shapes) == expected | {DEMO}
