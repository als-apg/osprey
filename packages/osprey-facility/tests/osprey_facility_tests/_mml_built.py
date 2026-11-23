"""The MML fixture trees, imported and built the way the facility tests read them.

A tree's exports are imported into a fresh ``data/facility/`` under the tree's
mapping, the limits records its build stops on are widened to the edge the
stop names, and the facility file is built in memory. :func:`build_tree` is
that chain as a plain function.

Two session fixtures serve the chain once per run. ``mml_imported`` hands a
test each tree's clean import (what it wrote and printed), and ``mml_built``
each tree's build, one model at a time: the facility file, the model's record
and its deck. Under xdist the first worker to ask for a tree imports or builds
it, holding a lock, in a directory every worker reads, and the others read its
record (``_harness._once.built_once``), so readers need no common xdist group.
Both are read-only: a test copies before writing, and never changes a record it
was handed. Each tree has its own pair of session fixtures (``TREE_FIXTURES``),
made from ``SUPPORTED`` and ``WIDENED``.

Everything here is framework-free, but running it is not: the import checks
that each imported model's engine (``pyat``) is registered, and only the
framework registers it. So the fixtures are registered by the root suite:
``tests/facility/_mml_built.py`` re-exports this module, adds each model's
wiring as the simulator view gives it, and ``tests/facility/conftest.py`` takes
every session fixture with its star import. The MML data trees stay at the
repository's ``tests/fixtures/mml/``.
"""

from __future__ import annotations

import contextlib
import functools
import io
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey_connectors.yaml_loader import safe_load
from osprey_facility_tests._harness import NO_EXTRA_CHECKS, REPO_ROOT
from osprey_facility_tests._harness._once import built_once
from osprey_facility_tests._mml_trees import SUPPORTED, names

#: The MML data trees, which stay in the repository's root test tree.
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "mml"

#: Each supported tree's exports, by the stem every file of one export is named after.
TREES: dict[str, tuple[str, ...]] = {tree.name: tree.exports for tree in SUPPORTED}

#: The trees the build case runs to a clean exit and holds to a fingerprint.
BUILT = names("golden")

#: The setpoints each tree's build starts outside the band its export states,
#: each with the edge that widens its limits record to hold the build's
#: operating point. A wired setpoint starts where its calibration puts the
#: deck's strength. For spear3 that is the nominal the export states, outside
#: the export's own ``Range``. nsls2 widens nothing.
WIDENED: dict[str, dict[str, tuple[str, float]]] = {
    "spear3": {
        "09S-QD1:CurrSetpt": ("min_value", -60.0),
        "MS1-BDMT:CurrSetpt": ("max_value", 600.0),
    },
    "nsls2": {},
}


def load_yaml(path: Path) -> Any:
    """``path`` parsed afresh, with the loader the facility build reads YAML with."""
    return safe_load(path.read_text(encoding="utf-8"))


#: The ``data/facility`` directories of the run's shared imports and builds. They
#: are read-only (a test copies before writing), so their files never change.
_SHARED: set[Path] = set()


def read_yaml(path: Path) -> Any:
    """``path`` parsed as :func:`load_yaml` parses it, for a caller that only reads it.

    A file of a shared import or build is parsed once per run; any other file,
    such as one in a test's own tree, is parsed afresh on every call.
    """
    if any(path.is_relative_to(root) for root in _SHARED):
        return _parse_shared(path)
    return load_yaml(path)


@functools.cache
def _parse_shared(path: Path) -> Any:
    return load_yaml(path)


def mapped_facility(root: Path, tree: str) -> Path:
    """A fresh ``root/data/facility`` holding only the tree's mapping."""
    from osprey_facility.layers.mml.mapping import MAPPING_FILE

    facility = root / "data" / "facility"
    target = facility / MAPPING_FILE
    target.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / tree / MAPPING_FILE, target)
    return facility


def export_files(tree: str) -> list[Path]:
    """The AO file of each export of the tree, in its ``TREES`` order."""
    return [FIXTURES / tree / f"{stem}.ao.json" for stem in TREES[tree]]


def import_tree(root: Path, tree: str) -> Path:
    """Import the tree's exports under its mapping into ``root/data/facility``."""
    from osprey_facility.layers.mml.importer import import_mml

    facility = mapped_facility(root, tree)
    import_mml(export_files(tree), facility)
    return facility


def widen(facility: Path, edges: dict[str, tuple[str, float]]) -> None:
    """Apply the ``seed-invalid`` remedy: widen each named limits record by hand."""
    path = facility / "limits.yaml"
    document = load_yaml(path)
    for row in document["records"]:
        if row["address"] in edges:
            key, value = edges[row["address"]]
            row[key] = value
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def build_tree(tree: str, root: Path) -> tuple[dict[str, Any], Path]:
    """Import ``tree`` under ``root``, widen its named bands and build it in memory.

    Returns:
        The facility file ``build_facility`` returned and the ``data/facility``
        directory it was built from.
    """
    return _widened_build(tree, import_tree(root, tree))


def _widened_build(tree: str, facility: Path) -> tuple[dict[str, Any], Path]:
    """Widen ``tree``'s named bands in the imported ``facility`` and build it in memory."""
    from osprey_facility.build import build_facility

    widen(facility, WIDENED[tree])
    return build_facility(facility, project_name="demo", extra_checks=NO_EXTRA_CHECKS), facility


def model_name(document: dict[str, Any], stem: str) -> str:
    """The model an export stem ``<machine>.<submachine>`` was imported as."""
    submachine = stem.split(".", 1)[1]
    names = [str(model["name"]) for model in document["models"]]
    matches = [name for name in names if name.lower() == submachine.lower()]
    assert len(matches) == 1, f"{stem}: the build names models {names}"
    return matches[0]


@dataclass(frozen=True)
class BuiltModel:
    """One model of a built fixture tree.

    Attributes:
        tree: The fixture tree.
        stem: The export stem the model was imported from.
        document: The tree's facility file.
        facility: The ``data/facility`` directory it was built from.
        record: The model's record in the facility file.
    """

    tree: str
    stem: str
    document: dict[str, Any]
    facility: Path
    record: dict[str, Any]

    @property
    def name(self) -> str:
        """The model's name."""
        return str(self.record["name"])

    @property
    def deck(self) -> Path:
        """The deck file the build names for the model."""
        return self.facility / str(self.record["deck"])

    @property
    def settings(self) -> Any:
        """The model's settings, or ``None``."""
        return self.record.get("settings")


@dataclass(frozen=True)
class MmlImport:
    """A tree's exports imported into a fresh ``data/facility``.

    Attributes:
        tree: The fixture tree.
        facility: The ``data/facility`` directory the import wrote.
        written: Each file the import returned, in its order.
        printed: What the import printed.
    """

    tree: str
    facility: Path
    written: list[Path]
    printed: str

    def rows(self, name: str) -> Any:
        """The layer's record file ``name``, parsed once for every reader."""
        from osprey_facility.layers.mml.importer import LAYER_DIR

        return read_yaml(self.facility / LAYER_DIR / name)

    def load(self, relative: str) -> Any:
        """The file at ``relative`` under ``facility``, parsed once for every reader."""
        return read_yaml(self.facility / relative)

    def copy(self, root: Path) -> Path:
        """The imported facility copied to ``root/data/facility``, for a case that changes it."""
        facility = root / "data" / "facility"
        shutil.copytree(self.facility, facility)
        return facility


def import_record(tree: str, root: Path) -> MmlImport:
    """Import ``tree`` under ``root`` as :func:`import_tree` does, keeping what it printed."""
    from osprey_facility.layers.mml.importer import import_mml

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        facility = mapped_facility(root, tree)
        written = import_mml(export_files(tree), facility)
    return MmlImport(tree, facility, written, printed.getvalue())


def _import_fixture(tree: str) -> Any:
    """The session fixture holding ``tree``'s one import of the run."""

    def imported(tmp_path_factory: pytest.TempPathFactory) -> MmlImport:
        """The tree's one import of the run. Read-only: copy before writing."""
        record: MmlImport = built_once(
            tmp_path_factory,
            f"mml-import-{tree}",
            f"mml-import-{tree}.pickle",
            lambda parent: import_record(tree, parent),
        )
        _SHARED.add(record.facility)
        return record

    return pytest.fixture(scope="session", name=f"mml_import_{tree}")(imported)


def _build_fixture(tree: str) -> Any:
    """The session fixture holding ``tree``'s one build of the run.

    The build is :func:`build_tree` run over a copy of the tree's one import:
    the import writes the same bytes wherever it is made, so the copy is the
    facility a fresh import there would be.
    """

    def built(
        tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
    ) -> tuple[dict[str, Any], Path]:
        """The tree's one build of the run. Read-only: copy before writing."""

        def make(parent: Path) -> tuple[dict[str, Any], Path]:
            imported: MmlImport = request.getfixturevalue(f"mml_import_{tree}")
            return _widened_build(tree, imported.copy(parent))

        made: tuple[dict[str, Any], Path] = built_once(
            tmp_path_factory, f"mml-build-{tree}", f"mml-build-{tree}.pickle", make
        )
        _SHARED.add(made[1])
        return made

    return pytest.fixture(scope="session", name=f"mml_build_{tree}")(built)


#: The per-tree session fixtures, by name: the import of every tree ``SUPPORTED``
#: imports, and the build of every tree ``WIDENED`` names. Under xdist each is
#: the run's one record, with its own lock, so a worker waits on no tree it does
#: not read.
TREE_FIXTURES: dict[str, Any] = {
    **{f"mml_import_{tree}": _import_fixture(tree) for tree in names("imports")},
    **{f"mml_build_{tree}": _build_fixture(tree) for tree in WIDENED},
}
globals().update(TREE_FIXTURES)


@pytest.fixture(scope="session")
def mml_imported(request: pytest.FixtureRequest) -> Callable[[str], MmlImport]:
    """Each fixture tree imported once per run, served whole.

    Returns a callable ``tree -> MmlImport``; a tree is imported on its first
    request. Read-only: a test copies before writing (:meth:`MmlImport.copy`)
    and never changes a parse it was handed.
    """

    def imported(tree: str) -> MmlImport:
        if f"mml_import_{tree}" not in TREE_FIXTURES:
            raise KeyError(
                f"{tree!r} is no tree SUPPORTED imports (osprey_facility_tests/_mml_trees.py)"
            )
        record: MmlImport = request.getfixturevalue(f"mml_import_{tree}")
        return record

    return imported


@pytest.fixture(scope="session")
def mml_built(request: pytest.FixtureRequest) -> Callable[[str, str], BuiltModel]:
    """Each fixture tree built once per run, served one model at a time.

    Returns a callable ``(tree, stem) -> BuiltModel``. A tree is imported and
    built on its first request; every model of it is read from that one build.
    Read-only: a test copies before writing and never changes the document.
    """

    def model(tree: str, stem: str) -> BuiltModel:
        if f"mml_build_{tree}" not in TREE_FIXTURES:
            raise KeyError(f"{tree!r}: name the bands its build widens in WIDENED, {__file__}")
        return built_model(request, tree, stem)

    return model


def built_model(request: pytest.FixtureRequest, tree: str, stem: str) -> BuiltModel:
    """The model imported from ``stem`` in ``tree``'s one build of the run."""
    document, facility = request.getfixturevalue(f"mml_build_{tree}")
    name = model_name(document, stem)
    (record,) = [entry for entry in document["models"] if entry["name"] == name]
    return BuiltModel(tree=tree, stem=stem, document=document, facility=facility, record=record)


#: What a ``conftest.py`` takes with its star import: every session fixture here.
__all__ = ["mml_built", "mml_imported", *TREE_FIXTURES]
