"""The MML fixture trees, imported and built the way every facility test reads them.

A tree's exports are imported into a fresh ``data/facility/`` under the tree's
mapping, the limits records its build stops on are widened to the edge the
stop names, and the facility file is built in memory. :func:`build_tree` is
that chain as a plain function.

Two session fixtures serve the chain once per run. ``mml_imported`` hands a
test each tree's clean import (what it wrote and printed), and ``mml_built``
each tree's build, one model at a time: the facility file, the model's record,
its deck and its wiring as the simulator reads it. Under xdist the first worker
to ask for a tree imports or builds it, holding a lock, in a directory every
worker reads, and the others read its record (``tests._builds._built_once``),
so readers need no common xdist group. Tests read both and never write to
them; one that changes an import works on a copy.

The YAML a test reads from an import parses with the loader the facility
build reads it with (:func:`load_yaml`); :func:`read_yaml` serves a file
parsed once to every reader that only reads it.
"""

from __future__ import annotations

import contextlib
import io
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.records import FacilityRecords
from osprey_connectors.yaml_loader import safe_load
from tests._builds import _built_once
from tests.fixtures.mml._trees import SUPPORTED, names

REPO_ROOT = Path(__file__).resolve().parents[2]
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


@lru_cache(maxsize=32)
def _parsed(path: Path, _mtime_ns: int, _size: int) -> Any:
    return load_yaml(path)


def read_yaml(path: Path) -> Any:
    """``path`` parsed once for every reader of the same bytes.

    The parse is shared, so a caller only reads it; one that changes what it
    read takes :func:`load_yaml`. A file written again since its last parse is
    parsed afresh: the parse is kept by the file's modification time and size.
    """
    stat = path.stat()
    return _parsed(path, stat.st_mtime_ns, stat.st_size)


def mapped_facility(root: Path, tree: str) -> Path:
    """A fresh ``root/data/facility`` holding only the tree's mapping."""
    from osprey.facility.layers.mml.mapping import MAPPING_FILE

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
    from osprey.facility.layers.mml.importer import import_mml

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
    from osprey.facility.build import build_facility

    facility = import_tree(root, tree)
    widen(facility, WIDENED[tree])
    return build_facility(facility, project_name="demo"), facility


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
        wiring: The model's wiring entries, as the simulator view gives them.
    """

    tree: str
    stem: str
    document: dict[str, Any]
    facility: Path
    record: dict[str, Any]
    wiring: list[dict[str, Any]]

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
        from osprey.facility.layers.mml.importer import LAYER_DIR

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
    from osprey.facility.layers.mml.importer import import_mml

    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        facility = mapped_facility(root, tree)
        written = import_mml(export_files(tree), facility)
    return MmlImport(tree, facility, written, printed.getvalue())


def _import_fixture(tree: str) -> Any:
    """The session fixture holding ``tree``'s one import of the run."""

    def imported(tmp_path_factory: pytest.TempPathFactory) -> MmlImport:
        record: MmlImport = _built_once(
            tmp_path_factory,
            f"mml-import-{tree}",
            f"mml-import-{tree}.pickle",
            lambda parent: import_record(tree, parent),
        )
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
        def make(parent: Path) -> tuple[dict[str, Any], Path]:
            from osprey.facility.build import build_facility

            imported: MmlImport = request.getfixturevalue(f"mml_import_{tree}")
            facility = imported.copy(parent)
            widen(facility, WIDENED[tree])
            return build_facility(facility, project_name="demo"), facility

        made: tuple[dict[str, Any], Path] = _built_once(
            tmp_path_factory, f"mml-build-{tree}", f"mml-build-{tree}.pickle", make
        )
        return made

    return pytest.fixture(scope="session", name=f"mml_build_{tree}")(built)


# Under xdist each record is the run's one: the fixtures are named per tree, so
# every tree has its own lock and a worker waits on no tree it does not read.
mml_import_nsls2 = _import_fixture("nsls2")
mml_import_spear3 = _import_fixture("spear3")
mml_import_synthetic = _import_fixture("synthetic")
mml_build_nsls2 = _build_fixture("nsls2")
mml_build_spear3 = _build_fixture("spear3")


@pytest.fixture(scope="session")
def mml_imported(request: pytest.FixtureRequest) -> Callable[[str], MmlImport]:
    """Each fixture tree imported once per run, served whole.

    Returns a callable ``tree -> MmlImport``; a tree is imported on its first
    request. Tests read the import and never write to it.
    """

    def imported(tree: str) -> MmlImport:
        record: MmlImport = request.getfixturevalue(f"mml_import_{tree}")
        return record

    return imported


@pytest.fixture(scope="session")
def mml_built(request: pytest.FixtureRequest) -> Callable[[str, str], BuiltModel]:
    """Each fixture tree built once per run, served one model at a time.

    Returns a callable ``(tree, stem) -> BuiltModel``. A tree is imported and
    built on its first request; every model of it is read from that one build.
    Tests read the build and never write to it.
    """
    from osprey.facility.views.simulator import simulator_wiring

    def model(tree: str, stem: str) -> BuiltModel:
        document, facility = request.getfixturevalue(f"mml_build_{tree}")
        name = model_name(document, stem)
        (record,) = [entry for entry in document["models"] if entry["name"] == name]
        return BuiltModel(
            tree=tree,
            stem=stem,
            document=document,
            facility=facility,
            record=record,
            wiring=simulator_wiring(FacilityRecords(document), name),
        )

    return model
