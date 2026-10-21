"""The MML fixture trees, imported and built the way every facility test reads them.

A tree's exports are imported into a fresh ``data/facility/`` under the tree's
mapping, the limits records its build stops on are widened to the edge the
stop names, and the facility file is built in memory. :func:`build_tree` is
that chain as a plain function; the session-scoped :func:`mml_built` fixture
runs it once per tree and hands each test one model of it: the facility file,
the model's record, its deck and its wiring as the simulator reads it.

A module whose tests read ``mml_built`` joins the ``mml_built`` xdist group,
so the one build per tree serves every reader on one worker.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

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


def _load(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def widen(facility: Path, edges: dict[str, tuple[str, float]]) -> None:
    """Apply the ``seed-invalid`` remedy: widen each named limits record by hand."""
    path = facility / "limits.yaml"
    document = _load(path)
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


@pytest.fixture(scope="session")
def mml_built(tmp_path_factory: pytest.TempPathFactory) -> Callable[[str, str], BuiltModel]:
    """Each fixture tree built once per session, served one model at a time.

    Returns a callable ``(tree, stem) -> BuiltModel``. A tree is imported and
    built on its first request; every model of it is read from that one build.
    Tests read the build and never write to it.
    """
    from osprey.facility.views.simulator import simulator_wiring

    built: dict[str, tuple[dict[str, Any], Path]] = {}

    def model(tree: str, stem: str) -> BuiltModel:
        if tree not in built:
            built[tree] = build_tree(tree, tmp_path_factory.mktemp(f"mml-{tree}"))
        document, facility = built[tree]
        name = model_name(document, stem)
        (record,) = [entry for entry in document["models"] if entry["name"] == name]
        return BuiltModel(
            tree=tree,
            stem=stem,
            document=document,
            facility=facility,
            record=record,
            wiring=simulator_wiring(document, name),
        )

    return model
