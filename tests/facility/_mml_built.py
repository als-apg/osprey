"""The MML fixture trees, imported and built the way every facility test reads them.

The chain itself is framework-free and lives in the facility package's suite
(:mod:`osprey_facility_tests._mml_built`): the import of each tree under its
mapping, the widening of the limits records its build stops on, the in-memory
build, and the session fixtures that serve each once per run. This module
re-exports it for the root tests and adds the one part that needs the
framework: ``mml_built`` here hands each model with its wiring as the
simulator view gives it. ``tests/facility/conftest.py`` takes every session
fixture with its star import.

The builds run without the framework's extra checks (``NO_EXTRA_CHECKS``), so
``check_scenario_servers`` does not run on them. The MML trees carry no
scenario sessions today, so nothing is lost; a tree that gains one needs its
build checked with ``FRAMEWORK_CHECKS``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from osprey_facility.records import FacilityRecords
from osprey_facility_tests._mml_built import (
    BUILT,
    FIXTURES,
    REPO_ROOT,
    TREE_FIXTURES,
    TREES,
    WIDENED,
    MmlImport,
    build_tree,
    built_model,
    export_files,
    import_record,
    import_tree,
    load_yaml,
    mapped_facility,
    mml_imported,
    model_name,
    read_yaml,
    widen,
)
from osprey_facility_tests._mml_built import BuiltModel as _DescriptionModel

globals().update(TREE_FIXTURES)


@dataclass(frozen=True)
class BuiltModel(_DescriptionModel):
    """One model of a built fixture tree, with its wiring.

    Attributes:
        wiring: The model's wiring entries, as the simulator view gives them.
    """

    wiring: list[dict[str, Any]]


@pytest.fixture(scope="session")
def mml_built(request: pytest.FixtureRequest) -> Callable[[str, str], BuiltModel]:
    """Each fixture tree built once per run, served one model at a time, with its wiring.

    Returns a callable ``(tree, stem) -> BuiltModel``. A tree is imported and
    built on its first request; every model of it is read from that one build.
    Read-only: a test copies before writing and never changes the document.
    """
    from osprey.facility.views.simulator import simulator_wiring

    def model(tree: str, stem: str) -> BuiltModel:
        if f"mml_build_{tree}" not in TREE_FIXTURES:
            raise KeyError(f"{tree!r}: name the bands its build widens in WIDENED, {__file__}")
        built = built_model(request, tree, stem)
        return BuiltModel(
            tree=built.tree,
            stem=built.stem,
            document=built.document,
            facility=built.facility,
            record=built.record,
            wiring=simulator_wiring(FacilityRecords(built.document), built.name),
        )

    return model


__all__ = [
    "BUILT",
    "FIXTURES",
    "REPO_ROOT",
    "TREES",
    "WIDENED",
    "BuiltModel",
    "MmlImport",
    "build_tree",
    "export_files",
    "import_record",
    "import_tree",
    "load_yaml",
    "mapped_facility",
    "mml_built",
    "mml_imported",
    "model_name",
    "read_yaml",
    "widen",
    *TREE_FIXTURES,
]
