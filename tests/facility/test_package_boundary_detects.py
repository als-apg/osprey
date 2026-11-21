"""The package-boundary checks fail on the edges they exist to catch.

``test_package_boundary`` passes on today's tree; this module proves it, and the
import-linter contracts behind it, would fail on a forbidden edge. Nothing here touches
``src/``:

* **Static imports** run the six shipped contracts, parsed by import-linter's own reader
  from ``pyproject.toml``, against an import graph of the real tree built once per session,
  with one edge added (or removed) in memory. Each case names the contract that must break.
  A hand-written contract would prove nothing: the point is that the shipped lists bite
  through import-linter's overlap skip (success criterion C6).
* **Dynamic reaches and third-party imports** (A1, A2) are invisible to import-linter, so
  they run the scan of ``test_package_boundary`` on a temporary ``src`` tree.

Both boundary files together stay under 10 s (SM-3). The contract classes and the option
reader are import-linter internals; the dev pin (``<3``) holds their shape.
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Mapping
from pathlib import Path

import grimp
import pytest
from importlinter.application import use_cases
from importlinter.configuration import configure
from importlinter.contracts.forbidden import ForbiddenContract
from importlinter.contracts.protected import ProtectedContract
from importlinter.domain.helpers import MissingImport

from tests.facility.test_package_boundary import (
    DESCRIPTION,
    DYNAMIC_RATCHET,
    FACILITY_CONTRACTS,
    FRAMEWORK,
    PANELS,
    REPO,
    THIRD_PARTY_RATCHET,
    crossed_row,
    dynamic_edges,
    read_sides,
    third_party_edges,
)

# --- the shipped contracts on the real graph ---------------------------------------------

#: The contract types the facility contracts use, as import-linter registers them.
_CONTRACT_TYPES = {"forbidden": ForbiddenContract, "protected": ProtectedContract}


class Shipped:
    """The facility contracts of ``pyproject.toml`` and the real import graph."""

    def __init__(self) -> None:
        configure()
        options = use_cases.read_user_options(str(REPO / "pyproject.toml"))
        self.contracts = {}
        for contract_options in options.contracts_options:
            if contract_options["id"] not in FACILITY_CONTRACTS:
                continue
            contract_class = _CONTRACT_TYPES[contract_options["type"]]
            self.contracts[contract_options["id"]] = contract_class(
                name=contract_options["name"],
                session_options=options.session_options,
                contract_options=contract_options,
            )
        missing = set(FACILITY_CONTRACTS) - self.contracts.keys()
        assert not missing, f"pyproject.toml has no import-linter contract {sorted(missing)}"
        self.graph = grimp.build_graph(*options.session_options["root_packages"], cache_dir=None)

    def broken(self, graph: grimp.ImportGraph) -> set[str]:
        """The ids of the facility contracts ``graph`` breaks."""
        return {
            cid
            for cid, contract in self.contracts.items()
            if not contract.check(copy.deepcopy(graph), verbose=False).kept
        }

    def with_edge(self, importer: str, imported: str) -> grimp.ImportGraph:
        graph = copy.deepcopy(self.graph)
        for module in (importer, imported):
            assert module in graph.modules, f"{module} is not in the real tree any more"
        graph.add_import(importer=importer, imported=imported, line_number=1, line_contents="")
        return graph


@pytest.fixture(scope="session")
def shipped() -> Shipped:
    return Shipped()


def test_the_real_tree_keeps_every_facility_contract(shipped: Shipped) -> None:
    """The control: without an injected edge nothing breaks, so a break is the edge's."""
    assert shipped.broken(shipped.graph) == set()


@pytest.mark.parametrize(
    ("importer", "imported", "contract"),
    [
        pytest.param(
            "osprey.facility.build",
            "osprey.cli.facility_cmd",
            "facility-description-standalone",
            id="S1-description-to-framework",
        ),
        pytest.param(
            "osprey.facility.build", "osprey", "facility-description-standalone", id="S1-to-root"
        ),
        pytest.param(
            "osprey.facility.compute",
            "osprey.simulation.engines.pyat",
            "facility-description-standalone",
            id="S1-description-to-engines",
        ),
        pytest.param(
            "osprey.facility.build",
            "osprey.facility.panels.emit",
            "facility-description-without-panels",
            id="S2-description-to-panels",
        ),
        pytest.param(
            "osprey.facility.panels.emit",
            "osprey.facility.records",
            "panels-document-only",
            id="S3-panels-to-description",
        ),
        pytest.param(
            "osprey.facility.panels.emit",
            "osprey.cli",
            "panels-standalone",
            id="S4-panels-to-framework",
        ),
        pytest.param(
            "osprey.facility.panels.emit",
            "osprey_connectors.config",
            "panels-standalone",
            id="S4-panels-to-sibling-osprey_connectors",
        ),
        pytest.param(
            "osprey.cli.main",
            "osprey.facility.panels",
            "framework-without-panels",
            id="S5-new-framework-import-of-panels",
        ),
        pytest.param(
            "osprey",
            "osprey.facility.panels.emit",
            "framework-without-panels",
            id="S5-root-to-panels",
        ),
        pytest.param(
            "osprey.cli.main",
            "osprey.facility.compute",
            "facility-public-api",
            id="S6-framework-to-non-public",
        ),
        pytest.param(
            "osprey.cli.main",
            "osprey.facility.layers.relations.claims",
            "facility-public-api",
            id="S6-framework-to-non-public-wildcard",
        ),
        pytest.param(
            "osprey.facility.render",
            "osprey.facility.views.simulator",
            "facility-description-standalone",
            id="CC5-new-edge-in-a-module-with-ratchet-entries",
        ),
    ],
)
def test_a_new_static_import_breaks_its_contract(
    shipped: Shipped, importer: str, imported: str, contract: str
) -> None:
    assert shipped.broken(shipped.with_edge(importer, imported)) == {contract}, (
        f"{importer} -> {imported} must break {contract} and only it"
    )


def test_an_import_of_a_public_description_module_breaks_nothing(shipped: Shipped) -> None:
    """The control for S6: the framework may use the description's public modules."""
    assert shipped.broken(shipped.with_edge("osprey.cli.main", "osprey.facility.render")) == set()


@pytest.mark.parametrize(
    ("importer", "imported", "contract"),
    [
        pytest.param(
            "osprey.facility.render",
            "osprey.facility.views",
            "facility-description-standalone",
            id="S1",
        ),
        pytest.param(
            "osprey.facility.panels.emit",
            "osprey_connectors.safety_file",
            "panels-standalone",
            id="S4",
        ),
        pytest.param(
            "osprey.cli.facility_cmd",
            "osprey.facility.panels",
            "framework-without-panels",
            id="S5",
        ),
    ],
)
def test_deleting_a_ratchet_edge_without_its_entry_fails(
    shipped: Shipped, importer: str, imported: str, contract: str
) -> None:
    graph = copy.deepcopy(shipped.graph)
    graph.remove_import(importer=importer, imported=imported)
    with pytest.raises(MissingImport, match=f"{importer} -> {imported}"):
        shipped.contracts[contract].check(graph, verbose=False)


# --- the scan on a temporary tree ----------------------------------------------------------


def tree(root: Path, files: Mapping[str, str]) -> Path:
    """A ``src`` tree under ``root`` holding ``files``, every directory a package.

    The scan is cached by its source root and pytest reuses the directory of a passed
    test, so each tree gets a name of its own.
    """
    src = root / uuid.uuid4().hex / "src"
    for name, text in files.items():
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        for parent in path.relative_to(src).parents:
            if parent != Path("."):
                (src / parent / "__init__.py").touch()
    return src


#: Dynamic reaches into a forbidden module: (module file, its source, expected key, row).
DYNAMIC_CASES = [
    pytest.param(
        "osprey/facility/build.py",
        "import importlib\nimportlib.import_module('osprey.cli')\n",
        "osprey.facility.build -> osprey.cli [import_module]",
        "S1",
        id="description-import_module",
    ),
    pytest.param(
        "osprey/facility/build.py",
        "from importlib.resources import files\nfiles('osprey.cli')\n",
        "osprey.facility.build -> osprey.cli [files]",
        "S1",
        id="description-resources-files",
    ),
    pytest.param(
        "osprey/facility/build.py",
        "from pathlib import Path\nHERE = Path(__file__).parents[1] / 'cli'\n",
        "osprey.facility.build -> osprey.cli [__file__]",
        "S1",
        id="description-__file__-climb",
    ),
    pytest.param(
        "osprey/facility/build.py",
        "import importlib\nimportlib.import_module('osprey.facility.panels.emit')\n",
        "osprey.facility.build -> osprey.facility.panels.emit [import_module]",
        "S2",
        id="description-to-panels-import_module",
    ),
    pytest.param(
        "osprey/facility/panels/emit.py",
        "import importlib\nimportlib.import_module('osprey.facility.records')\n",
        "osprey.facility.panels.emit -> osprey.facility.records [import_module]",
        "S3",
        id="panels-import_module",
    ),
    pytest.param(
        "osprey/facility/panels/emit.py",
        "import importlib.resources\nimportlib.resources.files('osprey.facility.records')\n",
        "osprey.facility.panels.emit -> osprey.facility.records [files]",
        "S3",
        id="panels-resources-files",
    ),
    pytest.param(
        "osprey/facility/panels/emit.py",
        "from pathlib import Path\nDOC = Path(__file__).parent.parent / 'records.py'\n",
        "osprey.facility.panels.emit -> osprey.facility.records [__file__]",
        "S3",
        id="panels-__file__-climb",
    ),
    pytest.param(
        "osprey/facility/panels/emit.py",
        "import importlib\nimportlib.import_module('osprey_connectors.config')\n",
        "osprey.facility.panels.emit -> osprey_connectors.config [import_module]",
        "S4",
        id="panels-to-osprey_connectors-import_module",
    ),
    pytest.param(
        "osprey/cli/main.py",
        "import importlib\nimportlib.import_module('osprey.facility.panels')\n",
        "osprey.cli.main -> osprey.facility.panels [import_module]",
        "S5",
        id="framework-to-panels-import_module",
    ),
    pytest.param(
        "osprey/cli/main.py",
        "from importlib.resources import files\nfiles('osprey.facility.compute')\n",
        "osprey.cli.main -> osprey.facility.compute [files]",
        "S6",
        id="framework-to-non-public-files",
    ),
]

#: Real-tree modules a dynamic case reaches, present so the target resolves exactly.
_TARGETS = {
    "osprey/cli/__init__.py": "",
    "osprey/facility/records.py": "",
    "osprey/facility/compute.py": "",
    "osprey/facility/panels/__init__.py": "",
}


@pytest.mark.parametrize(("path", "source", "key", "row"), DYNAMIC_CASES)
def test_a_dynamic_reach_across_a_row_fails(
    tmp_path: Path, path: str, source: str, key: str, row: str
) -> None:
    sides = read_sides()
    edges = dynamic_edges(tree(tmp_path, {**_TARGETS, path: source}), sides)
    assert key in edges.keys() - DYNAMIC_RATCHET.keys(), (
        f"the scan missed {key}; it found {sorted(edges)}"
    )
    reach = edges[key][0]
    assert crossed_row(sides.of(reach.module), reach.target, sides) == row


def test_the_dynamic_cases_start_on_the_sides_they_claim() -> None:
    """Guards the cases: a renamed module would silently move a case off its side."""
    sides = read_sides()
    assert sides.of("osprey.facility.build") == DESCRIPTION
    assert sides.of("osprey.facility.panels.emit") == PANELS
    assert sides.of("osprey.cli.main") == FRAMEWORK


@pytest.mark.parametrize(
    ("path", "source", "key"),
    [
        pytest.param(
            "osprey/facility/build.py",
            "import requests\n",
            "osprey.facility.build -> requests",
            id="description-outside-its-set",
        ),
        pytest.param(
            "osprey/facility/build.py",
            "from pyaml_cs_osprey import kinds\n",
            "osprey.facility.build -> pyaml_cs_osprey",
            id="description-to-sibling-pyaml_cs_osprey",
        ),
        pytest.param(
            "osprey/facility/build.py",
            "import scipy\n",
            "osprey.facility.build -> scipy",
            id="description-mml-extra-outside-mml",
        ),
        pytest.param(
            "osprey/facility/panels/emit.py",
            "import pydantic\n",
            "osprey.facility.panels.emit -> pydantic",
            id="panels-description-dependency",
        ),
        pytest.param(
            "osprey/facility/panels/emit.py",
            "import requests\n",
            "osprey.facility.panels.emit -> requests",
            id="panels-outside-its-set",
        ),
    ],
)
def test_a_third_party_import_outside_the_dependency_set_fails(
    tmp_path: Path, path: str, source: str, key: str
) -> None:
    edges = third_party_edges(tree(tmp_path, {path: source}), read_sides())
    assert key in edges.keys() - THIRD_PARTY_RATCHET.keys(), (
        f"the scan missed {key}; it found {sorted(edges)}"
    )


def test_a_third_party_import_inside_the_dependency_set_passes(tmp_path: Path) -> None:
    """The control for A1: the ruled sets, the stdlib and the root packages are allowed."""
    files = {
        "osprey/facility/build.py": "import json\nimport numpy\nimport pydantic\nimport osprey\n",
        "osprey/facility/layers/mml/importer.py": "import scipy\nimport at\n",
        "osprey/facility/panels/emit.py": "import jsonschema\nimport osprey_connectors\n",
    }
    assert third_party_edges(tree(tmp_path, files), read_sides()) == {}


#: Sources reproducing every scan-ratchet edge, by file.
_RATCHET_SOURCES = {
    "osprey/facility/errors.py": "import click\n",
    "osprey/facility/layers/__init__.py": "import click\n",
    "osprey/facility/layers/list.py": "import click\n",
    "osprey/facility/layers/mml/importer.py": "import click\n",
    "osprey/facility/layers/mml/loaders/json_any.py": "import click\n",
    "osprey/facility/layers/mml/loaders/mat.py": "import click\n",
    "osprey/facility/layers/mml/mapping.py": "import click\n",
    "osprey/facility/layers/mml/systems.py": "import click\n",
    "osprey/facility/validate.py": "import click\n",
    "osprey/facility/kinds.py": "import pyaml_cs_osprey\n",
    "osprey/facility/panels/emit.py": (
        "from pathlib import Path\n"
        "THEME = Path(__file__).parents[2] / 'templates' / 'apps' / 'demo'\n"
    ),
    "osprey/templates/apps/demo/__init__.py": "",
    "osprey/cli/facility_cmd.py": (
        "from importlib.resources import files\nfiles('osprey.facility.layers.mml')\n"
    ),
}


def _stale(src: Path) -> set[str]:
    sides = read_sides()
    return (THIRD_PARTY_RATCHET.keys() - third_party_edges(src, sides).keys()) | (
        DYNAMIC_RATCHET.keys() - dynamic_edges(src, sides).keys()
    )


def test_a_tree_with_every_scan_ratchet_edge_leaves_no_entry_stale(tmp_path: Path) -> None:
    """The control for the deletion cases below."""
    assert _stale(tree(tmp_path, _RATCHET_SOURCES)) == set()


@pytest.mark.parametrize(
    ("path", "key"),
    [
        pytest.param(
            "osprey/facility/validate.py", "osprey.facility.validate -> click", id="third-party"
        ),
        pytest.param(
            "osprey/cli/facility_cmd.py",
            "osprey.cli.facility_cmd -> osprey.facility.layers.mml [files]",
            id="dynamic-files",
        ),
        pytest.param(
            "osprey/facility/panels/emit.py",
            "osprey.facility.panels.emit -> osprey.templates.apps.demo [__file__]",
            id="dynamic-__file__",
        ),
    ],
)
def test_deleting_a_scan_ratchet_edge_without_its_entry_fails(
    tmp_path: Path, path: str, key: str
) -> None:
    src = tree(tmp_path, {**_RATCHET_SOURCES, path: "\n"})
    assert _stale(src) == {key}
