"""The framework import scanner finds every site and resolves it for real.

Each case builds a small framework tree and a small ``osprey_connectors`` in
``tmp_path`` and runs the scanner as CI does, ``python -I`` in a child with
``--connectors-path`` naming the synthetic package, so resolution imports
the synthetic modules and never the installed distribution.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from osprey_connectors_tests.compat import scan_framework_imports as scanner

SCRIPT = Path(scanner.__file__)

#: The synthetic connectors every case resolves against.
CONNECTORS = {
    "osprey_connectors/__init__.py": "",
    "osprey_connectors/types.py": "EPICS = 'epics'\n",
    "osprey_connectors/simulation/__init__.py": "",
    "osprey_connectors/simulation/view.py": "class SimulatorView: ...\n",
    "osprey_connectors/lazy.py": textwrap.dedent(
        """\
        __all__ = ["Lazy"]

        def __getattr__(name):
            if name == "Lazy":
                return object
            raise AttributeError(name)
        """
    ),
    "osprey_connectors/broken.py": "raise RuntimeError('cannot load')\n",
}


def _write(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")


def _scan(tmp_path: Path, framework: dict[str, str]) -> subprocess.CompletedProcess[str]:
    connectors = tmp_path / "connectors"
    _write(connectors, CONNECTORS)
    _write(tmp_path / "fw", {f"osprey/{name}": text for name, text in framework.items()})
    return subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            str(tmp_path / "fw" / "osprey"),
            "--connectors-path",
            str(connectors),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _reported(result: subprocess.CompletedProcess[str]) -> list[str]:
    """The ``file:line: module[:name]`` part of each reported line."""
    return [line.split("  (", 1)[0] for line in result.stdout.splitlines()]


def test_a_tree_whose_every_name_resolves_scans_clean(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "__init__.py": "import osprey_connectors.types\nfrom osprey_connectors.types import EPICS\n",
            "views.py": "from osprey_connectors.simulation.view import SimulatorView\n",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == ""
    assert "3 sites, 0 unresolved" in result.stderr


def test_a_missing_module_is_reported_at_its_line(tmp_path: Path) -> None:
    result = _scan(tmp_path, {"__init__.py": "\n\nimport osprey_connectors.channel_taxonomy\n"})
    assert result.returncode == 1
    assert _reported(result) == ["osprey/__init__.py:3: osprey_connectors.channel_taxonomy"]
    assert "ModuleNotFoundError" in result.stdout


def test_a_missing_name_is_reported(tmp_path: Path) -> None:
    result = _scan(tmp_path, {"__init__.py": "from osprey_connectors.types import EPICS, MOCK\n"})
    assert result.returncode == 1
    assert _reported(result) == ["osprey/__init__.py:1: osprey_connectors.types:MOCK"]


def test_a_name_that_is_a_submodule_resolves(tmp_path: Path) -> None:
    result = _scan(tmp_path, {"__init__.py": "from osprey_connectors.simulation import view\n"})
    assert result.returncode == 0, result.stdout


def test_a_lazy_getattr_export_resolves(tmp_path: Path) -> None:
    result = _scan(tmp_path, {"__init__.py": "from osprey_connectors.lazy import Lazy\n"})
    assert result.returncode == 0, result.stdout


def test_a_name_listed_in_all_but_not_served_by_getattr_is_reported(tmp_path: Path) -> None:
    """``__all__`` is a claim, not an export: ``from m import X`` reads the
    attribute, and a name ``__getattr__`` refuses fails there."""
    result = _scan(tmp_path, {"__init__.py": "from osprey_connectors.lazy import __all__, Gone\n"})
    assert _reported(result) == ["osprey/__init__.py:1: osprey_connectors.lazy:Gone"]


def test_a_module_that_raises_on_import_is_reported_with_its_error(tmp_path: Path) -> None:
    result = _scan(tmp_path, {"__init__.py": "def load():\n    import osprey_connectors.broken\n"})
    assert _reported(result) == ["osprey/__init__.py:2: osprey_connectors.broken"]
    assert "RuntimeError: cannot load" in result.stdout


def test_a_string_module_path_is_resolved(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "launch.py": """\
                HOST = "osprey_connectors.simulation.view"
                ATTR = "osprey_connectors.types.EPICS"
                ENTRY = "osprey_connectors.simulation.view:SimulatorView"
                GONE = "osprey_connectors.ipc.host"
                ARGV = ["python", "-m osprey_connectors.types.EPICS"]
                PROSE = "see osprey_connectors.ipc.host for details"
            """,
        },
    )
    assert _reported(result) == [
        "osprey/launch.py:4: osprey_connectors.ipc.host",
        "osprey/launch.py:5: osprey_connectors.types.EPICS",
    ]


def test_a_registration_class_name_is_resolved(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "registry.py": """\
                REGISTRATIONS = [
                    dict(module_path="osprey_connectors.simulation.view", class_name="SimulatorView"),
                    dict(module_path="osprey_connectors.simulation.view", class_name="MockView"),
                ]
            """,
        },
    )
    assert _reported(result) == ["osprey/registry.py:3: osprey_connectors.simulation.view:MockView"]


def test_an_attribute_read_off_a_from_bound_module_is_resolved(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "build.py": """\
                from osprey_connectors import types as connector_types

                KNOWN = connector_types.EPICS
                GONE = connector_types.MOCK
            """,
        },
    )
    assert result.returncode == 1
    assert _reported(result) == ["osprey/build.py:4: osprey_connectors.types:MOCK"]


def test_an_attribute_read_off_an_import_alias_is_resolved(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "runtime.py": """\
                def settle():
                    import osprey_connectors.simulation.view as view_step

                    view_step.SimulatorView()
                    return view_step.settle()
            """,
        },
    )
    assert _reported(result) == ["osprey/runtime.py:5: osprey_connectors.simulation.view:settle"]


def test_a_dotted_chain_off_a_plain_import_is_walked_through_submodules(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "__init__.py": """\
                import osprey_connectors.simulation

                osprey_connectors.simulation.view.SimulatorView.anything
                osprey_connectors.types.EPICS.upper()
                osprey_connectors.simulation.gone.Thing
            """,
        },
    )
    assert _reported(result) == ["osprey/__init__.py:5: osprey_connectors:simulation.gone.Thing"]


def test_a_name_bound_to_a_non_module_carries_no_attribute_site(tmp_path: Path) -> None:
    """``EPICS`` is a string: what is read off it is not the connectors' surface."""
    result = _scan(
        tmp_path,
        {"__init__.py": "from osprey_connectors.types import EPICS\nEPICS.upper()\n"},
    )
    assert result.returncode == 0, result.stdout
    assert result.stdout == ""


def test_an_attribute_assigned_or_deleted_on_a_module_need_not_exist(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "osprey/patch.py": """\
                from osprey_connectors import types as t

                t.NEW = 1
                del t.OLD
                t.sub.NEW = 2
            """,
        },
    )
    sites, _ = scanner.collect_sites(tmp_path / "osprey")
    assert [(site.line, site.label(), site.kind) for site in sites] == [
        (1, "osprey_connectors:types", "from"),
        (5, "osprey_connectors.types:sub", "attr"),
    ]


def test_relative_imports_are_not_sites(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {
            "__init__.py": "from osprey_connectors.types import EPICS\n",
            "osprey_connectors.py": "",
            "user.py": "from .osprey_connectors import anything\nfrom . import osprey_connectors\n",
        },
    )
    assert result.returncode == 0, result.stdout
    assert "1 sites" in result.stderr


def test_sites_are_found_wherever_the_import_sits(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {
            "osprey/deep.py": """\
                from typing import TYPE_CHECKING

                if TYPE_CHECKING:
                    from osprey_connectors.types import A

                try:
                    from osprey_connectors.types import B
                except ImportError:
                    B = None

                def later():
                    import osprey_connectors.c

                class Holder:
                    def method(self):
                        from osprey_connectors.types import *
            """,
        },
    )
    sites, unparsable = scanner.collect_sites(tmp_path / "osprey")
    assert unparsable == []
    assert [(site.line, site.label(), site.kind) for site in sites] == [
        (4, "osprey_connectors.types:A", "from"),
        (7, "osprey_connectors.types:B", "from"),
        (12, "osprey_connectors.c", "import"),
        (16, "osprey_connectors.types", "from"),
    ]


def test_an_unparsable_file_fails_the_scan(tmp_path: Path) -> None:
    result = _scan(
        tmp_path,
        {"__init__.py": "from osprey_connectors.types import EPICS\n", "bad.py": "def (:\n"},
    )
    assert result.returncode == 1
    assert "osprey/bad.py" in result.stdout and "(unparsable)" in result.stdout


def test_a_tree_with_no_site_is_refused(tmp_path: Path) -> None:
    """Zero sites means the scan read the wrong directory, not a clean framework."""
    result = _scan(tmp_path, {"__init__.py": "import json\n"})
    assert result.returncode == 2
    assert "no site found" in result.stderr


def test_a_connectors_path_that_does_not_supply_the_package_is_refused(tmp_path: Path) -> None:
    """The installed distribution must not stand in for the tree asked for."""
    _write(tmp_path / "fw", {"osprey/__init__.py": "import osprey_connectors.types\n"})
    (tmp_path / "empty").mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            str(tmp_path / "fw" / "osprey"),
            "--connectors-path",
            str(tmp_path / "empty"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "not from" in result.stderr


def test_every_tree_named_is_scanned(tmp_path: Path) -> None:
    connectors = tmp_path / "connectors"
    _write(connectors, CONNECTORS)
    _write(
        tmp_path,
        {
            "osprey/__init__.py": "from osprey_connectors.types import EPICS\n",
            "backend/__init__.py": "from osprey_connectors.types import MOCK\n",
        },
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(SCRIPT),
            str(tmp_path / "osprey"),
            str(tmp_path / "backend"),
            "--connectors-path",
            str(connectors),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 1
    assert _reported(result) == ["backend/__init__.py:1: osprey_connectors.types:MOCK"]
    assert result.stderr.count("scanned ") == 2


@pytest.mark.parametrize("argv", [[], ["--package", "osprey_connectors", "somewhere"]])
def test_the_cli_needs_trees_or_packages_not_both(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        scanner.main(argv)
    assert raised.value.code == 2
