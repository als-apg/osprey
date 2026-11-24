"""Pins scripts/ci/lean_import_walk.py: one walk, driven by the package table.

Each test builds a scratch venv holding stand-in packages under a real
package's ``import_name`` and runs the walk with that venv's interpreter in
isolated mode, the way the package check and the release workflows do. A
forbidden module that is importable is not enough to fail the walk; a package
module that imports it is.
"""

from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WALK = REPO_ROOT / "scripts" / "ci" / "lean_import_walk.py"


def _site_packages(python: Path) -> Path:
    out = subprocess.run(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


@pytest.fixture
def scratch(tmp_path: Path) -> tuple[Path, Path]:
    """A bare venv (no pip, nothing installed) and its site-packages."""
    env = tmp_path / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=sys.platform != "win32").create(env)
    python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    return python, _site_packages(python)


def _install(site: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = site / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


#: A clean stand-in for the connectors wheel: every entry module, a subpackage
#: leaf no entry module imports, and the PEP 561 marker.
CONNECTORS = {
    "osprey_connectors/__init__.py": "",
    "osprey_connectors/py.typed": "",
    "osprey_connectors/config.py": "",
    "osprey_connectors/errors.py": "",
    "osprey_connectors/logger.py": "",
    "osprey_connectors/simulation.py": "",
    "osprey_connectors/control_system/__init__.py": "",
    "osprey_connectors/control_system/_epics_connector.py": "",
    "osprey_connectors/control_system/leaf.py": "",
}

PANELS = {
    "osprey_panels/__init__.py": "",
    "osprey_panels/cli.py": "",
    "osprey_panels/layout.py": "",
}


def _walk(python: Path, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(python), "-I", str(WALK), *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )


def test_a_clean_wheel_passes_and_counts_its_modules(
    scratch: tuple[Path, Path], tmp_path: Path
) -> None:
    """Run ``-I`` from a foreign cwd: the walk still finds its stream table,
    because it loads it by path beside itself, never from ``sys.path``."""
    python, site = scratch
    _install(site, CONNECTORS)
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    result = _walk(python, "connectors", cwd=foreign)
    assert result.returncode == 0, result.stderr
    # config, control_system, control_system._epics_connector, control_system.leaf,
    # errors, logger, simulation.
    assert result.stdout.strip() == "LEAN WHEEL OK (7 modules)"


def test_the_table_loads_under_isolation_with_no_checkout_on_the_path(
    scratch: tuple[Path, Path], tmp_path: Path
) -> None:
    """A copy of the walk beside a copy of the table, outside the checkout,
    run ``-I`` from yet another directory: only path loading can find it."""
    python, site = scratch
    _install(site, PANELS)
    alone = tmp_path / "alone"
    alone.mkdir()
    for name in ("lean_import_walk.py", "release_streams.py"):
        (alone / name).write_text((WALK.parent / name).read_text(encoding="utf-8"))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    result = subprocess.run(
        [str(python), "-I", str(alone / "lean_import_walk.py"), "panels"],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LEAN WHEEL OK (2 modules)"


def test_an_importable_forbidden_module_alone_does_not_fail(scratch: tuple[Path, Path]) -> None:
    python, site = scratch
    _install(site, {**CONNECTORS, "litellm/__init__.py": ""})
    result = _walk(python, "connectors")
    assert result.returncode == 0, result.stderr


def test_a_leaf_that_imports_a_forbidden_module_fails(scratch: tuple[Path, Path]) -> None:
    """The leaf is reached only by the walk, not by an entry module: the
    regression the walk exists for."""
    python, site = scratch
    _install(
        site,
        {
            **CONNECTORS,
            "litellm/__init__.py": "",
            "osprey_connectors/control_system/leaf.py": "import litellm\n",
        },
    )
    result = _walk(python, "connectors")
    assert result.returncode != 0
    assert "imported ['litellm']" in result.stderr
    assert "LEAN WHEEL OK" not in result.stdout


def test_an_entry_module_that_imports_the_framework_fails(scratch: tuple[Path, Path]) -> None:
    python, site = scratch
    _install(
        site,
        {**CONNECTORS, "osprey/__init__.py": "", "osprey_connectors/config.py": "import osprey\n"},
    )
    result = _walk(python, "connectors")
    assert result.returncode != 0
    assert "lean wheel loaded: ['osprey']" in result.stderr


def test_panels_refuses_osprey_facility(scratch: tuple[Path, Path]) -> None:
    python, site = scratch
    _install(
        site,
        {
            **PANELS,
            "osprey_facility/__init__.py": "",
            "osprey_panels/layout.py": "import osprey_facility\n",
        },
    )
    result = _walk(python, "panels")
    assert result.returncode != 0
    assert "imported ['osprey_facility']" in result.stderr


def test_a_module_that_fails_to_import_fails_the_walk(scratch: tuple[Path, Path]) -> None:
    """walk_packages swallows a subpackage's import error unless onerror raises."""
    python, site = scratch
    _install(
        site,
        {
            **CONNECTORS,
            "osprey_connectors/broken/__init__.py": "raise RuntimeError('boom')\n",
            "osprey_connectors/broken/inner.py": "",
        },
    )
    result = _walk(python, "connectors")
    assert result.returncode != 0
    assert "boom" in result.stderr or "failed to walk" in result.stderr


def test_a_typed_stream_needs_its_marker(scratch: tuple[Path, Path]) -> None:
    python, site = scratch
    _install(site, {k: v for k, v in CONNECTORS.items() if not k.endswith("py.typed")})
    result = _walk(python, "connectors")
    assert result.returncode != 0
    assert "py.typed missing from the wheel" in result.stderr


def test_an_untyped_stream_needs_no_marker(scratch: tuple[Path, Path]) -> None:
    python, site = scratch
    _install(site, {"osprey_facility/__init__.py": "", "osprey_facility/schema.py": ""})
    result = _walk(python, "facility")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "LEAN WHEEL OK (1 modules)"


def test_a_missing_entry_module_fails(scratch: tuple[Path, Path]) -> None:
    python, site = scratch
    _install(site, {k: v for k, v in PANELS.items() if not k.endswith("cli.py")})
    result = _walk(python, "panels")
    assert result.returncode != 0
    assert "osprey_panels.cli" in result.stderr


@pytest.mark.parametrize("args", [(), ("framework",), ("connectors", "facility")])
def test_the_walk_needs_exactly_one_known_package_key(
    scratch: tuple[Path, Path], args: tuple[str, ...]
) -> None:
    python, site = scratch
    _install(site, CONNECTORS)
    result = _walk(python, *args)
    assert result.returncode == 2
    assert "usage: lean_import_walk.py <connectors|facility|panels>" in result.stderr
