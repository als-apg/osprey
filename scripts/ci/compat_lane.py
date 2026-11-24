#!/usr/bin/env python3
"""The steps of a component package's compatibility cell, shared by every lane that runs one.

``--package`` names the package under test (:data:`LANES`): ``connectors``
for ci.yml's ``connectors-compat`` (Lane A) and
``framework-at-connectors-floor`` (Lane B) and the nightly
``compat-nightly.yml``, ``facility`` for the facility lanes. The
lanes call these commands rather than spell the steps out, so a dispatched
nightly dry run against a named release exercises the code a PR runs:

``install``
    A venv holding one framework release from PyPI with the package under
    test in place of the one it resolved: the wheel built at HEAD in
    ``dist/`` (``--under-test head``), or a release from PyPI. The result
    must pass ``uv pip check`` and hold exactly that package.
``check``
    Lane A's checks on such a venv. The package and the framework import from
    its site-packages, and every name the release's packages take from the
    package resolves (``scan_framework_imports.py --target <import name>``).
    For connectors those packages are ``osprey``, ``pyaml_cs_osprey`` where
    the release depends on it and ``osprey_facility`` where it is installed;
    every module of them also imports (``import_walk.py``) and the file
    contracts hold (``contracts.py``), the simulator view required of every
    release :func:`compat_matrix.view_required` names. For facility the
    package scanned is ``osprey``.
``assert-installed``
    That the named modules import from the venv's site-packages (Lane B: the
    floor release, not the workspace member).
``replace-tree``
    Swaps a release checkout's ``<package dir>/src`` for the package
    installed in its venv, so a path the release's tests compute to "its
    connectors" (a child process's ``PYTHONPATH``) names the package under
    test. ``replace-connectors-tree`` is its older name.
``pytest``
    The release's or the framework's package-facing suites, with the
    :data:`GUARD_PLUGIN` holding the named modules to site-packages and the
    package's ``source-text-tests`` deselected.

The compat scripts run as ``python -I`` in the venv, so nothing of this
checkout is importable beside the release. This driver runs with the
runner's ``python3`` and the standard library only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPAT = (
    REPO_ROOT / "packages" / "osprey-connectors" / "tests" / "osprey_connectors_tests" / "compat"
)
#: Where the lanes build the package wheel at HEAD.
DIST = REPO_ROOT / "dist"
#: The directory put on the suites' ``PYTHONPATH`` for :data:`GUARD_PLUGIN`.
PLUGIN_DIR = REPO_ROOT / "scripts" / "ci" / "pytest_plugins"
GUARD_PLUGIN = "connectors_compat_guard"
#: The Python every lane venv is made with.
PYTHON_VERSION = "3.11"


class Lane(NamedTuple):
    """What one package's cell installs, checks and reports."""

    #: The ``compat_matrix.PACKAGES`` key, ``--package``'s value, and the
    #: :data:`_PROBE` key of the installed version.
    key: str
    dist: str
    import_name: str
    package_dir: str
    #: The packages ``check`` scans, or ``None`` for those the probe names
    #: (the connectors: what the installed release depends on).
    consumers: tuple[str, ...] | None
    #: Whether ``check`` also walks every module and holds the file
    #: contracts: both read the connectors, and nothing else.
    connectors_checks: bool

    @property
    def wheel_glob(self) -> str:
        return f"{self.import_name}-*.whl"


CONNECTORS = Lane(
    key="connectors",
    dist="osprey-connectors",
    import_name="osprey_connectors",
    package_dir="packages/osprey-connectors",
    consumers=None,
    connectors_checks=True,
)
FACILITY = Lane(
    key="facility",
    dist="osprey-facility",
    import_name="osprey_facility",
    package_dir="packages/osprey-facility",
    consumers=("osprey",),
    connectors_checks=False,
)
LANES: dict[str, Lane] = {lane.key: lane for lane in (CONNECTORS, FACILITY)}

#: Run inside the venv (``python -I -c``): what is installed, and from where.
#: ``packages`` is what the connectors lane scans: ``osprey``,
#: ``pyaml_cs_osprey`` where the framework depends on it outside an extra, and
#: ``osprey_facility`` whenever it is installed.
_PROBE = r"""
import importlib.metadata as m, importlib.util, json, re, sys, sysconfig
from pathlib import Path

def origin(name):
    spec = importlib.util.find_spec(name)
    return None if spec is None or spec.origin is None else str(Path(spec.origin).resolve())

def version(dist):
    try:
        return m.version(dist)
    except m.PackageNotFoundError:
        return None

requires = (m.requires("osprey-framework") or []) if version("osprey-framework") else []
backend = any(
    re.match(r"pyaml[-_.]cs[-_.]osprey(?![\w.-])", r, re.I) and "extra ==" not in r
    for r in requires
)
facility = importlib.util.find_spec("osprey_facility") is not None
print(json.dumps({
    "purelib": str(Path(sysconfig.get_path("purelib")).resolve()),
    "framework": version("osprey-framework"),
    "connectors": version("osprey-connectors"),
    "facility": version("osprey-facility"),
    "packages": (
        (["osprey", "pyaml_cs_osprey"] if backend else ["osprey"])
        + (["osprey_facility"] if facility else [])
    ),
    "origins": {name: origin(name) for name in sys.argv[1:]},
}))
"""


class LaneError(RuntimeError):
    """A step found the venv other than the lane needs it."""


def _matrix() -> ModuleType:
    name = "compat_matrix"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("compat_matrix.py")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(
    argv: Sequence[str | Path], cwd: Path | None = None, env: dict[str, str] | None = None
) -> int:
    print(f"+ {' '.join(map(str, argv))}", flush=True)
    return subprocess.run([str(arg) for arg in argv], cwd=cwd, env=env, check=False).returncode


def _must(argv: Sequence[str | Path]) -> None:
    if _run(argv) != 0:
        raise LaneError(f"{' '.join(map(str, argv[:3]))} failed")


def venv_python(venv: Path) -> Path:
    return venv / "bin" / "python"


def probe(python: Path, *modules: str) -> dict[str, object]:
    """What ``python``'s environment holds, and where ``modules`` import from."""
    result = subprocess.run(
        [str(python), "-I", "-c", _PROBE, *modules],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise LaneError(f"could not read the environment of {python}:\n{result.stderr}")
    return json.loads(result.stdout)


def installed_version(lane: Lane, info: dict[str, object]) -> object:
    """The version of ``lane``'s package a :func:`probe` found installed."""
    return info[lane.key]


def wheel_version(wheel: Path) -> str:
    """The ``Version`` a wheel's metadata states."""
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if re.match(r"[^/]+\.dist-info/METADATA$", n))
        for line in archive.read(name).decode("utf-8").splitlines():
            if line.startswith("Version: "):
                return line.removeprefix("Version: ").strip()
    raise LaneError(f"{wheel} states no version")


def target(lane: Lane, under_test: str, dist: Path = DIST) -> tuple[str, str]:
    """``(what uv installs, the version it must leave installed)`` for ``under_test``.

    ``under_test`` is ``head`` (the one wheel of ``lane``'s package ``uv
    build`` left in ``dist``) or a release version on PyPI.
    """
    if under_test == "head":
        wheels = sorted(dist.glob(lane.wheel_glob))
        if len(wheels) != 1:
            raise LaneError(
                f"expected one {lane.key} wheel built at HEAD in {dist}, found {wheels}"
            )
        return str(wheels[0]), wheel_version(wheels[0])
    _matrix().version_key(under_test)
    return f"{lane.dist}=={under_test}", under_test


def outside_site_packages(info: dict[str, object], modules: Iterable[str]) -> list[str]:
    """Each of ``modules`` that imports from anywhere but the venv's site-packages."""
    purelib = Path(str(info["purelib"]))
    origins = info["origins"]
    assert isinstance(origins, dict)
    return [
        f"{name} imports from {origins.get(name)}, not from {purelib}"
        for name in modules
        if origins.get(name) is None or purelib not in Path(origins[name]).parents
    ]


def install(lane: Lane, venv: Path, framework: str, under_test: str, extra: str | None) -> None:
    requirement, expected = target(lane, under_test)
    python = venv_python(venv)
    spec = f"osprey-framework{f'[{extra}]' if extra else ''}=={framework}"
    # `if-necessary` takes a pre-release only where nothing stable satisfies a
    # requirement — a pin that names one included, since no stable satisfies
    # `==X.YbN`. That is its meaning since uv 0.12, which made the older
    # `if-necessary-or-explicit` a deprecated alias of it (and warns on it).
    prerelease = "--prerelease=if-necessary"
    _must(["uv", "venv", "--clear", venv, "--python", PYTHON_VERSION])
    _must(["uv", "pip", "install", "--python", python, prerelease, spec])
    # With dependencies, then checked: a dependency the package added must
    # be installed, and one the release pins differently must show here
    # rather than as a quiet downgrade.
    _must(
        ["uv", "pip", "install", "--python", python, prerelease]
        + ["--reinstall-package", lane.dist, requirement]
    )
    _must(["uv", "pip", "check", "--python", python])
    info = probe(python)
    found = installed_version(lane, info)
    print(f"osprey-framework {info['framework']} / {lane.dist} {found}")
    if info["framework"] != framework:
        raise LaneError(f"osprey-framework {info['framework']} is installed, not {framework}")
    if found != expected:
        raise LaneError(f"{lane.dist} {found} is installed, not {expected}")


def check_commands(
    lane: Lane,
    python: Path,
    packages: Sequence[str],
    workdir: Path | None,
    *,
    require_view: bool,
) -> list[tuple[str, list[str]]]:
    """Lane A's checks, each ``(title, argv)``; every script runs ``python -I``."""
    named = [arg for package in packages for arg in ("--package", package)]
    scan = [str(python), "-I", str(COMPAT / "scan_framework_imports.py"), *named]
    if not lane.connectors_checks:
        return [
            (
                f"Scan the release for {lane.import_name} names that no longer resolve",
                [*scan, "--target", lane.import_name],
            )
        ]
    if workdir is None:
        raise LaneError("the connectors checks need --workdir for the file contracts")
    contracts = [str(python), "-I", str(COMPAT / "contracts.py"), "all", "--workdir", str(workdir)]
    return [
        ("Scan the release for connectors names that no longer resolve", scan),
        (
            "Import every module of the release",
            [str(python), "-I", str(COMPAT / "import_walk.py"), *named],
        ),
        (
            "Check the file contracts the release shares with the connectors",
            contracts + (["--require-view"] if require_view else []),
        ),
    ]


def check(lane: Lane, venv: Path, workdir: Path | None) -> int:
    python = venv_python(venv)
    info = probe(python, "osprey", lane.import_name)
    if info["framework"] is None:
        raise LaneError(f"no osprey-framework installed in {venv}")
    misplaced = outside_site_packages(info, ["osprey", lane.import_name])
    if misplaced:
        raise LaneError("; ".join(misplaced))
    packages = list(lane.consumers) if lane.consumers is not None else info["packages"]
    assert isinstance(packages, list)
    for package in packages:
        if probe(python, package)["origins"][package] is None:  # type: ignore[index]
            raise LaneError(f"the release depends on {package}, which does not import")
    require_view = lane.connectors_checks and _matrix().view_required(str(info["framework"]))
    view = (
        f"; simulator view {'required' if require_view else 'checked only if the release writes one'}"
        if lane.connectors_checks
        else ""
    )
    print(
        f"osprey-framework {info['framework']} / {lane.dist} {installed_version(lane, info)}; "
        f"checking {', '.join(packages)}{view}"
    )
    failed = []
    for title, argv in check_commands(lane, python, packages, workdir, require_view=require_view):
        print(f"::group::{title}", flush=True)
        status = _run(argv)
        print("::endgroup::", flush=True)
        if status != 0:
            print(f"::error::{title}: failed ({status})", flush=True)
            failed.append(title)
    return 1 if failed else 0


def assert_installed(
    lane: Lane, python: Path, modules: Sequence[str], expected_version: str | None
) -> None:
    info = probe(python, *modules)
    misplaced = outside_site_packages(info, modules)
    if misplaced:
        raise LaneError("; ".join(misplaced))
    found = installed_version(lane, info)
    if expected_version is not None and found != expected_version:
        raise LaneError(f"{lane.dist} {found} is installed, not {expected_version}")
    origins = info["origins"]
    assert isinstance(origins, dict)
    for name in modules:
        print(f"{name}: {origins[name]}")


def replace_tree(lane: Lane, venv: Path, release_dir: Path) -> None:
    info = probe(venv_python(venv), lane.import_name)
    if outside_site_packages(info, [lane.import_name]):
        raise LaneError(f"the {lane.dist} under test is not installed in the venv")
    installed = Path(str(info["origins"][lane.import_name])).parent  # type: ignore[index]
    tree = release_dir / lane.package_dir / "src"
    shutil.rmtree(tree, ignore_errors=True)
    shutil.copytree(
        installed, tree / lane.import_name, ignore=shutil.ignore_patterns("__pycache__")
    )
    print(f"{tree / lane.import_name} is now a copy of {installed}")


def pytest_command(python: Path, deselect: Iterable[str], pytest_args: Sequence[str]) -> list[str]:
    deselected = [arg for node_id in deselect for arg in ("--deselect", node_id)]
    return [str(python), "-m", "pytest", "-p", GUARD_PLUGIN, *deselected, *pytest_args]


def source_text_tests(lane: Lane) -> list[str]:
    """The ``source-text-tests`` the package's compat table lists."""
    matrix = _matrix()
    return matrix.source_text_tests(matrix.PACKAGES[lane.key])


def run_pytest(
    lane: Lane, python: Path, cwd: Path, installed: Sequence[str], pytest_args: Sequence[str]
) -> int:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(PLUGIN_DIR), os.environ.get("PYTHONPATH", "")])
        ),
        "CONNECTORS_COMPAT_INSTALLED": " ".join(installed),
    }
    # Anchored here: the child starts in *cwd*, where a relative interpreter path
    # (the nightly's `release/.venv/bin/python` with `--cwd release`) would be read
    # against the wrong directory. absolute(), not resolve(): a venv's python is
    # a symlink, and resolving it would leave the venv.
    argv = pytest_command(python.absolute(), source_text_tests(lane), pytest_args)
    return _run(argv, cwd=cwd, env=env)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--package", required=True, choices=tuple(LANES))
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("install", help="a framework release with the package under test")
    step.add_argument("--venv", type=Path, required=True)
    step.add_argument("--framework", required=True, help="the framework release, from PyPI")
    step.add_argument(
        "--under-test",
        "--connectors",
        dest="under_test",
        required=True,
        help="head (the wheel built in dist/), or a release of the package from PyPI",
    )
    step.add_argument("--extra", help="an extra of osprey-framework to install, e.g. dev")
    step = commands.add_parser("check", help="Lane A's checks on an installed venv")
    step.add_argument("--venv", type=Path, required=True)
    step.add_argument(
        "--workdir", type=Path, help="where the file contracts write (connectors: required)"
    )
    step = commands.add_parser("assert-installed", help="modules import from site-packages")
    step.add_argument("--python", type=Path, required=True)
    step.add_argument("--module", action="append", required=True)
    step.add_argument(
        "--version",
        "--connectors-version",
        dest="expected_version",
        help="the version of the package that must be installed",
    )
    step = commands.add_parser(
        "replace-tree",
        aliases=["replace-connectors-tree"],
        help="point a release's tree at the venv's",
    )
    step.add_argument("--venv", type=Path, required=True)
    step.add_argument("--release-dir", type=Path, required=True)
    step = commands.add_parser("pytest", help="the suites, guarded, source-text tests deselected")
    step.add_argument("--python", type=Path, required=True)
    step.add_argument("--cwd", type=Path, default=Path.cwd())
    step.add_argument("--installed", action="append", required=True)
    step.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv) if argv is not None else None)
    lane = LANES[args.package]

    try:
        if args.command == "install":
            install(lane, args.venv, args.framework, args.under_test, args.extra)
        elif args.command == "check":
            return check(lane, args.venv, args.workdir)
        elif args.command == "assert-installed":
            assert_installed(lane, args.python, args.module, args.expected_version)
        elif args.command in ("replace-tree", "replace-connectors-tree"):
            replace_tree(lane, args.venv, args.release_dir)
        else:
            pytest_args = (
                args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
            )
            return run_pytest(lane, args.python, args.cwd, args.installed, pytest_args)
    except (LaneError, ValueError) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
