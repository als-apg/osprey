"""The shared steps of a package's compatibility cell build the commands the lanes need."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import sysconfig
import textwrap
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_CI = REPO_ROOT / "scripts" / "ci"


def _load(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # import-time required because scripts/ is not a package: the script is loaded by
    # path and registered before exec so its dataclasses resolve cls.__module__, and
    # its tables feed module-level parametrize cases.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lane = _load("compat_lane", _CI / "compat_lane.py")

LANES = [pytest.param(lane.LANES[key], id=key) for key in ("connectors", "facility")]


@pytest.mark.parametrize(
    "require_view",
    [pytest.param(False, id="connectors-False"), pytest.param(True, id="connectors-True")],
)
def test_every_check_runs_its_script_isolated(require_view: bool, tmp_path: Path) -> None:
    """``python -I``: the checkout and the script directory stay off ``sys.path``,
    so the release under test is the only framework importable."""
    python = tmp_path / "bin" / "python"
    commands = lane.check_commands(
        lane.CONNECTORS,
        python,
        ["osprey", "pyaml_cs_osprey"],
        tmp_path / "w",
        require_view=require_view,
    )
    scripts = [Path(argv[2]).name for _, argv in commands]
    assert scripts == ["scan_framework_imports.py", "import_walk.py", "contracts.py"]
    for _, argv in commands:
        assert argv[:2] == [str(python), "-I"]
        assert Path(argv[2]).is_file()
    for _, argv in commands[:2]:
        assert argv[3:] == ["--package", "osprey", "--package", "pyaml_cs_osprey"]
    assert ("--require-view" in commands[2][1]) is require_view


def test_the_facility_check_scans_the_framework_for_facility_names(tmp_path: Path) -> None:
    """The facility cell scans the framework's ``osprey_facility`` sites, and
    holds neither the import walk nor the connectors' file contracts."""
    python = tmp_path / "bin" / "python"
    commands = lane.check_commands(lane.FACILITY, python, ["osprey"], None, require_view=False)
    assert len(commands) == 1
    _, argv = commands[0]
    assert argv[:2] == [str(python), "-I"]
    assert Path(argv[2]).name == "scan_framework_imports.py" and Path(argv[2]).is_file()
    assert argv[3:] == ["--package", "osprey", "--target", "osprey_facility"]


def test_the_connectors_checks_need_a_workdir(tmp_path: Path) -> None:
    with pytest.raises(lane.LaneError, match="--workdir"):
        lane.check_commands(lane.CONNECTORS, tmp_path / "py", ["osprey"], None, require_view=False)


def _wheel(path: Path, version: str, dist: str = "osprey-connectors") -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{dist.replace('-', '_')}-x.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: {dist}\nVersion: {version}\n",
        )
    return path


@pytest.mark.parametrize("package", LANES)
def test_head_is_the_built_wheel_at_the_version_its_metadata_states(
    package, tmp_path: Path
) -> None:
    wheel = _wheel(
        tmp_path / f"{package.import_name}-1-py3-none-any.whl", "2026.10.1.post3+gabc", package.dist
    )
    assert lane.target(package, "head", tmp_path) == (str(wheel), "2026.10.1.post3+gabc")


def test_head_is_only_the_named_packages_wheel(tmp_path: Path) -> None:
    """``dist/`` holds every wheel the lane built; each cell takes its own."""
    connectors = _wheel(tmp_path / "osprey_connectors-1-py3-none-any.whl", "1")
    facility = _wheel(tmp_path / "osprey_facility-2-py3-none-any.whl", "2", "osprey-facility")
    assert lane.target(lane.CONNECTORS, "head", tmp_path) == (str(connectors), "1")
    assert lane.target(lane.FACILITY, "head", tmp_path) == (str(facility), "2")


@pytest.mark.parametrize("package", LANES)
def test_head_needs_exactly_one_built_wheel(package, tmp_path: Path) -> None:
    with pytest.raises(lane.LaneError, match="expected one"):
        lane.target(package, "head", tmp_path)
    _wheel(tmp_path / f"{package.import_name}-1-py3-none-any.whl", "1", package.dist)
    _wheel(tmp_path / f"{package.import_name}-2-py3-none-any.whl", "2", package.dist)
    with pytest.raises(lane.LaneError, match="expected one"):
        lane.target(package, "head", tmp_path)


@pytest.mark.parametrize("package", LANES)
def test_a_release_is_installed_from_pypi_by_version(package) -> None:
    assert lane.target(package, "2026.9.0b3") == (f"{package.dist}==2026.9.0b3", "2026.9.0b3")


@pytest.mark.parametrize(
    ("package", "under_test"),
    [
        pytest.param(lane.LANES[key], value, id=f"{key}-{value}")
        for key in ("connectors", "facility")
        for value in ("HEAD", "latest", f"dist/osprey_{key}-1.whl")
    ],
)
def test_anything_else_is_refused(package, under_test: str) -> None:
    with pytest.raises(ValueError, match="not a release version"):
        lane.target(package, under_test)


def test_a_module_outside_site_packages_is_reported(tmp_path: Path) -> None:
    purelib = tmp_path / "site-packages"
    info = {
        "purelib": str(purelib),
        "origins": {
            "osprey_connectors": str(purelib / "osprey_connectors" / "__init__.py"),
            "osprey": str(tmp_path / "src" / "osprey" / "__init__.py"),
            "gone": None,
        },
    }
    assert lane.outside_site_packages(info, ["osprey_connectors"]) == []
    reported = lane.outside_site_packages(info, ["osprey", "gone", "osprey_connectors"])
    assert [line.split()[0] for line in reported] == ["osprey", "gone"]


def test_the_probe_names_the_backend_the_framework_depends_on() -> None:
    """The framework at HEAD depends on pyaml-cs-osprey, so Lane A scans and walks
    it as well as osprey -- and osprey_facility, installed beside it here."""
    info = lane.probe(Path(sys.executable), "osprey_connectors", "osprey_facility")
    assert info["packages"][:2] == ["osprey", "pyaml_cs_osprey"]
    assert info["packages"] == ["osprey", "pyaml_cs_osprey", "osprey_facility"]
    assert info["origins"]["osprey_connectors"].endswith("osprey_connectors/__init__.py")
    assert info["origins"]["osprey_facility"].endswith("osprey_facility/__init__.py")
    assert info["facility"] is not None and info["connectors"] is not None


def test_the_suites_load_the_guard_and_deselect_the_source_text_tests() -> None:
    argv = lane.pytest_command(
        Path("py"), ["tests/a.py::t", "tests/b.py"], ["tests/connectors", "-n", "4"]
    )
    assert argv == [
        "py",
        "-m",
        "pytest",
        "-p",
        "connectors_compat_guard",
        "--deselect",
        "tests/a.py::t",
        "--deselect",
        "tests/b.py",
        "tests/connectors",
        "-n",
        "4",
    ]


@pytest.mark.parametrize("package", LANES)
def test_a_release_tree_is_pointed_at_the_installed_package(
    package, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = package.import_name
    purelib = tmp_path / "site-packages"
    (purelib / name).mkdir(parents=True)
    (purelib / name / "__init__.py").write_text("HEAD = True\n")
    release = tmp_path / "release"
    stale = release / package.package_dir / "src" / name
    stale.mkdir(parents=True)
    (stale / "old.py").write_text("")
    monkeypatch.setattr(
        lane,
        "probe",
        lambda python, *modules: {
            "purelib": str(purelib),
            "origins": {name: str(purelib / name / "__init__.py")},
        },
    )
    lane.replace_tree(package, tmp_path / "venv", release)
    assert sorted(p.name for p in stale.iterdir()) == ["__init__.py"]
    assert (stale / "__init__.py").read_text() == "HEAD = True\n"


@pytest.mark.parametrize("package", LANES)
def test_a_relative_interpreter_is_read_against_the_callers_directory(
    package, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The nightly passes ``--python release/.venv/bin/python --cwd release``; the child
    starts in ``release``, so the interpreter must not be looked up from there."""
    seen: dict[str, object] = {}

    def _fake_run(argv, cwd=None, **_kwargs):
        seen["argv"], seen["cwd"] = [str(a) for a in argv], cwd
        return 0

    monkeypatch.setattr(lane, "_run", _fake_run)
    monkeypatch.chdir(tmp_path)
    argv = ["--package", package.key, "pytest", "--python", "release/.venv/bin/python"]
    argv += ["--cwd", "release", "--installed", package.import_name, "--", "-q"]
    assert lane.main(argv) == 0
    assert seen["argv"][0] == str(tmp_path / "release" / ".venv" / "bin" / "python")
    assert Path(seen["cwd"]) == Path("release")


def _guarded(tmp_path: Path, installed: str) -> subprocess.CompletedProcess[str]:
    (tmp_path / "test_one.py").write_text("def test_one():\n    pass\n")
    env = {
        **os.environ,
        "PYTHONPATH": str(lane.PLUGIN_DIR),
        "CONNECTORS_COMPAT_INSTALLED": installed,
    }
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", lane.GUARD_PLUGIN, "-q", "-p", "no:cacheprovider"]
        + ["-c", os.devnull, str(tmp_path / "test_one.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_the_guard_passes_a_module_from_site_packages(tmp_path: Path) -> None:
    assert Path(sysconfig.get_path("purelib")).resolve() in Path(pytest.__file__).resolve().parents
    result = _guarded(tmp_path, "pytest")
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_guard_fails_a_module_from_anywhere_else(tmp_path: Path) -> None:
    result = _guarded(tmp_path, "json")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "json imports from" in result.stdout


# --- The scans a cell runs, on a venv that holds one scratch package ---------


def _scratch_venv(tmp_path: Path, package: str, source: str) -> Path:
    """A venv whose own site-packages holds ``package`` as ``source`` and
    sees everything else this environment has installed behind it."""
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=120
    )
    python = lane.venv_python(venv)
    purelib = Path(
        subprocess.run(
            [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        ).stdout.strip()
    )
    (purelib / package).mkdir(parents=True)
    (purelib / package / "__init__.py").write_text(textwrap.dedent(source))
    # addsitedir, not a bare path: this environment's editable installs are
    # .pth files of their own. Appended, so the scratch package comes first.
    here = sysconfig.get_path("purelib")
    (purelib / "zz_this_environment.pth").write_text(f"import site; site.addsitedir({here!r})\n")
    return python


def _scan_step(package, python: Path, packages: list[str]) -> subprocess.CompletedProcess[str]:
    commands = lane.check_commands(package, python, packages, python.parent, require_view=False)
    _, argv = commands[0]
    assert Path(argv[2]).name == "scan_framework_imports.py"
    return subprocess.run(argv, capture_output=True, text=True, timeout=300, check=False)


def test_the_connectors_lane_scans_the_installed_facility(tmp_path: Path) -> None:
    python = _scratch_venv(
        tmp_path, "osprey_facility", "from osprey_connectors.types import EPICS\n"
    )
    packages = lane.probe(python)["packages"]
    assert "osprey_facility" in packages
    result = _scan_step(lane.CONNECTORS, python, ["osprey_facility"])
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_connectors_lane_scans_the_installed_facility__mutation_imports_a_removed_name(
    tmp_path: Path,
) -> None:
    """A facility release that imports a name the connectors at HEAD no longer
    carry fails the connectors lane."""
    python = _scratch_venv(
        tmp_path,
        "osprey_facility",
        """\
        def load():
            from osprey_connectors.types import REMOVED_AT_HEAD
        """,
    )
    packages = lane.probe(python)["packages"]
    assert "osprey_facility" in packages
    result = _scan_step(lane.CONNECTORS, python, packages)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "osprey_facility/__init__.py:2: osprey_connectors.types:REMOVED_AT_HEAD" in result.stdout


def test_the_facility_lane_scans_the_installed_framework() -> None:
    result = _scan_step(lane.FACILITY, Path(sys.executable), list(lane.FACILITY.consumers))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "against osprey_facility" in result.stderr


def test_the_facility_lane_scans_the_installed_framework__mutation_imports_a_removed_name(
    tmp_path: Path,
) -> None:
    python = _scratch_venv(tmp_path, "osprey", "from osprey_facility import REMOVED_AT_HEAD\n")
    result = _scan_step(lane.FACILITY, python, list(lane.FACILITY.consumers))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "osprey/__init__.py:1: osprey_facility:REMOVED_AT_HEAD" in result.stdout


# --- The connectors flags, kept as aliases ----------------------------------


@pytest.mark.parametrize(
    ("argv", "step", "expected"),
    [
        pytest.param(
            ["install", "--venv", "v", "--framework", "2026.11.2", "--connectors", "head"],
            "install",
            ("v", "2026.11.2", "head", None),
            id="install",
        ),
        pytest.param(["check", "--venv", "v", "--workdir", "w"], "check", ("v", "w"), id="check"),
        pytest.param(
            ["assert-installed", "--python", "p", "--module", "osprey_connectors"]
            + ["--connectors-version", "2026.11.15"],
            "assert_installed",
            ("p", ["osprey_connectors"], "2026.11.15"),
            id="assert-installed",
        ),
        pytest.param(
            ["replace-connectors-tree", "--venv", "v", "--release-dir", "r"],
            "replace_tree",
            ("v", "r"),
            id="replace-connectors-tree",
        ),
    ],
)
def test_the_connectors_flags_reach_the_same_steps(
    argv: list[str], step: str, expected: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The connectors flags (``--connectors``, ``--connectors-version``,
    ``replace-connectors-tree``) stay aliases; each reaches the connectors
    cell's step with the same values."""
    calls: list[tuple] = []
    monkeypatch.setattr(lane, step, lambda *args: calls.append(args) or 0)
    assert lane.main(["--package", "connectors", *argv]) == 0
    (call,) = calls
    assert call[0] is lane.CONNECTORS
    assert tuple(str(a) if isinstance(a, Path) else a for a in call[1:]) == expected


def test_the_connectors_lane_runs_from_the_command_line(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(_CI / "compat_lane.py"), "--package", "connectors", "install"]
        + ["--venv", str(tmp_path / "v"), "--framework", "2026.11.2", "--connectors", "HEAD"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 1
    assert "::error::not a release version: 'HEAD'" in result.stderr
    assert not (tmp_path / "v").exists()
