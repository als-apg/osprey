"""The shared steps of a connectors compatibility cell build the commands the lanes need."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import sysconfig
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = REPO_ROOT / "scripts" / "ci" / "connectors_compat_lane.py"
_spec = importlib.util.spec_from_file_location("connectors_compat_lane", _MODULE_PATH)
lane = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lane)


@pytest.mark.parametrize("require_view", [False, True])
def test_every_check_runs_its_script_isolated(require_view: bool, tmp_path: Path) -> None:
    """``python -I``: the checkout and the script directory stay off ``sys.path``,
    so the release under test is the only framework importable."""
    python = tmp_path / "bin" / "python"
    commands = lane.check_commands(
        python, ["osprey", "pyaml_cs_osprey"], tmp_path / "w", require_view=require_view
    )
    scripts = [Path(argv[2]).name for _, argv in commands]
    assert scripts == ["scan_framework_imports.py", "import_walk.py", "contracts.py"]
    for _, argv in commands:
        assert argv[:2] == [str(python), "-I"]
        assert Path(argv[2]).is_file()
    for _, argv in commands[:2]:
        assert argv[3:] == ["--package", "osprey", "--package", "pyaml_cs_osprey"]
    assert ("--require-view" in commands[2][1]) is require_view


def _wheel(path: Path, version: str) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "osprey_connectors-x.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: osprey-connectors\nVersion: {version}\n",
        )
    return path


def test_head_is_the_built_wheel_at_the_version_its_metadata_states(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path / "osprey_connectors-1-py3-none-any.whl", "2026.10.1.post3+gabc")
    assert lane.connectors_target("head", tmp_path) == (str(wheel), "2026.10.1.post3+gabc")


def test_head_needs_exactly_one_built_wheel(tmp_path: Path) -> None:
    with pytest.raises(lane.LaneError, match="expected one"):
        lane.connectors_target("head", tmp_path)
    _wheel(tmp_path / "osprey_connectors-1-py3-none-any.whl", "1")
    _wheel(tmp_path / "osprey_connectors-2-py3-none-any.whl", "2")
    with pytest.raises(lane.LaneError, match="expected one"):
        lane.connectors_target("head", tmp_path)


def test_a_release_is_installed_from_pypi_by_version() -> None:
    assert lane.connectors_target("2026.9.0b3") == ("osprey-connectors==2026.9.0b3", "2026.9.0b3")


@pytest.mark.parametrize("connectors", ["HEAD", "latest", "dist/osprey_connectors-1.whl"])
def test_anything_else_is_refused(connectors: str) -> None:
    with pytest.raises(ValueError, match="not a release version"):
        lane.connectors_target(connectors)


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
    it as well as osprey."""
    info = lane.probe(Path(sys.executable), "osprey_connectors")
    assert info["packages"] == ["osprey", "pyaml_cs_osprey"]
    assert info["origins"]["osprey_connectors"].endswith("osprey_connectors/__init__.py")


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


def test_a_release_tree_is_pointed_at_the_installed_connectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    purelib = tmp_path / "site-packages"
    (purelib / "osprey_connectors").mkdir(parents=True)
    (purelib / "osprey_connectors" / "__init__.py").write_text("HEAD = True\n")
    release = tmp_path / "release"
    stale = release / "packages" / "osprey-connectors" / "src" / "osprey_connectors"
    stale.mkdir(parents=True)
    (stale / "old.py").write_text("")
    monkeypatch.setattr(
        lane,
        "probe",
        lambda python, *modules: {
            "purelib": str(purelib),
            "origins": {"osprey_connectors": str(purelib / "osprey_connectors" / "__init__.py")},
        },
    )
    lane.replace_connectors_tree(tmp_path / "venv", release)
    assert sorted(p.name for p in stale.iterdir()) == ["__init__.py"]
    assert (stale / "__init__.py").read_text() == "HEAD = True\n"


def test_a_relative_interpreter_is_read_against_the_callers_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The nightly passes ``--python release/.venv/bin/python --cwd release``; the child
    starts in ``release``, so the interpreter must not be looked up from there."""
    seen: dict[str, object] = {}

    def _fake_run(argv, cwd=None, **_kwargs):
        seen["argv"], seen["cwd"] = [str(a) for a in argv], cwd
        return 0

    monkeypatch.setattr(lane, "_run", _fake_run)
    monkeypatch.chdir(tmp_path)
    argv = ["pytest", "--python", "release/.venv/bin/python", "--cwd", "release"]
    assert lane.main([*argv, "--installed", "osprey_connectors", "--", "-q"]) == 0
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
