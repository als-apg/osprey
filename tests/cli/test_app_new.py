"""``osprey app new``: the shipped example, copied under a new name, listed and buildable."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.app_cmd import EXAMPLE_APP_DIR, app
from tests.cli._lifecycle_build import invoke_build, rendered_config

EXAMPLE = "knob-example"
VENDOR_IGNORE = "/apps/*/panel/vendor/"


def _new(runner: CliRunner, repo: Path, *args: str) -> Result:
    return runner.invoke(app, ["new", *args, "--repo", str(repo)])


def _check(runner: CliRunner, repo: Path) -> Result:
    return runner.invoke(app, ["check", "--repo", str(repo)])


def _listed(repo: Path) -> dict[str, Any]:
    profile = yaml.safe_load((repo / "profile.yml").read_text(encoding="utf-8"))
    return profile.get("apps") or {}


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def no_js(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Node present and every JS tool passing, so ``app check`` judges the rest."""

    def run(argv: Any, _cwd: Path, _env: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    monkeypatch.setattr("osprey.cli.app_cmd._run", run)
    monkeypatch.setattr("osprey.cli.app_cmd.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def test_the_copy_is_the_example_under_the_new_name(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    result = _new(runner, lifecycle_repo, "demo-x")

    assert result.exit_code == 0, result.output
    copy = _tree(lifecycle_repo / "apps" / "demo-x")
    example = _tree(EXAMPLE_APP_DIR)
    renamed = {
        path.replace("knob_example", "demo_x"): text.replace(b"knob-example", b"demo-x").replace(
            b"knob_example", b"demo_x"
        )
        for path, text in example.items()
        if "__pycache__" not in path and not path.endswith((".pyc", ".pyo"))
    }
    assert copy == renamed
    assert "demo_x/__init__.py" in copy
    for path, text in copy.items():
        assert b"knob-example" not in text and b"knob_example" not in text, path


def test_the_copy_pins_no_port_and_states_no_write_policy(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    assert _new(runner, lifecycle_repo, "demo-x").exit_code == 0

    manifest = yaml.safe_load(
        (lifecycle_repo / "apps" / "demo-x" / "app.yml").read_text(encoding="utf-8")
    )
    assert manifest["name"] == "demo-x"
    assert all("port" not in container for container in manifest["containers"].values())
    text = (lifecycle_repo / "apps" / "demo-x" / "app.yml").read_text(encoding="utf-8")
    assert "writes_enabled" not in text and "arm_writes" not in text


def test_byte_code_of_the_example_is_never_copied(
    runner: CliRunner, lifecycle_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "example"
    shutil.copytree(EXAMPLE_APP_DIR, source)
    cache = source / "knob_example" / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "tools.cpython-313.pyc").write_bytes(b"knob_example")
    (source / "knob_example" / "stray.pyo").write_bytes(b"knob_example")
    monkeypatch.setattr("osprey.cli.app_cmd.EXAMPLE_APP_DIR", source)

    assert _new(runner, lifecycle_repo, "demo-x").exit_code == 0

    copied = _tree(lifecycle_repo / "apps" / "demo-x")
    assert not [path for path in copied if "__pycache__" in path or path.endswith((".pyc", ".pyo"))]


def test_the_new_application_is_listed_beside_the_example(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    before = (lifecycle_repo / "profile.yml").read_text(encoding="utf-8")
    assert EXAMPLE in _listed(lifecycle_repo)

    result = _new(runner, lifecycle_repo, "demo-x")

    assert result.exit_code == 0, result.output
    assert _listed(lifecycle_repo) == {**_listed_from(before), "demo-x": {}}
    after = (lifecycle_repo / "profile.yml").read_text(encoding="utf-8")
    comments = [line for line in before.splitlines() if line.lstrip().startswith("#")]
    assert comments == [line for line in after.splitlines() if line.lstrip().startswith("#")]


def _listed_from(text: str) -> dict[str, Any]:
    return yaml.safe_load(text).get("apps") or {}


def test_an_existing_directory_is_refused_without_force(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    target = lifecycle_repo / "apps" / "demo-x"
    target.mkdir(parents=True)
    (target / "notes.txt").write_text("mine\n", encoding="utf-8")
    profile = (lifecycle_repo / "profile.yml").read_text(encoding="utf-8")

    result = _new(runner, lifecycle_repo, "demo-x")

    assert result.exit_code != 0
    assert "--force" in result.output
    assert (target / "notes.txt").read_text(encoding="utf-8") == "mine\n"
    assert (lifecycle_repo / "profile.yml").read_text(encoding="utf-8") == profile


def test_force_replaces_the_directory_and_lists_the_application_once(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    assert _new(runner, lifecycle_repo, "demo-x").exit_code == 0
    stray = lifecycle_repo / "apps" / "demo-x" / "notes.txt"
    stray.write_text("old\n", encoding="utf-8")

    result = _new(runner, lifecycle_repo, "demo-x", "--force")

    assert result.exit_code == 0, result.output
    assert not stray.exists()
    assert list(_listed(lifecycle_repo)).count("demo-x") == 1


def test_an_application_already_listed_keeps_its_entry(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    profile = yaml.safe_load((lifecycle_repo / "profile.yml").read_text(encoding="utf-8"))
    profile["apps"]["demo-x"] = {"containers": {"demo-x": {"env": {"A": "1"}}}}
    (lifecycle_repo / "profile.yml").write_text(yaml.safe_dump(profile), encoding="utf-8")

    assert _new(runner, lifecycle_repo, "demo-x").exit_code == 0

    assert _listed(lifecycle_repo)["demo-x"] == {"containers": {"demo-x": {"env": {"A": "1"}}}}


def test_the_gitignore_gains_the_vendor_entry_once(runner: CliRunner, lifecycle_repo: Path) -> None:
    gitignore = lifecycle_repo / ".gitignore"
    assert VENDOR_IGNORE in gitignore.read_text(encoding="utf-8").splitlines()
    gitignore.write_text("/build/\n", encoding="utf-8")

    assert _new(runner, lifecycle_repo, "demo-x").exit_code == 0
    assert _new(runner, lifecycle_repo, "demo-y").exit_code == 0

    assert gitignore.read_text(encoding="utf-8").splitlines() == ["/build/", VENDOR_IGNORE]


def test_a_missing_gitignore_is_written_with_the_vendor_entry(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    (lifecycle_repo / ".gitignore").unlink()

    assert _new(runner, lifecycle_repo, "demo-x").exit_code == 0

    assert (lifecycle_repo / ".gitignore").read_text(encoding="utf-8").splitlines() == [
        VENDOR_IGNORE
    ]


@pytest.mark.parametrize("name", ["Demo", "demo_x", "1demo", "-demo", "demo/x", ""])
def test_a_name_that_is_no_application_name_is_refused(
    runner: CliRunner, lifecycle_repo: Path, name: str
) -> None:
    before = sorted(path.name for path in (lifecycle_repo / "apps").iterdir())

    result = _new(runner, lifecycle_repo, name)

    assert result.exit_code == 2
    assert "NAME" in result.output
    assert sorted(path.name for path in (lifecycle_repo / "apps").iterdir()) == before
    assert list(_listed(lifecycle_repo)) == [EXAMPLE]


def test_a_directory_outside_any_repo_is_refused(runner: CliRunner, tmp_path: Path) -> None:
    result = _new(runner, tmp_path, "demo-x")

    assert result.exit_code == 1
    assert "No OSPREY deployment repo" in result.output
    assert not (tmp_path / "apps").exists()


@pytest.mark.usefixtures("no_js")
def test_the_copy_passes_app_check_and_builds(runner: CliRunner, lifecycle_repo: Path) -> None:
    assert _new(runner, lifecycle_repo, "demo-x").exit_code == 0

    checked = _check(runner, lifecycle_repo)
    built = invoke_build(lifecycle_repo)

    assert checked.exit_code == 0, checked.output
    assert "2 application(s) pass" in checked.output
    assert built.exit_code == 0, built.output
    assert "demo-x" in yaml.safe_dump(rendered_config(lifecycle_repo))


@pytest.mark.skipif(
    os.environ.get("OSPREY_JS_CHECK") != "1", reason="fetches a JS toolchain: OSPREY_JS_CHECK=1"
)
def test_the_pinned_toolchain_passes_the_shipped_example_and_its_copy(
    runner: CliRunner, lifecycle_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert _new(runner, lifecycle_repo, "demo-x").exit_code == 0

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert "2 application(s) pass" in result.output
