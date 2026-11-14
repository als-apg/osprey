"""``osprey app new``: the shipped scaffold, copied under a new name, listed and buildable."""

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
from tests.fixtures.lifecycle_repo import ORBIT_BUMP, enable_orbit_bump

#: The scaffold ``app new`` copies, by the name its own ``app.yml`` declares.
TEMPLATE: str = yaml.safe_load((EXAMPLE_APP_DIR / "app.yml").read_text(encoding="utf-8"))["name"]
TEMPLATE_PACKAGE = TEMPLATE.replace("-", "_")
#: The application the lifecycle repo's profile already lists.
LISTED = ORBIT_BUMP
VENDOR_IGNORE = "/apps/*/panel/vendor/"


@pytest.fixture(autouse=True)
def _one_listed_application(request: pytest.FixtureRequest) -> None:
    """The lifecycle repo with orbit bump on, so ``app new`` adds beside a listed application."""
    if "lifecycle_repo" in request.fixturenames:
        enable_orbit_bump(request.getfixturevalue("lifecycle_repo"))


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


def test_the_copy_is_the_scaffold_under_the_new_name(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    result = _new(runner, lifecycle_repo, "demo-x")

    assert result.exit_code == 0, result.output
    copy = _tree(lifecycle_repo / "apps" / "demo-x")
    template = _tree(EXAMPLE_APP_DIR)
    dashed, package = TEMPLATE.encode(), TEMPLATE_PACKAGE.encode()
    renamed = {
        path.replace(TEMPLATE, "demo-x").replace(TEMPLATE_PACKAGE, "demo_x"): text.replace(
            dashed, b"demo-x"
        ).replace(package, b"demo_x")
        for path, text in template.items()
        if "__pycache__" not in path and not path.endswith((".pyc", ".pyo"))
    }
    assert copy == renamed
    assert "demo_x/__init__.py" in copy
    for path, text in copy.items():
        assert dashed not in text and package not in text, path
        assert TEMPLATE not in path and TEMPLATE_PACKAGE not in path, path
    assert TEMPLATE in result.output


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


def test_the_copy_carries_its_own_skill_under_the_new_name(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    assert _new(runner, lifecycle_repo, "demo-x").exit_code == 0

    app_dir = lifecycle_repo / "apps" / "demo-x"
    assert yaml.safe_load((app_dir / "app.yml").read_text(encoding="utf-8"))["skills"] == ["demo-x"]
    skill = (app_dir / "skills" / "demo-x" / "SKILL.md").read_text(encoding="utf-8")
    head = yaml.safe_load(skill.split("---\n")[1])
    assert head["name"] == "demo-x"
    # The skill names every tool the scaffold registers, with its panel equivalent.
    for tool in ("read_channel", "set_channel"):
        assert f"`{tool}" in skill


def test_byte_code_of_the_scaffold_is_never_copied(
    runner: CliRunner, lifecycle_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "scaffold"
    shutil.copytree(EXAMPLE_APP_DIR, source)
    cache = source / TEMPLATE_PACKAGE / "__pycache__"
    cache.mkdir(exist_ok=True)
    (cache / "tools.cpython-313.pyc").write_bytes(TEMPLATE_PACKAGE.encode())
    (source / TEMPLATE_PACKAGE / "stray.pyo").write_bytes(TEMPLATE_PACKAGE.encode())
    monkeypatch.setattr("osprey.cli.app_cmd.EXAMPLE_APP_DIR", source)

    assert _new(runner, lifecycle_repo, "demo-x").exit_code == 0

    copied = _tree(lifecycle_repo / "apps" / "demo-x")
    assert not [path for path in copied if "__pycache__" in path or path.endswith((".pyc", ".pyo"))]


def test_the_new_application_is_listed_beside_the_listed_one(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    before = (lifecycle_repo / "profile.yml").read_text(encoding="utf-8")
    assert LISTED in _listed(lifecycle_repo)
    assert TEMPLATE not in _listed(lifecycle_repo)

    result = _new(runner, lifecycle_repo, "demo-x")

    assert result.exit_code == 0, result.output
    assert _listed(lifecycle_repo) == {**_listed_from(before), "demo-x": {}}
    after = (lifecycle_repo / "profile.yml").read_text(encoding="utf-8")
    comments = [line for line in before.splitlines() if line.lstrip().startswith("#")]
    assert comments == [line for line in after.splitlines() if line.lstrip().startswith("#")]


def _listed_from(text: str) -> dict[str, Any]:
    return yaml.safe_load(text).get("apps") or {}


def test_the_scaffold_name_itself_is_a_valid_new_application(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    result = _new(runner, lifecycle_repo, TEMPLATE)

    assert result.exit_code == 0, result.output
    assert _tree(lifecycle_repo / "apps" / TEMPLATE) == {
        path: text
        for path, text in _tree(EXAMPLE_APP_DIR).items()
        if "__pycache__" not in path and not path.endswith((".pyc", ".pyo"))
    }
    assert _listed(lifecycle_repo)[TEMPLATE] == {}
    assert VENDOR_IGNORE in (lifecycle_repo / ".gitignore").read_text(encoding="utf-8").splitlines()


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


@pytest.mark.parametrize("force", [(), ("--force",)], ids=["plain", "force"])
def test_an_installed_applications_name_is_refused_before_anything_is_written(
    runner: CliRunner,
    lifecycle_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    force: tuple[str, ...],
) -> None:
    from tests.apps import _installed

    _installed.install(monkeypatch)
    profile = (lifecycle_repo / "profile.yml").read_text(encoding="utf-8")
    gitignore = (lifecycle_repo / ".gitignore").read_text(encoding="utf-8")

    result = _new(runner, lifecycle_repo, _installed.NAME, *force)

    assert result.exit_code != 0
    assert (
        f"apps/{_installed.NAME}/ has the name of the application {_installed.NAME} that "
        f"{_installed.DIST} {_installed.VERSION} installs; rename the directory and its "
        "app.yml name"
    ) in " ".join(result.output.split())
    assert not (lifecycle_repo / "apps" / _installed.NAME).exists()
    assert (lifecycle_repo / "profile.yml").read_text(encoding="utf-8") == profile
    assert (lifecycle_repo / ".gitignore").read_text(encoding="utf-8") == gitignore


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
    assert list(_listed(lifecycle_repo)) == [LISTED]


@pytest.mark.parametrize("name", ["class", "osprey", "osprey-connectors", "json"])
def test_a_name_whose_package_would_not_be_its_own_is_refused(
    runner: CliRunner, lifecycle_repo: Path, name: str
) -> None:
    result = _new(runner, lifecycle_repo, name)

    assert result.exit_code == 2
    assert "Python package" in result.output
    assert not (lifecycle_repo / "apps" / name).exists()
    assert list(_listed(lifecycle_repo)) == [LISTED]


@pytest.mark.parametrize(
    "profile",
    [
        "apps: [\n",
        "apps: [demo-x]\n",
        "- a\n",
        "[]\n",
        "false\n",
        "0\n",
        "apps: {}\napps: {}\n",
    ],
)
def test_force_over_an_unreadable_profile_leaves_the_application_alone(
    runner: CliRunner, lifecycle_repo: Path, profile: str
) -> None:
    target = lifecycle_repo / "apps" / "demo-x"
    target.mkdir(parents=True)
    (target / "notes.txt").write_text("mine\n", encoding="utf-8")
    (lifecycle_repo / "profile.yml").write_text(profile, encoding="utf-8")

    result = _new(runner, lifecycle_repo, "demo-x", "--force")

    assert result.exit_code == 2
    assert "profile.yml" in result.output
    assert (target / "notes.txt").read_text(encoding="utf-8") == "mine\n"
    assert (lifecycle_repo / "profile.yml").read_text(encoding="utf-8") == profile


def test_a_plain_file_at_the_target_is_refused_then_replaced_by_force(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    target = lifecycle_repo / "apps" / "demo-x"
    target.write_text("mine\n", encoding="utf-8")

    refused = _new(runner, lifecycle_repo, "demo-x")
    assert refused.exit_code == 2
    assert target.read_text(encoding="utf-8") == "mine\n"

    forced = _new(runner, lifecycle_repo, "demo-x", "--force")
    assert forced.exit_code == 0, forced.output
    assert (target / "app.yml").is_file()


def test_force_over_a_symlink_replaces_the_link_and_not_what_it_names(
    runner: CliRunner, lifecycle_repo: Path, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "notes.txt").write_text("mine\n", encoding="utf-8")
    target = lifecycle_repo / "apps" / "demo-x"
    target.symlink_to(elsewhere)

    result = _new(runner, lifecycle_repo, "demo-x", "--force")

    assert result.exit_code == 0, result.output
    assert not target.is_symlink() and (target / "app.yml").is_file()
    assert (elsewhere / "notes.txt").read_text(encoding="utf-8") == "mine\n"


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


@pytest.mark.usefixtures("no_js")
def test_a_copy_whose_panel_never_marks_itself_ready_is_refused(
    runner: CliRunner, lifecycle_repo: Path
) -> None:
    assert _new(runner, lifecycle_repo, "demo-x").exit_code == 0
    script = lifecycle_repo / "apps" / "demo-x" / "panel" / "panel.js"
    text = script.read_text(encoding="utf-8")
    script.write_text(
        text.replace(
            "import { markPageReady } from '/design-system/js/page-ready.js';\n", ""
        ).replace(".finally(() => markPageReady())", ""),
        encoding="utf-8",
    )

    checked = _check(runner, lifecycle_repo)

    assert checked.exit_code == 1
    assert "apps/demo-x/panel/index.html: " in checked.output
    assert "[page_ready]" in checked.output


@pytest.mark.skipif(
    os.environ.get("OSPREY_JS_CHECK") != "1", reason="fetches a JS toolchain: OSPREY_JS_CHECK=1"
)
def test_the_pinned_toolchain_passes_the_listed_app_and_the_scaffold_copy(
    runner: CliRunner, lifecycle_repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    assert _new(runner, lifecycle_repo, "demo-x").exit_code == 0

    result = _check(runner, lifecycle_repo)

    assert result.exit_code == 0, result.output
    assert "2 application(s) pass" in result.output
