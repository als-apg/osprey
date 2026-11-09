"""``osprey build`` of a deployment that mounts its machine from a machine root.

``machine: <dir>`` names the facility directory of a machine repository inside
the deployment repo. The build reads every machine fact from it and only the
deployment's own content (``knowledge/``, ``ariel/``, ``benchmarks/``) from
``data/facility``. The render assembles the two into one ``data/facility``
again, so the runtime sees one tree as it does for a single-tree deployment.

The guards that protect deployment files keep checking the deployment; only
their remedies name the machine. A deployment file restating a machine fact
stops the build before the venv, and a start set the deployment inherits from
the ``demo`` preset is refused until the profile cancels it with ``null``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, NoReturn

import pytest
import yaml

from osprey.cli.build_persistence import _profile_known_root_entries
from osprey.cli.build_profile_model import BuildProfile
from osprey.facility.roots import DEPLOYMENT_CONTENT_DIRS
from osprey_connectors.workspace import BUILD_DIR_NAME
from tests._builds import init_project, run_build

pytestmark = [pytest.mark.slow]

#: The project name, and the repo's directory name.
PROJECT = "demo"

#: Where the machine root is mounted, relative to the repo.
MACHINE_DIR = "machine"

#: A knowledge page the machine repository ships for its own VA-only setup.
MACHINE_KNOWLEDGE = "knowledge/machine-only-note.md"

#: The scenario the machine marks as its start set.
MARKED_SCENARIO = "rf-thermal"

#: The start-set key the demo preset states.
DEFAULT_SCENARIOS_KEY = "simulation.default_scenarios"


def _git_init(directory: Path) -> None:
    subprocess.run(["git", "init", "-q", str(directory)], check=True)


@pytest.fixture(scope="module")
def initialised(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A demo repo whose machine facts sit in a mounted machine root.

    The machine root is a git repository (it holds ``.git``), ships its own
    knowledge page and marks one scenario ``start: true``. The deployment's
    ``data/facility`` keeps only its content directories.
    """
    repo = init_project(tmp_path_factory.mktemp("mounted"), "demo", PROJECT)
    facility = repo / "data" / "facility"
    machine = repo / MACHINE_DIR
    machine.mkdir()
    for entry in sorted(facility.iterdir()):
        if entry.name not in DEPLOYMENT_CONTENT_DIRS:
            shutil.move(str(entry), machine / entry.name)
    _git_init(machine)
    note = machine / MACHINE_KNOWLEDGE
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("# The machine repository's own note\n", encoding="utf-8")
    scenario = machine / "scenarios" / f"{MARKED_SCENARIO}.yaml"
    scenario.write_text(
        scenario.read_text(encoding="utf-8").rstrip("\n") + "\nstart: true\n", encoding="utf-8"
    )
    _edit_profile(repo, machine=MACHINE_DIR)
    return repo


def _edit_profile(repo: Path, **top: Any) -> None:
    profile = repo / "profile.yml"
    data = yaml.safe_load(profile.read_text(encoding="utf-8"))
    data.update(top)
    profile.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _state_config(repo: Path, key: str, value: Any) -> None:
    profile = repo / "profile.yml"
    data = yaml.safe_load(profile.read_text(encoding="utf-8"))
    data.setdefault("config", {})[key] = value
    profile.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


@pytest.fixture
def repo(initialised: Path, tmp_path: Path) -> Path:
    """A fresh copy of the mounted repo, so each test edits its own."""
    copy = tmp_path / PROJECT
    shutil.copytree(initialised, copy, symlinks=True)
    return copy


def _files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _rendered_start_set(repo: Path) -> Any:
    config = yaml.safe_load((repo / BUILD_DIR_NAME / "config.yml").read_text(encoding="utf-8"))
    if DEFAULT_SCENARIOS_KEY in config:
        return config[DEFAULT_SCENARIOS_KEY]
    return config["simulation"]["default_scenarios"]


# --- a mounted build ----------------------------------------------------------------


@pytest.fixture(scope="module")
def built(initialised: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Any]:
    """The mounted repo, built once with the preset's start set cancelled."""
    repo = tmp_path_factory.mktemp("built") / PROJECT
    shutil.copytree(initialised, repo, symlinks=True)
    _state_config(repo, DEFAULT_SCENARIOS_KEY, None)
    result = run_build(repo)
    assert result.exit_code == 0, result.output
    return repo, result


def test_the_render_holds_the_machine_files_and_the_deployment_knowledge(
    built: tuple[Path, Any],
) -> None:
    repo, _result = built
    rendered = _files(repo / BUILD_DIR_NAME / "data" / "facility")
    machine = repo / MACHINE_DIR
    machine_facts = {
        rel
        for rel in _files(machine)
        if not rel.startswith(".git/") and rel.split("/", 1)[0] not in DEPLOYMENT_CONTENT_DIRS
    }
    knowledge = {
        rel for rel in _files(repo / "data" / "facility") if rel.split("/", 1)[0] == "knowledge"
    }

    assert machine_facts and knowledge
    assert machine_facts <= rendered
    assert knowledge <= rendered
    assert (repo / BUILD_DIR_NAME / "data" / "facility" / "identity.yaml").read_bytes() == (
        machine / "identity.yaml"
    ).read_bytes()


def test_the_render_carries_no_vcs_metadata_and_no_machine_side_knowledge(
    built: tuple[Path, Any],
) -> None:
    repo, _result = built
    rendered = repo / BUILD_DIR_NAME / "data" / "facility"

    assert (repo / MACHINE_DIR / ".git").exists()
    assert not any(".git" in Path(rel).parts for rel in _files(rendered))
    assert not (rendered / ".git").exists()
    assert not (rendered / MACHINE_KNOWLEDGE).exists()


def test_a_null_start_set_renders_the_machines_marked_scenarios(
    built: tuple[Path, Any],
) -> None:
    repo, _result = built

    assert _rendered_start_set(repo) == [MARKED_SCENARIO]


def test_a_mounted_build_prints_no_unrecognized_root_entry_warning(
    built: tuple[Path, Any],
) -> None:
    _repo, result = built

    assert "unrecognized top-level" not in result.output


def test_the_machine_root_is_a_known_root_entry() -> None:
    profile = BuildProfile(name=PROJECT, data="data", machine="vendor/als-machine/facility")

    assert "vendor" in _profile_known_root_entries(profile, None)


# --- stops ----------------------------------------------------------------------------


def test_a_preset_start_set_is_refused_when_mounted(repo: Path) -> None:
    result = run_build(repo)

    assert result.exit_code == 1, result.output
    last = result.stderr.splitlines()[-1]
    assert last.startswith(f"facility: profile-invalid: path {DEFAULT_SCENARIOS_KEY} — ")
    assert f"`{DEFAULT_SCENARIOS_KEY}: null`" in last


def test_a_shipped_limits_file_stops_with_the_machine_limits_remedy(repo: Path) -> None:
    (repo / "data" / "channel_limits.json").write_text('{"_version": "4.0"}\n', encoding="utf-8")

    result = run_build(repo)

    assert result.exit_code == 1, result.output
    assert result.stderr == (
        "facility: profile-invalid: path data/channel_limits.json — the build writes "
        "data/channel_limits.json from machine/limits.yaml; fix: move its limits into "
        "machine/limits.yaml and remove data/channel_limits.json\n"
    )


def test_a_restated_record_stops_before_the_venv(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from click.testing import CliRunner

    from osprey.cli import build_cmd

    def venv_spy(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("the project venv was created before the machine-restated stop")

    monkeypatch.setattr(build_cmd, "_create_project_venv", venv_spy)
    restated = repo / "data" / "facility" / "identity.yaml"
    shutil.copy2(repo / MACHINE_DIR / "identity.yaml", restated)

    result = CliRunner().invoke(build_cmd.build, ["--repo", str(repo), "--skip-lifecycle"])

    assert result.exit_code == 1, result.output
    assert result.stderr == (
        "facility: machine-restated: path data/facility/identity.yaml — the mounted machine "
        "states machine/identity.yaml, and a mounted deployment's data/facility/ holds only "
        "knowledge/, ariel/, benchmarks/; fix: remove data/facility/identity.yaml; edit "
        "machine/identity.yaml in the machine repository instead\n"
    )
    assert not (repo / BUILD_DIR_NAME / "config.yml").exists()
