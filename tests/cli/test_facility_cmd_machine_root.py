"""``osprey facility`` on a deployment that mounts its machine from a machine root.

With ``machine: <dir>`` in the profile, the imports write their layer files and
seeds into the machine root, while the deployment's own content directories
(``knowledge/``, ``ariel/``, ``benchmarks/``) stay under ``data/facility``: a
bundled page there is replaced like any bundled file, and the knowledge index
is settled there. ``facility validate`` stops on a deployment file that
restates a machine fact.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner, Result

from osprey.cli.main import cli
from osprey.facility.roots import DEPLOYMENT_CONTENT_DIRS
from tests._builds import init_project

pytestmark = [pytest.mark.slow]

#: The project name, and the repo's directory name.
PROJECT = "demo"

#: Where the machine root is mounted, relative to the repo.
MACHINE_DIR = "machine"

#: A knowledge page the machine repository ships for its own use.
MACHINE_KNOWLEDGE = "knowledge/machine-only-note.md"

#: A bundled knowledge page of the demo preset, in the deployment's content root.
BUNDLED_PAGE = "knowledge/devices/bpm.md"

#: The start-set key the demo preset states.
DEFAULT_SCENARIOS_KEY = "simulation.default_scenarios"


@pytest.fixture(scope="module")
def initialised(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A demo repo whose machine facts sit in a mounted machine root.

    The machine root is a git repository and ships a knowledge page of its
    own; the deployment's ``data/facility`` keeps only its content directories.
    """
    repo = init_project(tmp_path_factory.mktemp("mounted"), "demo", PROJECT)
    facility = repo / "data" / "facility"
    machine = repo / MACHINE_DIR
    machine.mkdir()
    for entry in sorted(facility.iterdir()):
        if entry.name not in DEPLOYMENT_CONTENT_DIRS:
            shutil.move(str(entry), machine / entry.name)
    subprocess.run(["git", "init", "-q", str(machine)], check=True)
    note = machine / MACHINE_KNOWLEDGE
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("# The machine repository's own note\n", encoding="utf-8")
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
    return copy.resolve()


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _directories(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()}


def _written(result: Result) -> list[str]:
    return [line[6:] for line in result.stdout.splitlines() if line.startswith("wrote ")]


def _import_list(repo: Path, tmp_path: Path) -> Result:
    listing = tmp_path / "channels.csv"
    listing.write_text("address,device,s,model\nSR:BPM1:X,BPM1,1.5,ring\n", encoding="utf-8")
    return CliRunner().invoke(
        cli, ["facility", "import", "list", str(listing), "--repo", str(repo)]
    )


def _validate(repo: Path) -> Result:
    return CliRunner().invoke(cli, ["facility", "validate", "--repo", str(repo)])


# --- import list --------------------------------------------------------------------


def test_import_list_writes_its_layer_under_the_machine_root(repo: Path, tmp_path: Path) -> None:
    result = _import_list(repo, tmp_path)

    assert result.exit_code == 0, result.output
    machine = repo / MACHINE_DIR
    assert (machine / "imported" / "list" / "channels.yaml").is_file()
    assert (machine / "imported" / "list" / "devices.yaml").is_file()
    assert not (repo / "data" / "facility" / "imported").exists()
    assert f"{MACHINE_DIR}/imported/list/channels.yaml" in _written(result)


def test_import_list_settles_the_knowledge_bundle_in_the_content_root(
    repo: Path, tmp_path: Path
) -> None:
    result = _import_list(repo, tmp_path)

    assert result.exit_code == 0, result.output
    index = repo / "data" / "facility" / "knowledge" / "index.md"
    assert index.is_file()
    assert "data/facility/knowledge/index.md" in _written(result)
    assert not (repo / MACHINE_DIR / "knowledge" / "index.md").exists()


def test_import_list_replaces_a_bundled_page_in_the_content_root(
    repo: Path, tmp_path: Path
) -> None:
    content = repo / "data" / "facility"
    assert (content / BUNDLED_PAGE).is_file()

    result = _import_list(repo, tmp_path)

    assert result.exit_code == 0, result.output
    assert not (content / BUNDLED_PAGE).exists()
    assert "import list: replaced the bundled example facility" in result.stderr
    # Every bundled file is gone from both roots: only the settled index and
    # the import's own files are left of the deployment's facility.
    assert set(_snapshot(content)) == {"knowledge/index.md"}


def test_import_list_leaves_the_deployments_other_files_untouched(
    repo: Path, tmp_path: Path
) -> None:
    machine = repo / MACHINE_DIR
    content = repo / "data" / "facility"
    own_page = content / "knowledge" / "our-notes.md"
    own_page.write_text("# Our notes\n\nWritten by the deployment.\n", encoding="utf-8")
    own_machine_file = machine / "README.md"
    own_machine_file.write_text("# The machine repository\n", encoding="utf-8")
    profile = (repo / "profile.yml").read_bytes()
    git = _snapshot(machine / ".git")
    git_dirs = _directories(machine / ".git")
    note = (machine / MACHINE_KNOWLEDGE).read_bytes()

    result = _import_list(repo, tmp_path)

    assert result.exit_code == 0, result.output
    assert own_page.read_text(encoding="utf-8") == "# Our notes\n\nWritten by the deployment.\n"
    assert own_machine_file.read_text(encoding="utf-8") == "# The machine repository\n"
    assert (repo / "profile.yml").read_bytes() == profile
    assert _snapshot(machine / ".git") == git
    assert _directories(machine / ".git") == git_dirs
    # The machine's own knowledge directory is not the deployment's content:
    # the import neither classifies nor removes it.
    assert (machine / MACHINE_KNOWLEDGE).read_bytes() == note


# --- validate -----------------------------------------------------------------------


def test_validate_reports_a_deployment_file_that_restates_the_machine(repo: Path) -> None:
    restated = repo / "data" / "facility" / "records" / "channels.yaml"
    restated.parent.mkdir(parents=True)
    restated.write_text("channels: []\n", encoding="utf-8")

    result = _validate(repo)

    assert result.exit_code == 1, result.output
    assert "machine-restated" in result.stderr
    assert "records/channels.yaml" in result.stderr


def test_validate_passes_a_mounted_deployment(repo: Path) -> None:
    # The demo preset states a start set, which a mounted machine refuses
    # until the profile cancels it.
    _state_config(repo, DEFAULT_SCENARIOS_KEY, None)

    result = _validate(repo)

    assert result.exit_code == 0, result.output


# --- help ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [["validate"], ["import"], ["import", "mml"], ["import", "list"]],
    ids=" ".join,
)
def test_help_says_a_machine_profile_uses_that_root(command: list[str]) -> None:
    result = CliRunner().invoke(cli, ["facility", *command, "--help"])

    assert result.exit_code == 0, result.output
    assert "machine:" in result.stdout
    assert "machine root" in result.stdout
