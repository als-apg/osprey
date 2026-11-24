"""Every configuration file a real build writes carries the one protected-path set.

The build computes the set once, across the deployment's profile and every
login file, and writes it at the one place every render goes through. The other
``config.yml`` files under ``build/`` — the image copies, the staged service
copies, the dispatch worker's — are re-serialisations of those renders, so they
carry it too. These tests prove that by building a deployment with several
logins, a dispatch worker and an application for real, then reading every
``config.yml`` under ``build/`` through the executor's own reader.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.utils.workspace import ProtectedPathsState, read_protected_paths
from tests.cli._lifecycle_build import invoke_build_in
from tests.fixtures.lifecycle_repo import (
    EXEMPLAR_DIRNAME,
    build_exemplar_repo,
    preserved_environ,
)

pytestmark = pytest.mark.slow

#: The plan directory the base profile states, in the profile's own spelling.
BASE_PLAN_DIR = "plans/"

#: The plan directory one login file states under its ``config:`` block.
LOGIN_PLAN_DIR = "login_plans"

#: The login file that states it.
LOGIN_WITH_PLANS = "readwrite"

#: The application the deployment installs.
APP: dict[str, Any] = {
    "name": "knob",
    "containers": {
        "knob": {
            "runtime": "osprey",
            "command": ["python", "-m", "knob_app.server"],
            "health": "/health",
        },
    },
}


def _load(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return document


def _dump(path: Path, document: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _deployment(dest: Path, *, base_plan_dir: str = BASE_PLAN_DIR) -> Path:
    """The exemplar with a plan directory in its profile and one in a login, and an app."""
    repo = build_exemplar_repo(dest)
    profile = _load(repo / "profile.yml")
    profile["bluesky"]["plan_dir"] = base_plan_dir
    profile["apps"] = {APP["name"]: {}}
    _dump(repo / "profile.yml", profile)

    login_path = repo / "tiers" / f"{LOGIN_WITH_PLANS}.yml"
    login = _load(login_path)
    login.setdefault("config", {})["bluesky.plan_dirs"] = [LOGIN_PLAN_DIR]
    _dump(login_path, login)

    app_dir = repo / "apps" / APP["name"]
    app_dir.mkdir(parents=True, exist_ok=True)
    _dump(app_dir / "app.yml", APP)

    (repo / base_plan_dir).mkdir(parents=True, exist_ok=True)
    (repo / LOGIN_PLAN_DIR).mkdir(parents=True, exist_ok=True)
    return repo


def _build(repo: Path):
    with preserved_environ():
        return invoke_build_in(repo)


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """One real build of the deployment, shared by the module."""
    repo = _deployment(tmp_path_factory.mktemp("protected-render") / EXEMPLAR_DIRNAME)
    result = _build(repo)
    assert result.exit_code == 0, result.output
    return repo


def _sets(repo: Path) -> dict[str, tuple[str, ...]]:
    """Every ``config.yml`` under ``build/``, build-relative, and the set it carries."""
    sets: dict[str, tuple[str, ...]] = {}
    for path in sorted((repo / "build").rglob("config.yml")):
        if ".venv" in path.parts:
            continue
        reading = read_protected_paths(_load(path))
        relative = path.relative_to(repo / "build").as_posix()
        assert reading.state is ProtectedPathsState.PRESENT, f"{relative}: {reading.state}"
        sets[relative] = reading.entries
    return sets


def test_the_build_writes_every_render_and_copy(built: Path) -> None:
    """The shape of the build this module reads: logins, a worker, an app, image copies."""
    logins = sorted(path.stem for path in (built / "tiers").glob("*.yml"))
    assert len(logins) >= 2 and LOGIN_WITH_PLANS in logins
    documents = _sets(built)
    assert "config.yml" in documents
    for login in logins:
        assert f"{EXEMPLAR_DIRNAME}-{login}/config.yml" in documents
    assert any(name.startswith(".image/") for name in documents), sorted(documents)
    assert any(name.startswith("services/") for name in documents), sorted(documents)


def test_every_config_carries_the_identical_set(built: Path) -> None:
    documents = _sets(built)
    distinct = set(documents.values())
    assert len(distinct) == 1, documents


def test_the_set_is_the_union_of_profile_and_logins(built: Path) -> None:
    entries = _sets(built)["config.yml"]
    assert BASE_PLAN_DIR in entries
    assert LOGIN_PLAN_DIR in entries
    # The inputs the build already enumerates: the login files, the triggers.
    for login in (built / "tiers").glob("*.yml"):
        assert f"tiers/{login.name}" in entries
    assert "triggers.yml" in entries


def test_in_repo_inputs_are_relative_to_the_root(built: Path) -> None:
    entries = _sets(built)["config.yml"]
    root = str(built.resolve())
    assert not [entry for entry in entries if entry.startswith((root, str(built)))]
    assert not [entry for entry in entries if Path(entry).is_absolute()], entries


def test_a_protected_path_in_the_agent_data_zone_stops_the_build(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repo = _deployment(tmp_path / EXEMPLAR_DIRNAME, base_plan_dir="var/agent_data/plans")
    shutil.rmtree(repo / "build", ignore_errors=True)

    result = _build(repo)

    assert result.exit_code != 0
    said = result.output + caplog.text
    assert "bluesky.plan_dir" in said
    assert "var/agent_data/plans" in said
    assert not (repo / "build" / "config.yml").exists()
