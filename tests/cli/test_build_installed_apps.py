"""``osprey build`` of a deployment whose profile lists an application an installed package ships."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.apps.origin import AppOrigin
from osprey.deployment.web_terminals.render_naming import render_project
from osprey.deployment.wheel_build import refuse_unshipped_apps
from osprey_connectors.errors import BuildProfileError
from tests.apps import _installed
from tests.apps._installed import DIRECTORY, DIST, NAME, VERSION, entry_point
from tests.cli._lifecycle_build import invoke_build_in, profile_hash, rendered_config
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

#: A login of the deployment that shows every listed application.
LOGIN = "asker"


def _append(repo: Path, block: dict[str, Any]) -> None:
    with (repo / "profile.yml").open("a", encoding="utf-8") as profile:
        profile.write("\n" + yaml.safe_dump(block, sort_keys=False))


def _deployment(root: Path, *, listed: bool, shipped: bool = True) -> Path:
    repo = build_exemplar_repo(root / EXEMPLAR_DIRNAME)
    if listed:
        _append(repo, {"apps": {NAME: {}}})
    if shipped:
        _append(repo, {"dependencies": [DIST]})
    (repo / "tiers" / f"{LOGIN}.yml").write_text(
        "name: Asker\nconfig:\n  control_system.writes: asked\n", encoding="utf-8"
    )
    return repo


def _skills(repo: Path, login: str = LOGIN) -> Path:
    project, _ = render_project(EXEMPLAR_DIRNAME, login)
    return repo / "build" / project / ".claude" / "skills"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    with pytest.MonkeyPatch.context() as patch:
        _installed.install(patch)
        repo = _deployment(tmp_path_factory.mktemp("installed"), listed=True)
        result = invoke_build_in(repo, "--skip-deps", "--skip-lifecycle")
        assert result.exit_code == 0, result.output
        yield repo


def test_the_listed_installed_app_projects_its_tool_server(built: Path) -> None:
    server = rendered_config(built)["claude_code"]["servers"][NAME]

    assert server["args"] == ["-m", "fixture_installed_app.tools"]
    assert server["permissions"]["allow"] == ["read_fixture"]


def test_the_listed_installed_app_projects_its_container(built: Path) -> None:
    service = rendered_config(built)["services"][NAME]

    assert service["app"] == NAME


def test_the_listed_installed_apps_skill_is_copied_from_its_package(built: Path) -> None:
    copied = _skills(built) / "fixture-installed" / "SKILL.md"

    assert (
        copied.read_bytes()
        == (DIRECTORY / "skills" / "fixture-installed" / "SKILL.md").read_bytes()
    )


def test_an_unlisted_installed_app_renders_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _installed.install(monkeypatch)
    repo = _deployment(tmp_path, listed=False, shipped=False)

    result = invoke_build_in(repo, "--skip-deps", "--skip-lifecycle")

    assert result.exit_code == 0, result.output
    render = rendered_config(repo)
    assert NAME not in render["claude_code"].get("servers", {})
    assert NAME not in render.get("services", {})
    assert not (_skills(repo) / "fixture-installed").exists()


def test_a_listed_installed_app_the_image_would_not_install_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _installed.install(monkeypatch)
    repo = _deployment(tmp_path, listed=True, shipped=False)

    result = invoke_build_in(repo, "--skip-deps", "--skip-lifecycle")

    assert result.exit_code != 0
    assert f"application '{NAME}' (package {DIST})" in caplog.text
    assert not (repo / "build" / "config.yml").exists()


def _origin(distribution: str) -> AppOrigin:
    return AppOrigin("installed", DIRECTORY, distribution, VERSION, "fixture_installed_app")


def test_the_image_requirement_names_the_application_and_its_distribution() -> None:
    refuse_unshipped_apps([f"{DIST}>=1"], {NAME: _origin(DIST)})
    refuse_unshipped_apps([], {NAME: _origin("osprey-framework")})

    with pytest.raises(BuildProfileError, match=f"'{NAME}' \\(package other-dist\\)"):
        refuse_unshipped_apps([DIST], {NAME: _origin("other-dist")})


def test_a_listed_installed_app_folds_its_distribution_version_into_the_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _deployment(tmp_path, listed=True)
    _installed.install(monkeypatch, entry_point(version="1.0.0"))
    before = profile_hash(repo)

    _installed.install(monkeypatch, entry_point(version="1.0.1"))

    assert profile_hash(repo) != before


def test_an_unlisted_installed_app_leaves_the_fingerprint_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _deployment(tmp_path, listed=False, shipped=False)
    _installed.install(monkeypatch, entry_point(name="other-installed"))
    before = profile_hash(repo)

    _installed.install(monkeypatch, entry_point(name="other-installed"), entry_point())

    assert profile_hash(repo) == before
