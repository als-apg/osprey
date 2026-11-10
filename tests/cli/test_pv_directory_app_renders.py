"""Of the shared logins, only the channel finder carries the ``pv-directory`` server.

One ``osprey init --preset demo`` materializes the three shared logins; the host's
profile then replaces the preset's ``pv_directory`` section with a facility
directory and one build renders every login. The section reaches each login's
render through the deployment's config, so the logbook and knowledge logins
switch the server off by name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.init_cmd import init
from osprey_connectors.workspace import BUILD_DIR_NAME
from tests.fixtures.lifecycle_repo import preserved_environ

SERVER = "pv-directory"
LOGINS = ("channel-finder", "logbook", "knowledge")
REPO = "demo"
SECTION = {"type": "channelfinder", "settings": {"url": "https://directory.example/api"}}


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The demo preset with a configured ``pv_directory`` section, built once."""
    outside = tmp_path_factory.mktemp("pv-directory-logins")
    repo = outside / REPO
    with pytest.MonkeyPatch.context() as patch, preserved_environ():
        patch.chdir(outside)
        made = CliRunner().invoke(init, [REPO, "--preset", "demo", "--no-git"])
        assert made.exit_code == 0, made.output
        profile = repo / "profile.yml"
        document = yaml.safe_load(profile.read_text(encoding="utf-8"))
        document["config"]["pv_directory"] = SECTION
        profile.write_text(yaml.safe_dump(document, sort_keys=False), "utf-8")
        result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output
    return repo / BUILD_DIR_NAME


def _servers(render: Path) -> set[str]:
    return set(json.loads((render / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"])


def test_the_host_render_carries_the_server(built: Path) -> None:
    assert SERVER in _servers(built)


def test_only_the_channel_finder_login_carries_the_server(built: Path) -> None:
    logins = {login: SERVER in _servers(built / f"{REPO}-{login}") for login in LOGINS}

    assert logins == {"channel-finder": True, "logbook": False, "knowledge": False}


@pytest.mark.parametrize("login", ["logbook", "knowledge"])
def test_an_opted_out_login_names_the_switch_in_its_config(built: Path, login: str) -> None:
    config = yaml.safe_load((built / f"{REPO}-{login}" / "config.yml").read_text(encoding="utf-8"))

    assert config["claude_code"]["servers"][SERVER]["enabled"] is False
