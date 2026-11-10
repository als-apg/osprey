"""Of the bundled apps, only the channel finder carries the ``pv-directory`` server.

One ``osprey init --preset demo`` materializes the three bundled apps; the host's
profile then configures a ``pv_directory`` section and one build renders every
app. The section reaches each app render through the deployment's config, so
the logbook and knowledge apps switch the server off by name.
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
APPS = ("channel-finder", "logbook", "knowledge")
REPO = "demo"
SECTION = (
    "  pv_directory:\n"
    "    type: channelfinder\n"
    "    settings:\n"
    "      url: https://directory.example/api\n"
)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The demo preset with a configured ``pv_directory`` section, built once."""
    outside = tmp_path_factory.mktemp("pv-directory-apps")
    repo = outside / REPO
    with pytest.MonkeyPatch.context() as patch, preserved_environ():
        patch.chdir(outside)
        made = CliRunner().invoke(init, [REPO, "--preset", "demo", "--no-git"])
        assert made.exit_code == 0, made.output
        profile = repo / "profile.yml"
        text = profile.read_text(encoding="utf-8")
        assert "\nconfig:\n" in text
        profile.write_text(text.replace("\nconfig:\n", "\nconfig:\n" + SECTION, 1), "utf-8")
        result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output
    return repo / BUILD_DIR_NAME


def _servers(render: Path) -> set[str]:
    return set(json.loads((render / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"])


def test_the_host_render_carries_the_server(built: Path) -> None:
    assert SERVER in _servers(built)


def test_only_the_channel_finder_app_carries_the_server(built: Path) -> None:
    apps = {app: SERVER in _servers(built / f"{REPO}-{app}") for app in APPS}

    assert apps == {"channel-finder": True, "logbook": False, "knowledge": False}


@pytest.mark.parametrize("app", ["logbook", "knowledge"])
def test_an_opted_out_app_names_the_switch_in_its_config(built: Path, app: str) -> None:
    config = yaml.safe_load((built / f"{REPO}-{app}" / "config.yml").read_text(encoding="utf-8"))

    assert config["claude_code"]["servers"][SERVER]["enabled"] is False
