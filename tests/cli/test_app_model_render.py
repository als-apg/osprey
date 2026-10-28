"""An app file may name the model its render runs; the provider stays the host's.

One build of the control-assistant exemplar, with the logbook app naming a
model the host's provider lists and the knowledge app naming one it does not.
The logbook render runs its own model; the knowledge render runs its id as
written and the build warns that the provider does not list it, as it does for
an id ``profile.yml`` names; the channel-finder app, which names none, runs the
host's model like the deployment render does.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey_connectors.workspace import BUILD_DIR_NAME
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo, preserved_environ

#: Listed by the packaged ``anthropic`` entry, and not its default model.
LISTED = "claude-haiku-4-5"
#: Not listed by any packaged provider entry.
UNLISTED = "claude-unlisted-9"


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _name_model(app: Path, model: str) -> None:
    app.write_text(app.read_text(encoding="utf-8") + f"\nmodel: {model}\n", encoding="utf-8")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """The exemplar with two apps naming a model, built once."""
    outside = tmp_path_factory.mktemp("app-model")
    repo = build_exemplar_repo(outside / EXEMPLAR_DIRNAME)
    _name_model(repo / "apps" / "logbook.yml", LISTED)
    _name_model(repo / "apps" / "knowledge.yml", UNLISTED)
    records = _Records()
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(records)
    root.setLevel(logging.DEBUG)
    try:
        with pytest.MonkeyPatch.context() as patch, preserved_environ():
            patch.chdir(outside)
            result = CliRunner().invoke(
                build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"]
            )
    finally:
        root.removeHandler(records)
        root.setLevel(previous_level)
    assert result.exit_code == 0, result.output
    return {"repo": repo, "output": result.output, "log": records.messages}


def _claude_code(repo: Path, render: str = "") -> dict:
    """The rendered ``claude_code`` block; the deployment render is ``build/`` itself."""
    path = repo / BUILD_DIR_NAME / render / "config.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["claude_code"]


def test_an_app_naming_a_listed_model_runs_it(built):
    host = _claude_code(built["repo"])
    logbook = _claude_code(built["repo"], f"{EXEMPLAR_DIRNAME}-logbook")

    assert logbook["default_model"] == LISTED != host.get("default_model")
    assert logbook["provider"] == host["provider"]


def test_an_app_naming_no_model_runs_the_hosts(built):
    host = _claude_code(built["repo"])
    finder = _claude_code(built["repo"], f"{EXEMPLAR_DIRNAME}-channel-finder")

    assert finder.get("default_model") == host.get("default_model")


def test_an_unlisted_app_model_is_used_as_written_and_warned(built):
    knowledge = _claude_code(built["repo"], f"{EXEMPLAR_DIRNAME}-knowledge")

    assert knowledge["default_model"] == UNLISTED
    unlisted = [m.split(";")[0] for m in built["log"] if "does not list" in m]
    assert f"'anthropic' does not list {UNLISTED}" in unlisted, built["log"]
    assert not [m for m in unlisted if LISTED in m]
