"""Dropping the app template's graphdb block from a profile (issue #714).

The control-assistant app template injects a graph store into every render: a
Neo4j container, two published host ports, and the ``graph`` MCP server in the
rendered ``.mcp.json``. The documented way out is the profile's ``config:``
overlay — ``_renders_graphdb_block``'s own comment says the bare
``services.graphdb:`` (YAML ``null``) whole-block override removes the block.

These tests build a real minimal profile through ``osprey build`` and pin the
two spellings an operator will try:

* ``services.graphdb:`` (bare null) — the documented removal spelling. It must
  render a project with NO graph store anywhere it would otherwise appear:
  no ``services.graphdb`` block, no ``graphdb`` in ``deployed_services``, no
  compose fragment, no ``graph`` MCP server. It used to crash the build with
  ``'NoneType' object has no attribute 'get'`` in the service-template copy.
* ``services.graphdb: {}`` — an empty whole-block override. It is NOT removal
  (an empty mapping reads as "a store at the defaults" everywhere else) but it
  also cannot deploy, because replacing the block drops the ``path`` the
  compose render needs. It used to die late with "Service 'graphdb' not found
  in configuration"; it must instead be refused at validation with a message
  that points at the working spellings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile import load_profile
from osprey.errors import BuildProfileError

CI_FLAGS = ["--skip-deps", "--skip-lifecycle"]

#: A minimal deploying profile on the app template that injects the store.
PROFILE = """\
name: Graphdb Drop
app_template: control_assistant
provider: anthropic
channel_finder_mode: hierarchical
config:
  control_system.type: mock
{override}"""


def _build_repo(tmp_path: Path, monkeypatch, override: str):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "profile.yml").write_text(PROFILE.format(override=override))
    # The bundle's source zone `osprey init` lays down beside the profile; the
    # Reach Contract refuses a render whose bind source is not there.
    (repo / "data" / "facility_knowledge").mkdir(parents=True)
    monkeypatch.chdir(repo)
    result = CliRunner().invoke(build, CI_FLAGS)
    return repo, result


def _rendered_config(repo: Path) -> dict:
    return yaml.safe_load((repo / "build" / "config.yml").read_text())


class TestBareNullRemovesTheStore:
    """``services.graphdb:`` renders a project with no graph store at all."""

    def test_build_succeeds(self, tmp_path: Path, monkeypatch):
        _, result = _build_repo(tmp_path, monkeypatch, "  services.graphdb:\n")
        assert result.exit_code == 0, result.output

    def test_rendered_config_has_no_graphdb_block(self, tmp_path: Path, monkeypatch):
        repo, result = _build_repo(tmp_path, monkeypatch, "  services.graphdb:\n")
        assert result.exit_code == 0, result.output
        config = _rendered_config(repo)
        assert "graphdb" not in (config.get("services") or {})

    def test_graphdb_leaves_deployed_services(self, tmp_path: Path, monkeypatch):
        repo, result = _build_repo(tmp_path, monkeypatch, "  services.graphdb:\n")
        assert result.exit_code == 0, result.output
        config = _rendered_config(repo)
        assert "graphdb" not in [str(s) for s in config.get("deployed_services") or []]

    def test_no_compose_fragment_is_bundled(self, tmp_path: Path, monkeypatch):
        repo, result = _build_repo(tmp_path, monkeypatch, "  services.graphdb:\n")
        assert result.exit_code == 0, result.output
        assert not (repo / "build" / "services" / "graphdb").exists()

    def test_no_graph_mcp_server_is_rendered(self, tmp_path: Path, monkeypatch):
        repo, result = _build_repo(tmp_path, monkeypatch, "  services.graphdb:\n")
        assert result.exit_code == 0, result.output
        mcp = json.loads((repo / "build" / ".mcp.json").read_text())
        assert "graph" not in mcp.get("mcpServers", {})


class TestEmptyMappingIsRefusedByName:
    """``services.graphdb: {}`` is a named refusal, not a late crash."""

    def test_profile_validation_names_the_working_spellings(self, tmp_path: Path):
        (tmp_path / "profile.yml").write_text(PROFILE.format(override="  services.graphdb: {}\n"))
        with pytest.raises(BuildProfileError) as excinfo:
            load_profile(tmp_path / "profile.yml")
        message = str(excinfo.value)
        assert "services.graphdb: {}" in message
        # The refusal hands the operator both working spellings.
        assert "services.graphdb:" in message
        assert "services.graphdb.<key>:" in message

    def test_build_refuses_before_rendering(self, tmp_path: Path, monkeypatch, caplog):
        with caplog.at_level(logging.ERROR):
            repo, result = _build_repo(tmp_path, monkeypatch, "  services.graphdb: {}\n")
        assert result.exit_code != 0
        reported = result.output + caplog.text + str(result.exception or "")
        assert "services.graphdb" in reported
        # A refusal, not the crashes this spelling used to die with.
        assert "Unexpected error" not in reported
        assert "not found in configuration" not in reported
        assert not (repo / "build").exists()


@pytest.fixture()
def baseline(tmp_path: Path, monkeypatch):
    """A build with no override at all — what the removal is removing."""
    return _build_repo(tmp_path, monkeypatch, "")


class TestBaselineKeepsTheStore:
    """Guard against the removal tests passing vacuously: with no override the
    template's store must actually be there to remove."""

    def test_template_renders_the_store(self, baseline):
        repo, result = baseline
        assert result.exit_code == 0, result.output
        config = _rendered_config(repo)
        assert "graphdb" in config["services"]
        assert "graphdb" in [str(s) for s in config["deployed_services"]]
        mcp = json.loads((repo / "build" / ".mcp.json").read_text())
        assert "graph" in mcp["mcpServers"]
        assert (repo / "build" / "services" / "graphdb").is_dir()
