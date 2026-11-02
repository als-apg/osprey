"""A profile-declared MCP server keeps an ``enabled`` switch set over it.

A child profile (a tier) that inherits ``mcp_servers.X`` from its parent
switches the server off with ``claude_code.servers.X.enabled: false`` in its
config overrides. The build applies those overrides first and only then
persists the profile's ``mcp_servers`` into ``claude_code.servers``; the
persisted entry must keep the switch, or the server renders again.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from osprey.cli.build_persistence import _persist_mcp_servers
from osprey.cli.build_profile import McpServerDef
from osprey.registry.mcp import resolve_servers


def _project(tmp_path: Path, config: dict) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return project


def _inherited_servers() -> dict[str, McpServerDef]:
    return {
        "screens": McpServerDef(
            url="http://localhost:10916/mcp",
            port=10916,
            permissions={"allow": ["list_screens"], "ask": ["new_screen"]},
        ),
        "wiki": McpServerDef(url="http://localhost:8001/mcp", port=8001),
    }


def _rendered(project: Path) -> dict[str, dict]:
    config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
    return {s["name"]: s for s in resolve_servers(config["claude_code"], {})}


def test_disabled_override_survives_persistence_and_renders_no_server(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        {"facility_name": "test", "claude_code": {"servers": {"screens": {"enabled": False}}}},
    )

    _persist_mcp_servers(project, _inherited_servers())

    config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
    entry = config["claude_code"]["servers"]["screens"]
    assert entry["enabled"] is False
    # The rest of the rebuilt spec is still the profile's declaration.
    assert entry["url"] == "http://localhost:10916/mcp"
    assert entry["network"]["port"] == 10916

    rendered = _rendered(project)
    assert not rendered.get("screens", {}).get("enabled", False)
    # The sibling the tier did not switch off still renders.
    assert rendered["wiki"]["enabled"] is True


def test_undisabled_inherited_server_still_renders(tmp_path: Path) -> None:
    project = _project(tmp_path, {"facility_name": "test", "claude_code": {"servers": {}}})

    _persist_mcp_servers(project, _inherited_servers())

    config = yaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))
    assert "enabled" not in config["claude_code"]["servers"]["screens"]
    rendered = _rendered(project)
    assert rendered["screens"]["enabled"] is True
    assert rendered["wiki"]["enabled"] is True


def test_explicit_enabled_true_is_carried_too(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        {"facility_name": "test", "claude_code": {"servers": {"screens": {"enabled": True}}}},
    )

    _persist_mcp_servers(project, _inherited_servers())

    rendered = _rendered(project)
    assert rendered["screens"]["enabled"] is True
