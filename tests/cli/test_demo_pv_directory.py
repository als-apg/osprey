"""The demo preset's ``pv_directory`` section: VA-only, kept by a connector switch, turned off by a child.

The preset ships the section without ``type``, so a demo build lists the simulator's
channels and IOC health PVs and replays a packaged alarm feed with no EPICS service
and no Kafka client. ``osprey set connector=epics`` keeps the section and the build
(D7: a section without ``type`` is never a refusal), and a profile extending the
preset turns it off with ``pv_directory: null``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.build_profile_archiver import _expand_dotted
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.cli.init_cmd import init
from osprey.services.pv_directory.resolve import resolve_section
from osprey_connectors.workspace import BUILD_DIR_NAME
from tests._builds import BuiltProject
from tests.fixtures.lifecycle_repo import preserved_environ

SERVER = "pv-directory"
SECTION = {"alarm_state": {"replay_file": "demo:example-alarms"}}


def _init_and_build(parent: Path, name: str, *init_args: str, profile: str | None = None) -> Path:
    """``osprey init <name> --preset demo`` under *parent*, then its build; the render dir.

    *profile*, when given, replaces the materialized ``profile.yml`` before the build.
    """
    repo = parent / name
    with pytest.MonkeyPatch.context() as patch, preserved_environ():
        patch.chdir(parent)
        made = CliRunner().invoke(init, [name, "--preset", "demo", "--no-git", *init_args])
        assert made.exit_code == 0, made.output
        if profile is not None:
            (repo / "profile.yml").write_text(profile, encoding="utf-8")
        result = CliRunner().invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    assert result.exit_code == 0, result.output
    return repo / BUILD_DIR_NAME


def _config(render: Path) -> dict[str, Any]:
    return yaml.safe_load((render / "config.yml").read_text(encoding="utf-8"))


def _servers(render: Path) -> set[str]:
    return set(json.loads((render / ".mcp.json").read_text(encoding="utf-8"))["mcpServers"])


def test_the_preset_states_the_section_as_one_nested_mapping() -> None:
    profile, _profile_dir = resolve_build_profile(None, "demo")

    assert profile.config["pv_directory"] == SECTION
    assert not [key for key in profile.config if key.startswith("pv_directory.")]


def test_the_preset_section_is_va_only() -> None:
    profile, _profile_dir = resolve_build_profile(None, "demo")

    resolved = resolve_section(_expand_dotted(profile.config))

    assert resolved is not None
    assert resolved.type is None
    assert resolved.replay_file == "demo:example-alarms"


def test_the_demo_build_renders_the_section_and_the_server(built_demo: BuiltProject) -> None:
    render = built_demo.build_dir

    assert _config(render)["pv_directory"] == SECTION
    assert SERVER in _servers(render)


def test_connector_epics_still_builds_with_the_section(tmp_path: Path) -> None:
    render = _init_and_build(tmp_path, "demo", "--set", "connector=epics")

    config = _config(render)
    assert config["control_system"]["type"] == "epics"
    assert config["pv_directory"] == SECTION
    assert SERVER in _servers(render)


def test_a_child_turns_the_nested_section_off_with_null(tmp_path: Path) -> None:
    child = (
        "name: Narrowed\n"
        "project_name: demo\n"
        "extends: demo\n"
        "data: data\n"
        "config:\n"
        "  pv_directory: null\n"
    )
    render = _init_and_build(tmp_path, "demo", profile=child)

    assert resolve_section(_config(render)) is None
    assert SERVER not in _servers(render)
