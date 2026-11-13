"""``osprey build`` installs an application's subagents into each render that shows it."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.apps.agents import parse_agent
from osprey.apps.manifest import parse_manifest
from osprey.apps.rules import AppSelection, LoginSelection
from osprey.cli.build_cmd import _copy_app_agents
from osprey.deployment.web_terminals.render_naming import render_project
from osprey_connectors.errors import BuildProfileError
from tests.cli._lifecycle_build import invoke_build
from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

KNOB: dict[str, Any] = {
    "name": "knob",
    "containers": {
        "knob": {
            "runtime": "osprey",
            "command": ["python", "-m", "knob_app.server"],
            "health": "/health",
            "drives_target": True,
        },
    },
    "tools": {
        "module": "knob_app.tools",
        "read": ["knob_read"],
        "approval": [{"tool": "knob_write", "prompt": "Write the knob"}],
    },
    "agents": ["knob-helper"],
}

AGENT_MD = (
    "---\n"
    "name: knob-helper\n"
    "description: Reads the knob and explains its value.\n"
    "tools:\n"
    "  - mcp__knob__knob_read\n"
    "delegates:\n"
    "  - knob_read\n"
    "delegation: >-\n"
    "  For questions about the knob, delegate to knob-helper; do not call\n"
    "  {delegated_tools} yourself.\n"
    "---\n"
    "\n"
    "You read the knob for {{ facility_name }} and explain it.\n"
)

NOTE = (
    "- **knob-helper** — For questions about the knob, delegate to knob-helper; "
    "do not call `knob_read` yourself."
)

PINNED = "claude-haiku-4-5"


def _knob_repo(root: Path, *config_lines: str) -> Path:
    repo = build_exemplar_repo(root / EXEMPLAR_DIRNAME)
    app_dir = repo / "apps" / "knob"
    (app_dir / "agents").mkdir(parents=True)
    (app_dir / "app.yml").write_text(yaml.safe_dump(KNOB, sort_keys=False), encoding="utf-8")
    (app_dir / "agents" / "knob-helper.md").write_text(AGENT_MD, encoding="utf-8")
    with (repo / "profile.yml").open("a", encoding="utf-8") as profile:
        profile.write("\n" + yaml.safe_dump({"apps": {"knob": {}}}, sort_keys=False))
    if config_lines:
        profile_path = repo / "profile.yml"
        text = profile_path.read_text(encoding="utf-8")
        inserted = "".join(f"  {line}\n" for line in config_lines)
        profile_path.write_text(
            text.replace("\nconfig:\n", "\nconfig:\n" + inserted, 1), encoding="utf-8"
        )
    (repo / "tiers" / "quiet.yml").write_text(
        "name: Quiet\nconfig:\n  control_system.writes: asked\napps:\n  knob:\n    shown: false\n",
        encoding="utf-8",
    )
    (repo / "tiers" / "asker.yml").write_text(
        "name: Asker\nconfig:\n  control_system.writes: asked\n", encoding="utf-8"
    )
    result = invoke_build(repo)
    assert result.exit_code == 0, result.output
    return repo


@pytest.fixture(scope="module")
def agents_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _knob_repo(tmp_path_factory.mktemp("agents"))


@pytest.fixture(scope="module")
def pinned_repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _knob_repo(
        tmp_path_factory.mktemp("pinned"), f"claude_code.agent_models.knob-helper: {PINNED}"
    )


#: The renders that show the application: the root login, and a login file.
SHOWING = [None, "asker"]


def _render(repo: Path, login: str | None) -> Path:
    if login is None:
        return repo / "build"
    project, _ = render_project(EXEMPLAR_DIRNAME, login)
    return repo / "build" / project


def _model_line(agent_file: Path) -> str:
    [line] = [
        line
        for line in agent_file.read_text(encoding="utf-8").splitlines()
        if line.startswith("model:")
    ]
    return line


def _allowed(render: Path) -> list[str]:
    settings = json.loads((render / ".claude" / "settings.json").read_text(encoding="utf-8"))
    return settings["permissions"]["allow"]


@pytest.mark.parametrize("login", SHOWING)
def test_a_showing_login_carries_the_agent_file(agents_repo: Path, login: str | None) -> None:
    agent_file = _render(agents_repo, login) / ".claude" / "agents" / "knob-helper.md"
    text = agent_file.read_text(encoding="utf-8")

    assert _model_line(agent_file).startswith("model: ")
    assert text.replace(_model_line(agent_file) + "\n", "") == AGENT_MD


@pytest.mark.parametrize("login", SHOWING)
def test_the_brief_carries_the_agents_note(agents_repo: Path, login: str | None) -> None:
    brief = (_render(agents_repo, login) / "CLAUDE.md").read_text(encoding="utf-8")
    assert NOTE in brief


@pytest.mark.parametrize("login", SHOWING)
def test_the_agent_may_be_delegated_to(agents_repo: Path, login: str | None) -> None:
    assert "Task(knob-helper)" in _allowed(_render(agents_repo, login))


def test_a_login_that_hides_the_application_carries_none_of_its_agents(agents_repo: Path) -> None:
    render = _render(agents_repo, "quiet")
    assert not (render / ".claude" / "agents" / "knob-helper.md").exists()
    assert "knob-helper" not in (render / "CLAUDE.md").read_text(encoding="utf-8")
    assert "Task(knob-helper)" not in _allowed(render)


def test_the_agent_runs_the_main_model_unless_pinned(agents_repo: Path, pinned_repo: Path) -> None:
    unpinned = _model_line(_render(agents_repo, None) / ".claude" / "agents" / "knob-helper.md")
    pinned = _model_line(_render(pinned_repo, None) / ".claude" / "agents" / "knob-helper.md")

    assert unpinned != f"model: {PINNED}"
    assert pinned == f"model: {PINNED}"


def test_an_agent_name_the_render_already_has_is_refused(tmp_path: Path) -> None:
    manifest = parse_manifest(KNOB, app="knob")
    (tmp_path / "apps" / "knob" / "agents").mkdir(parents=True)
    (tmp_path / "apps" / "knob" / "agents" / "knob-helper.md").write_text(AGENT_MD, "utf-8")
    agent = parse_agent(AGENT_MD, app="knob", name="knob-helper")
    render = tmp_path / "build"
    (render / ".claude" / "agents").mkdir(parents=True)
    (render / ".claude" / "agents" / "knob-helper.md").write_text("profile's own\n", "utf-8")
    selection = LoginSelection(
        audience="person", apps={"knob": AppSelection(shown=True)}, tier="asked", shared_cards=()
    )

    with pytest.raises(BuildProfileError, match="already has an agent by that name"):
        _copy_app_agents(tmp_path, render, [manifest], {"knob": (agent,)}, selection)


def test_a_profile_agent_of_the_same_name_refuses_the_build(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    repo = _knob_repo(tmp_path)
    (repo / "agents").mkdir(exist_ok=True)
    (repo / "agents" / "knob-helper.md").write_text(
        "---\nname: knob-helper\ndescription: The profile's own.\n---\n", encoding="utf-8"
    )
    profile = repo / "profile.yml"
    text = profile.read_text(encoding="utf-8")
    assert text.count("\nagents:\n") == 1
    profile.write_text(
        text.replace("\nagents:\n", "\nagents:\n  - agents/knob-helper\n"), encoding="utf-8"
    )

    with caplog.at_level(logging.ERROR):
        result = invoke_build(repo)

    assert result.exit_code != 0
    assert "this render already has an agent by that name" in caplog.text


class _Spec:
    def agent_model(self, _name: str) -> str:
        return PINNED


@pytest.mark.parametrize(
    "raw",
    [AGENT_MD.encode("utf-8"), b"\xef\xbb\xbf" + AGENT_MD.replace("\n", "\r\n").encode("utf-8")],
    ids=["plain", "bom-and-crlf"],
)
def test_the_model_line_goes_inside_the_frontmatter_the_loader_read(
    tmp_path: Path, raw: bytes
) -> None:
    from osprey.cli.templates.claude_code import stamp_app_agent_models

    agent_file = tmp_path / ".claude" / "agents" / "knob-helper.md"
    agent_file.parent.mkdir(parents=True)
    agent_file.write_bytes(raw + b"\n---\n\nmodel: body text\n")
    ctx = {"agents": [{"name": "knob-helper", "app": "knob"}], "claude_code_model_spec": _Spec()}

    stamp_app_agent_models(tmp_path, ctx)

    text = agent_file.read_text(encoding="utf-8")
    assert text.startswith(f"---\nname: knob-helper\nmodel: {PINNED}\n")
    assert text.endswith("\n---\n\nmodel: body text\n")
    assert parse_agent(text.replace(f"model: {PINNED}\n", ""), app="knob", name="knob-helper")
