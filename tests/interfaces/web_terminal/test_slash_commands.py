"""Tests for the project slash-command listing behind the Simple view's suggestions.

Each naming rule pinned here is the one the agent CLI applies when it lists a
project's skills and command files, so a suggestion is always a name the agent
accepts.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import osprey.interfaces.web_terminal.routes.chat as chat_module
from osprey.interfaces.web_terminal.slash_commands import (
    SlashCommand,
    list_project_slash_commands,
)


def _write(path: Path, text: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")


def _names(project: Path) -> list[str]:
    return [c.name for c in list_project_slash_commands(project)]


def _by_name(project: Path) -> dict[str, SlashCommand]:
    return {c.name: c for c in list_project_slash_commands(project)}


def test_no_claude_dir_lists_nothing(tmp_path: Path) -> None:
    """No ``.claude/``, or one without skills/commands, lists nothing."""
    assert list_project_slash_commands(tmp_path) == []
    (tmp_path / ".claude").mkdir()
    assert list_project_slash_commands(tmp_path) == []


def test_skill_named_by_frontmatter_then_directory(tmp_path: Path) -> None:
    """A skill is named by its frontmatter ``name``, else its directory."""
    skills = tmp_path / ".claude" / "skills"
    _write(skills / "dirname-x" / "SKILL.md", "---\nname: other-name\n---\nBody\n")
    _write(skills / "nofm" / "SKILL.md", "Just a body\n")
    (skills / "emptydir").mkdir(parents=True)
    assert _names(tmp_path) == ["nofm", "other-name"]


def test_skill_not_user_invocable_is_hidden(tmp_path: Path) -> None:
    """``user-invocable: false`` hides a skill; ``disable-model-invocation`` does not."""
    skills = tmp_path / ".claude" / "skills"
    _write(skills / "hidden" / "SKILL.md", "---\nuser-invocable: false\n---\nx\n")
    _write(skills / "manual" / "SKILL.md", "---\ndisable-model-invocation: true\n---\nx\n")
    assert _names(tmp_path) == ["manual"]


def test_command_named_by_relative_path(tmp_path: Path) -> None:
    """A command file is named by its path under ``commands/``, parts joined by ``:``."""
    commands = tmp_path / ".claude" / "commands"
    _write(commands / "plain.md", "Plain\n")
    _write(commands / "ops" / "nested.md", "Nested\n")
    _write(commands / ".hidden.md", "Hidden\n")
    _write(commands / ".private" / "inner.md", "Inner\n")
    _write(commands / "notes.txt", "Not markdown\n")
    assert _names(tmp_path) == ["ops:nested", "plain"]


def test_description_prefers_summary_then_description_then_first_line(tmp_path: Path) -> None:
    """Description ranks ``summary``, ``description``, then the first body line."""
    commands = tmp_path / ".claude" / "commands"
    _write(commands / "a.md", "---\nsummary: Short line\ndescription: Long line\n---\nBody\n")
    _write(
        commands / "b.md",
        "---\ndescription: >\n  Folded   text\n  over lines\n---\nBody\n",
    )
    _write(commands / "c.md", "\n\n  First body line  \nSecond\n")
    _write(commands / "d.md", "---\nargument-hint: x\n---\n\n")
    found = _by_name(tmp_path)
    assert found["a"].description == "Short line"
    assert found["b"].description == "Folded text over lines"
    assert found["c"].description == "First body line"
    assert found["d"].description == ""


def test_argument_hint_from_frontmatter(tmp_path: Path) -> None:
    """``argument-hint`` is carried through; absent it is empty."""
    commands = tmp_path / ".claude" / "commands"
    _write(commands / "with.md", "---\nargument-hint: <pv>\n---\nBody\n")
    _write(commands / "without.md", "Body\n")
    found = _by_name(tmp_path)
    assert found["with"].argument_hint == "<pv>"
    assert found["without"].argument_hint == ""


def test_skill_shadows_command_of_same_name(tmp_path: Path) -> None:
    """A skill and a command of the same name list once, as the skill."""
    _write(tmp_path / ".claude" / "skills" / "fruit" / "SKILL.md", "Skill fruit\n")
    _write(tmp_path / ".claude" / "commands" / "fruit.md", "Command fruit\n")
    result = list_project_slash_commands(tmp_path)
    assert len(result) == 1
    assert result[0].kind == "skill"
    assert result[0].description == "Skill fruit"


def test_sorted_by_name_and_shaped(tmp_path: Path) -> None:
    """The result is sorted by name and each entry is a ``SlashCommand``."""
    _write(tmp_path / ".claude" / "skills" / "zeta" / "SKILL.md", "Z\n")
    _write(tmp_path / ".claude" / "commands" / "alpha.md", "A\n")
    _write(tmp_path / ".claude" / "skills" / "mid" / "SKILL.md", "M\n")
    result = list_project_slash_commands(tmp_path)
    assert [c.name for c in result] == ["alpha", "mid", "zeta"]
    assert all(isinstance(c, SlashCommand) for c in result)
    assert result[0] == SlashCommand(
        name="alpha", description="A", argument_hint="", kind="command"
    )
    assert result[1].kind == "skill"


def test_unreadable_entry_is_skipped(tmp_path: Path) -> None:
    """Bad UTF-8 is skipped; bad YAML lists under its directory name."""
    skills = tmp_path / ".claude" / "skills"
    _write(skills / "binary" / "SKILL.md", b"\xff\xfe\x00bad")
    _write(skills / "badyaml" / "SKILL.md", "---\nname: [unclosed\n---\nAfter fence\n")
    found = _by_name(tmp_path)
    assert list(found) == ["badyaml"]
    assert found["badyaml"].description == "After fence"


class TestCommandsRoute:
    """``GET /api/chat/commands`` answers the project's slash commands."""

    @staticmethod
    def _client(project: Path) -> tuple[FastAPI, TestClient]:
        app = FastAPI()
        app.state.project_cwd = str(project)
        app.include_router(chat_module.router)
        return app, TestClient(app)

    def test_lists_project_commands(self, tmp_path: Path) -> None:
        _write(
            tmp_path / ".claude" / "skills" / "diagnose" / "SKILL.md",
            "---\nsummary: Investigate failures\n---\nBody\n",
        )
        _write(
            tmp_path / ".claude" / "commands" / "check.md",
            "---\nargument-hint: <pv>\n---\nCheck a PV\n",
        )
        _app, client = self._client(tmp_path)
        resp = client.get("/api/chat/commands")
        assert resp.status_code == 200
        assert resp.json() == {
            "commands": [
                {
                    "name": "check",
                    "description": "Check a PV",
                    "argument_hint": "<pv>",
                    "kind": "command",
                },
                {
                    "name": "diagnose",
                    "description": "Investigate failures",
                    "argument_hint": "",
                    "kind": "skill",
                },
            ]
        }

    def test_empty_project_answers_empty_list(self, tmp_path: Path) -> None:
        _app, client = self._client(tmp_path)
        resp = client.get("/api/chat/commands")
        assert resp.status_code == 200
        assert resp.json() == {"commands": []}

    def test_post_chat_route_unchanged(self, tmp_path: Path) -> None:
        app, _client = self._client(tmp_path)
        assert app.url_path_for("chat") == "/api/chat"
