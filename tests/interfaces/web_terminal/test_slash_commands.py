"""Tests for the project slash-command listing behind the Simple view's suggestions.

Each naming rule pinned here is the one the agent CLI applies when it lists a
project's skills and command files, so a suggestion is always a name the agent
accepts.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import osprey.interfaces.web_terminal.routes.chat as chat_module
from osprey.interfaces.web_terminal.slash_commands import (
    SlashCommand,
    list_chat_slash_commands,
    list_project_slash_commands,
    operator_skill,
    strip_attachment_block,
    unavailable_command,
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


def test_only_a_skill_the_model_may_run_is_agent_invocable(tmp_path: Path) -> None:
    """A skill is the agent's to run too unless it disables model invocation; a command never."""
    claude = tmp_path / ".claude"
    _write(claude / "skills" / "diagnose" / "SKILL.md", "Body\n")
    _write(
        claude / "skills" / "report" / "SKILL.md", "---\ndisable-model-invocation: true\n---\nx\n"
    )
    _write(claude / "commands" / "check.md", "Check a PV\n")
    invocable = {c.name: c.agent_invocable for c in list_chat_slash_commands(tmp_path)}
    assert invocable == {
        "check": False,
        "clear": False,
        "compact": False,
        "diagnose": True,
        "report": False,
    }


class TestOperatorSkill:
    """Which prompts run a project skill by its slash command."""

    @pytest.fixture
    def project(self, tmp_path: Path) -> Path:
        _write(tmp_path / ".claude" / "skills" / "diagnose" / "SKILL.md", "Body\n")
        _write(tmp_path / ".claude" / "commands" / "check.md", "Check a PV\n")
        return tmp_path

    @pytest.mark.parametrize(
        ("prompt", "expected"),
        [
            pytest.param("/diagnose", ("diagnose", ""), id="bare"),
            pytest.param(
                " /diagnose  SR03 orbit\nsince noon ",
                ("diagnose", "SR03 orbit\nsince noon"),
                id="with-args",
            ),
            pytest.param("/check SR:DCCT", None, id="a-command-file"),
            pytest.param("/clear", None, id="a-builtin"),
            pytest.param("/unknown", None, id="no-such-skill"),
            pytest.param("please /diagnose", None, id="not-at-the-start"),
            pytest.param("/diagnose/x", None, id="a-path"),
        ],
    )
    def test_a_prompt_runs_a_skill_only_when_it_opens_with_its_name(
        self, project: Path, prompt: str, expected: tuple[str, str] | None
    ) -> None:
        assert operator_skill(prompt, project) == expected

    @pytest.mark.parametrize(
        "typed",
        [pytest.param("/diagnose SR03", id="with-args"), pytest.param("/diagnose", id="bare")],
    )
    def test_the_attached_files_block_is_not_part_of_the_args(
        self, project: Path, typed: str
    ) -> None:
        """The composer appends the files a message carries; they are not the skill's args."""
        prompt = typed + _BLOCK + "- /data/uploads/c/0123456789abcdef-plot.png (image/png, 12 KB)"
        name, args = operator_skill(prompt, project) or ("", "")
        assert name == "diagnose"
        assert args == typed.removeprefix("/diagnose").strip()


#: The head of the block the composer (``chat-attach.js``) appends to a sent message.
_BLOCK = "\n\nAttached files (read them with the Read tool):\n"


class TestStripAttachmentBlock:
    """The text the operator typed, without the attached-files block the composer added."""

    @pytest.mark.parametrize(
        ("text", "typed"),
        [
            pytest.param(
                "look" + _BLOCK + "- /u/a.png (image/png, 1 KB)\n- /u/b.txt (text/plain, 3 B)",
                "look",
                id="two-files",
            ),
            pytest.param("look at this", "look at this", id="no-block"),
            pytest.param(
                "look" + _BLOCK + "not a file line",
                "look" + _BLOCK + "not a file line",
                id="a-block-that-is-not-one",
            ),
        ],
    )
    def test_only_a_well_formed_block_is_removed(self, text: str, typed: str) -> None:
        assert strip_attachment_block(text) == typed


class TestUnavailableCommand:
    """A prompt that asks for a slash command the chat does not offer is named here.

    The chat offers what ``list_chat_slash_commands`` lists. Any other ``/name``
    the CLI would run itself (``/login``, ``/model``, ``/config``), so the route
    refuses it. Text that only starts with a slash, such as a path, is a prompt.
    """

    @pytest.fixture
    def project(self, tmp_path: Path) -> Path:
        _write(tmp_path / ".claude" / "skills" / "diagnose" / "SKILL.md", "Body\n")
        _write(tmp_path / ".claude" / "commands" / "ops" / "check.md", "Check a PV\n")
        return tmp_path

    @pytest.mark.parametrize(
        ("prompt", "expected"),
        [
            pytest.param("/login", "login", id="login"),
            pytest.param("/logout", "logout", id="logout"),
            pytest.param("/model opus", "model", id="model-with-args"),
            pytest.param(" /config ", "config", id="config-padded"),
            pytest.param("/cost" + _BLOCK + "- /u/a.png (image/png, 1 KB)", "cost", id="files"),
            pytest.param("/clear", None, id="builtin-clear"),
            pytest.param("/compact keep the BPM notes", None, id="builtin-compact"),
            pytest.param("/diagnose SR03", None, id="project-skill"),
            pytest.param("/ops:check SR:DCCT", None, id="project-command"),
            pytest.param("/home/x/y is full", None, id="a-path"),
            pytest.param("/home/kiliev/x", None, id="a-bare-path"),
            pytest.param("/ alone", None, id="a-lone-slash"),
            pytest.param("/10 of the BPMs", None, id="not-a-name"),
            pytest.param("why does /login fail?", None, id="not-at-the-start"),
            pytest.param("hello", None, id="plain-text"),
        ],
    )
    def test_only_an_unlisted_command_name_is_unavailable(
        self, project: Path, prompt: str, expected: str | None
    ) -> None:
        assert unavailable_command(prompt, project) == expected


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

    def test_lists_project_commands_beside_the_builtins(self, tmp_path: Path) -> None:
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
        commands = resp.json()["commands"]
        assert [c["name"] for c in commands] == ["check", "clear", "compact", "diagnose"]
        assert commands[0] == {
            "name": "check",
            "description": "Check a PV",
            "argument_hint": "<pv>",
            "kind": "command",
            "agent_invocable": False,
        }
        assert commands[3] == {
            "name": "diagnose",
            "description": "Investigate failures",
            "argument_hint": "",
            "kind": "skill",
            "agent_invocable": True,
        }
        assert [c["kind"] for c in commands[1:3]] == ["builtin", "builtin"]

    def test_an_empty_project_lists_the_builtins(self, tmp_path: Path) -> None:
        _app, client = self._client(tmp_path)
        resp = client.get("/api/chat/commands")
        assert resp.status_code == 200
        assert [c["name"] for c in resp.json()["commands"]] == ["clear", "compact"]

    def test_post_chat_route_unchanged(self, tmp_path: Path) -> None:
        app, _client = self._client(tmp_path)
        assert app.url_path_for("chat") == "/api/chat"


class TestBuiltinCommands:
    """The CLI's own commands the chat offers beside the project's.

    The route tests above pin the built-ins listed and sorted in with the project's own;
    this one pins the clash between the two.
    """

    def test_a_project_command_named_as_a_builtin_is_listed_once_as_the_projects(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path / ".claude" / "commands" / "clear.md", "Project clear\n")
        found = list_chat_slash_commands(tmp_path)
        assert [c.name for c in found] == ["clear", "compact"]
        assert found[0].kind == "command"
        assert found[0].description == "Project clear"
