"""The slash commands a project defines, as the agent CLI names them.

A surface that has no terminal can offer the same names the terminal offers by
reading them here. Only ``.claude/skills/`` and ``.claude/commands/`` under the
project are read, never the user's own configuration, matching the
``setting_sources=["project"]`` the chat agent is launched with
(:mod:`osprey.interfaces.web_terminal.operator_session`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

_FRONTMATTER_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """One slash command the chat agent accepts in a project.

    Attributes:
        name: The command as typed after ``/``.
        description: A one-line summary, or ``""``.
        argument_hint: The frontmatter ``argument-hint``, or ``""``.
        kind: ``"skill"`` for a project skill, ``"command"`` for a command file,
            ``"builtin"`` for one of the CLI's own commands (:data:`BUILTIN_COMMANDS`).
        agent_invocable: Whether the agent may also run it on its own: true for a
            skill unless its frontmatter sets ``disable-model-invocation: true``,
            false for a command file and a built-in, which only the operator runs.
    """

    name: str
    description: str
    argument_hint: str
    kind: Literal["skill", "command", "builtin"]
    agent_invocable: bool = False


#: The CLI's own commands the chat offers beside the project's. The chat sends
#: them to the agent like any prompt, and the CLI runs them itself.
BUILTIN_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand(
        name="clear",
        description="Start a new conversation",
        argument_hint="",
        kind="builtin",
    ),
    SlashCommand(
        name="compact",
        description="Summarise the conversation to free up context",
        argument_hint="[instructions]",
        kind="builtin",
    ),
)


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a markdown text into its YAML frontmatter and its body.

    A missing fence yields an empty mapping and the whole text; a fence whose
    content is not a YAML mapping yields an empty mapping and the text after it.
    """
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}, text
    body = text[match.end() :]
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, body
    if not isinstance(data, dict):
        return {}, body
    return data, body


def _read(path: Path) -> str | None:
    """Return the file's text, or None when it cannot be read as UTF-8."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _description(fm: dict[str, Any], body: str) -> str:
    """The first non-empty of ``summary``, ``description``, the first body line."""
    candidates: list[Any] = [fm.get("summary"), fm.get("description")]
    candidates.extend(line for line in body.splitlines() if line.strip())
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    return ""


def _argument_hint(fm: dict[str, Any]) -> str:
    hint = fm.get("argument-hint")
    return hint if isinstance(hint, str) else ""


def _skills(skills_dir: Path) -> list[SlashCommand]:
    if not skills_dir.is_dir():
        return []
    found: list[SlashCommand] = []
    for child in sorted(skills_dir.iterdir()):
        skill_file = child / "SKILL.md"
        if not child.is_dir() or not skill_file.is_file():
            continue
        text = _read(skill_file)
        if text is None:
            continue
        fm, body = _split_frontmatter(text)
        if fm.get("user-invocable") is False:
            continue
        name = fm.get("name")
        if not isinstance(name, str) or not name.strip():
            name = child.name
        found.append(
            SlashCommand(
                name=name,
                description=_description(fm, body),
                argument_hint=_argument_hint(fm),
                kind="skill",
                agent_invocable=fm.get("disable-model-invocation") is not True,
            )
        )
    return found


def _commands(commands_dir: Path) -> list[SlashCommand]:
    if not commands_dir.is_dir():
        return []
    found: list[SlashCommand] = []
    for path in sorted(commands_dir.rglob("*.md")):
        if not path.is_file():
            continue
        rel = path.relative_to(commands_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue
        text = _read(path)
        if text is None:
            continue
        fm, body = _split_frontmatter(text)
        found.append(
            SlashCommand(
                name=":".join(rel.with_suffix("").parts),
                description=_description(fm, body),
                argument_hint=_argument_hint(fm),
                kind="command",
            )
        )
    return found


def list_project_slash_commands(project_dir: Path) -> list[SlashCommand]:
    """List the slash commands the project at *project_dir* defines.

    Naming rules, as the agent CLI applies them:

    - A skill is a direct child directory of ``.claude/skills/`` holding a
      regular file ``SKILL.md``. It is named by its frontmatter ``name`` when
      that is a non-empty string, else by its directory name. A skill whose
      frontmatter sets ``user-invocable: false`` is not listed, and one that
      sets ``disable-model-invocation: true`` is the operator's alone.
    - A command is a ``*.md`` regular file anywhere under ``.claude/commands/``
      whose relative path has no part starting with ``.``. It is named by that
      relative path without ``.md``, its parts joined with ``:``.
    - The description is the first non-empty of the frontmatter ``summary``,
      the frontmatter ``description`` and the first non-empty body line, with
      whitespace collapsed; otherwise ``""``.
    - The argument hint is the frontmatter ``argument-hint`` when it is a
      string; otherwise ``""``.
    - A file that cannot be read as UTF-8 is not listed; frontmatter that is not
      a YAML mapping counts as absent.
    - A skill holds its name against a command of the same name.

    Args:
        project_dir: The directory the chat agent runs in.

    Returns:
        The commands, sorted by name.
    """
    claude_dir = project_dir / ".claude"
    skills = _skills(claude_dir / "skills")
    taken = {s.name for s in skills}
    commands = [c for c in _commands(claude_dir / "commands") if c.name not in taken]
    return sorted(skills + commands, key=lambda c: c.name)


def list_chat_slash_commands(project_dir: Path) -> list[SlashCommand]:
    """List the commands the chat offers in the project: its own and the CLI's built-ins.

    The built-ins (:data:`BUILTIN_COMMANDS`) sit beside the project's commands,
    sorted by name with them. A project command that shares a built-in's name is
    listed once, under the project's entry.

    Args:
        project_dir: The directory the chat agent runs in.

    Returns:
        The commands, sorted by name.
    """
    project = list_project_slash_commands(project_dir)
    taken = {c.name for c in project}
    builtins = [c for c in BUILTIN_COMMANDS if c.name not in taken]
    return sorted(project + builtins, key=lambda c: c.name)


#: A prompt that is one slash command: ``/name``, then any arguments.
_SLASH_PROMPT_RE = re.compile(r"^/(?P<name>[^\s/]+)(?:\s+(?P<args>.*))?$", re.DOTALL)

#: A name the CLI would take for a command: a letter, then letters, digits,
#: ``:``, ``_`` or ``-``. A first token outside this grammar (``/10``,
#: ``/x.txt``) reads as text, not as a command.
_COMMAND_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9:_-]*")

#: The head of the block the composer appends to a message that carries files
#: (``static/js/chat-attach.js``), and one file line of that block.
_ATTACHMENT_HEAD = "\n\nAttached files (read them with the Read tool):\n"
_ATTACHMENT_LINE_RE = re.compile(r"^- (.+) \(([^()]*), ([^()]*)\)$")


def strip_attachment_block(text: str) -> str:
    """Return *text* without the attached-files block the composer appends to it.

    The block is the last ``Attached files`` head and the file lines after it,
    as ``splitAttachmentBlock`` in ``chat-attach.js`` reads it. Text with no
    such block, or with a line after the head that is not a file line, is
    returned unchanged.
    """
    at = text.rfind(_ATTACHMENT_HEAD)
    if at == -1:
        return text
    lines = text[at + len(_ATTACHMENT_HEAD) :].split("\n")
    if not all(_ATTACHMENT_LINE_RE.match(line) for line in lines):
        return text
    return text[:at]


def operator_skill(prompt: str, project_dir: Path) -> tuple[str, str] | None:
    """The project skill an operator's prompt runs by its slash command, and its arguments.

    The CLI expands such a prompt into the skill without a tool call, so nothing
    in the agent's stream says a skill ran; the chat draws it from the prompt.
    The attached-files block the composer appends is not part of the arguments.

    Returns:
        ``(name, args)`` when *prompt* is ``/<skill>`` (``args`` is ``""`` when
        none follow), ``None`` for any other prompt, a command file or a built-in.
    """
    match = _SLASH_PROMPT_RE.match(strip_attachment_block(prompt).strip())
    if match is None:
        return None
    name = match["name"]
    if not any(
        c.kind == "skill" and c.name == name for c in list_project_slash_commands(project_dir)
    ):
        return None
    return name, (match["args"] or "").strip()


def unavailable_command(prompt: str, project_dir: Path) -> str | None:
    """The slash command *prompt* asks for when the chat does not offer it.

    The chat offers what :func:`list_chat_slash_commands` lists. Any other
    command (``/login``, ``/model``, ``/config``) the CLI would run itself, so
    the chat route refuses the prompt. A prompt whose first token is not a
    command name is text: a path such as ``/home/x is full`` (a ``/`` inside
    the token), or ``/10 of them`` (not a name), goes to the agent.

    Returns:
        The name typed after ``/`` when *prompt* opens with an unlisted
        command name, else ``None``.
    """
    match = _SLASH_PROMPT_RE.match(strip_attachment_block(prompt).strip())
    if match is None:
        return None
    name = match["name"]
    if _COMMAND_NAME_RE.fullmatch(name) is None:
        return None
    if any(c.name == name for c in list_chat_slash_commands(project_dir)):
        return None
    return name


def operator_skill_frames(name: str, args: str, call_id: str) -> list[dict[str, Any]]:
    """The chat frames for a skill the operator ran: one finished ``Skill`` call.

    They are the frames a skill the agent chose would carry, with ``by:
    "operator"`` in the input, so the chat draws both as one kind of row. The
    live stream (``operator_session``) and a transcript's replay
    (``routes/session``) both build them here.
    """
    tool_input: dict[str, Any] = {"skill": name, "by": "operator"}
    if args:
        tool_input["args"] = args
    return [
        {
            "type": "tool_use",
            "tool_name": "Skill",
            "tool_name_raw": "Skill",
            "tool_use_id": call_id,
            "input": tool_input,
            "parent_tool_use_id": None,
        },
        {
            "type": "tool_result",
            "tool_use_id": call_id,
            "is_error": False,
            "preview": f"Launching skill: {name}",
            "truncated": False,
            "refusal": None,
            "parent_tool_use_id": None,
        },
    ]
