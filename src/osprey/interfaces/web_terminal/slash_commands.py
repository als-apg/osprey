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
        kind: ``"skill"`` for a project skill, ``"command"`` for a command file.
    """

    name: str
    description: str
    argument_hint: str
    kind: Literal["skill", "command"]


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
      frontmatter sets ``user-invocable: false`` is not listed.
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
