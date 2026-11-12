"""Validate rendered Claude Code artifacts after ``osprey build``.

Catches three classes of drift between profile inputs and rendered ``.claude/`` output:

1. **Wildcard tools in agent frontmatter** — every agent must list its MCP tools
   explicitly so the lockdown is auditable. ``mcp__<server>__*`` is rejected.
2. **Unbacked tool declarations** — every ``mcp__`` tool entry in an agent's
   ``tools:`` allowlist must be *backed*: present in the project's
   ``.claude/settings.json`` ``permissions.allow`` **or** ``permissions.ask``.
   Approval-gated tools (e.g. ``mcp__python__execute``) render into
   ``permissions.ask`` by design — they are available to the agent, just
   subject to an approval prompt. An exact-literal ``permissions.deny`` entry
   removes a tool from the backed set (deny wins at runtime). A tool in neither
   allow nor ask — or one explicitly denied — is a real drift: the agent thinks
   it has a tool the MCP gateway will refuse.
3. **Agent model pins that do not apply** — every ``claude_code.agent_models``
   key must name an agent, and an agent the render ships must run the pinned
   model. Without that, a pin on a deployment's own agent file (claimed or
   profile, copied rather than rendered) is ignored while the config says it
   applies.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from osprey.utils.tool_rules import matches_denylist

#: A frontmatter delimiter: a line that is exactly ``---``, trailing spaces or tabs allowed.
_DELIMITER_RE = re.compile(r"---[ \t]*")


def _normalised_text(path: Path) -> str:
    """The file's text with a leading byte-order mark dropped and every line ending ``\\n``."""
    text = path.read_bytes().decode("utf-8").removeprefix("\ufeff")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _frontmatter_lines(md_text: str) -> tuple[list[str], bool] | None:
    """The lines inside the frontmatter block, and whether a delimiter line closes it.

    The block opens on a first line that is a delimiter and closes on the next line
    that is one (the close may end the file), as Claude Code reads it. ``None`` when
    the text opens no block.
    """
    lines = md_text.split("\n")
    if not _DELIMITER_RE.fullmatch(lines[0]):
        return None
    for index, line in enumerate(lines[1:], start=1):
        if _DELIMITER_RE.fullmatch(line):
            return lines[1:index], True
    return lines[1:], False


def _frontmatter_head_length(md_text: str) -> int | None:
    """Where the closing delimiter line starts, or ``None`` when no block is closed.

    ``md_text[:n]`` is the opening line and the block, each line ending ``\\n``.
    """
    block = _frontmatter_lines(md_text)
    if block is None or not block[1]:
        return None
    opening = md_text.split("\n", 1)[0]
    return len(opening) + 1 + sum(len(line) + 1 for line in block[0])


def _parse_frontmatter(md_text: str) -> dict[str, Any] | None:
    block = _frontmatter_lines(md_text)
    if block is None or not block[1]:
        return None
    try:
        data = yaml.safe_load("\n".join(block[0]))
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _loose_delimiter(line: str) -> bool:
    """Whether a looser reader could take *line* for a delimiter.

    True for a line that starts with ``---`` once a byte-order mark is dropped, or that
    is ``---`` once any Unicode whitespace (a no-break space, a form feed) is stripped.
    """
    return line.removeprefix("\ufeff").startswith("---") or line.strip() == "---"


def _unreadable_frontmatter(md_text: str) -> bool:
    """Whether the text opens a frontmatter block a reader could take differently.

    True when a looser reader opens a block on the first line but the delimiter rule
    does not, when no delimiter line closes the block, when a line inside it a looser
    reader would close on (one starting with ``---``, or ``---`` after a Unicode strip)
    hides the keys after it, or when the block is not a YAML mapping. A caller that
    copies the file on refuses it.
    """
    block = _frontmatter_lines(md_text)
    if block is None:
        return _loose_delimiter(md_text.split("\n", 1)[0])
    lines, closed = block
    if not closed or any(_loose_delimiter(line) for line in lines):
        return True
    return _parse_frontmatter(md_text) is None


def _split_tools(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    return [t.strip() for t in str(value).split(",") if t.strip()]


def validate_agent_tools_against_permissions(project_dir: Path) -> list[str]:
    """Return a list of error strings; empty list ⇒ artifacts are coherent.

    Rules:
      - Non-``mcp__`` entries (``Read``, ``Bash``, …) are Claude Code built-ins
        and are ignored.
      - Wildcards in agent ``tools:`` (``mcp__<server>__*``) are rejected —
        list MCP tools explicitly so the lockdown is auditable.
      - Each literal ``mcp__<server>__<tool>`` must appear in
        ``.claude/settings.json`` ``permissions.allow`` **or** ``permissions.ask``
        (the latter covers approval-gated tools, which are backed but prompt on
        use), and must not be matched by any ``permissions.deny`` entry,
        literal or ``*`` namespace glob (deny wins at runtime). A tool that is
        unbacked or explicitly denied is reported.
    """
    project_dir = Path(project_dir)
    settings_path = project_dir / ".claude" / "settings.json"
    agents_dir = project_dir / ".claude" / "agents"

    if not settings_path.exists() or not agents_dir.is_dir():
        return []

    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"{settings_path}: invalid JSON ({e})"]

    permissions = settings.get("permissions", {})
    # A tool is "backed" if it renders into either allow (auto-approved) or ask
    # (approval-gated). Approval-gated tools like mcp__python__execute are still
    # available to the agent — they just prompt on use — so both lists count.
    backed_set: set[str] = {
        str(entry)
        for key in ("allow", "ask")
        for entry in permissions.get(key, [])
        if isinstance(entry, str)
    }
    # deny wins at runtime, including a namespace glob such as ``mcp__plugin_*``,
    # so a tool any deny entry matches is not actually backed.
    deny_entries: list[str] = [
        str(entry) for entry in permissions.get("deny", []) if isinstance(entry, str)
    ]

    errors: list[str] = []
    for md_file in sorted(agents_dir.glob("*.md")):
        text = _normalised_text(md_file)
        if _unreadable_frontmatter(text):
            errors.append(
                f"agent file {md_file.name}: the frontmatter does not parse as a YAML mapping "
                "closed by a line that is exactly ---"
            )
            continue
        fm = _parse_frontmatter(text)
        if not fm:
            continue
        agent_name = str(fm.get("name") or md_file.stem)
        for entry in _split_tools(fm.get("tools")):
            if not entry.startswith("mcp__"):
                continue
            if "*" in entry:
                errors.append(
                    f"agent {agent_name}: tool '{entry}' uses wildcard; "
                    "list MCP tools explicitly so the lockdown is auditable"
                )
                continue
            if entry not in backed_set or matches_denylist(entry, deny_entries):
                errors.append(
                    f"agent {agent_name}: tool '{entry}' not present in "
                    ".claude/settings.json permissions.allow or permissions.ask"
                )

    return errors


def agent_file_models(agents_dir: Path) -> dict[str, str | None]:
    """Map each agent file's stem to the model its frontmatter names.

    Reads a render's ``.claude/agents/`` or a repo's own ``agents/``. A file with
    no ``model:`` line, or no frontmatter, maps to ``None``.

    Args:
        agents_dir: Directory holding the agent ``*.md`` files.

    Returns:
        Agent name to model id, sorted by name; ``{}`` when the directory is missing.
    """
    agents_dir = Path(agents_dir)
    if not agents_dir.is_dir():
        return {}
    models: dict[str, str | None] = {}
    for md_file in sorted(agents_dir.glob("*.md")):
        fm = _parse_frontmatter(_normalised_text(md_file)) or {}
        model = fm.get("model")
        models[md_file.stem] = None if model is None else str(model)
    return models


def agent_model_pin_errors(
    agents_dir: Path, pins: Mapping[str, Any], known_agents: Iterable[str]
) -> list[str]:
    """Return one error per ``claude_code.agent_models`` pin that does not apply.

    Rules, per pin:
      - A name that is neither in *known_agents* nor an agent file in
        *agents_dir* is refused, and the error lists the agent names there are.
      - A known name with no file is not an error: in a render the agent is not
        shipped, and in a repo the framework renders the pin into the file.
      - A file whose ``model:`` line differs from the pin is refused: that file
        decides the agent's model, so the pin would be ignored.

    Args:
        agents_dir: A render's ``.claude/agents/`` or a repo's own ``agents/``.
        pins: Agent name to pinned model id.
        known_agents: Agent names valid without a file in *agents_dir*.

    Returns:
        Error strings, sorted by agent; empty when every pin applies.
    """
    files = agent_file_models(agents_dir)
    names = sorted({*known_agents, *files})
    errors: list[str] = []
    for agent in sorted(pins):
        key = f"claude_code.agent_models.{agent}"
        model_id = str(pins[agent])
        if agent not in names:
            errors.append(f"{key}: no agent is called {agent!r}. Agents: {', '.join(names)}.")
            continue
        if agent not in files:
            continue
        runs = files[agent]
        if runs != model_id:
            errors.append(
                f"{key}: {model_id} is not what agents/{agent}.md runs "
                f"({runs or 'it has no model: line'}). That file is the deployment's own, "
                "so set model: there or remove the pin."
            )
    return errors
