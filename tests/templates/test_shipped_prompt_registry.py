"""Shipped prompts name only tools and config keys that exist.

Two guards. The persona templates (``CLAUDE*.md.j2``) forbid and recommend
tools by name, and a name that is not a registered tool is an instruction the
agent cannot follow — ``python_execute`` was the module, the tool is
``mcp__python__execute``. The setup-mode skill lists which config keys take
effect hot and which need a rebuild, and a key with no reader in the config-key
manifest is a knob the operator will turn to no effect.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

import osprey

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(osprey.__file__).resolve().parent
CLAUDE_CODE_TEMPLATES = PACKAGE_ROOT / "templates" / "claude_code"
MCP_SERVER_ROOT = PACKAGE_ROOT / "mcp_server"
CONFIG_KEY_MANIFEST = REPO_ROOT / "scripts" / "config_key_manifest.yml"

#: Tools the agent harness provides on its own, outside any MCP server.
_HARNESS_TOOLS = frozenset(
    {
        "Agent",
        "Bash",
        "Edit",
        "Glob",
        "Grep",
        "MultiEdit",
        "NotebookEdit",
        "Read",
        "Skill",
        "Task",
        "TodoWrite",
        "WebFetch",
        "WebSearch",
        "Write",
    }
)


# ---------------------------------------------------------------------------
# 4. Every tool a persona names is a registered tool
# ---------------------------------------------------------------------------


def registered_tool_names() -> set[str]:
    """Every ``@<server>.tool`` function name under ``osprey.mcp_server``, by AST."""
    names: set[str] = set()
    for path in sorted(MCP_SERVER_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                func = decorator.func if isinstance(decorator, ast.Call) else decorator
                if not (isinstance(func, ast.Attribute) and func.attr == "tool"):
                    continue
                name = node.name
                if isinstance(decorator, ast.Call):
                    for keyword in decorator.keywords:
                        if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                            name = keyword.value.value
                    if decorator.args and isinstance(decorator.args[0], ast.Constant):
                        name = decorator.args[0].value
                names.add(name)
    return names


#: A backticked identifier: bare (``list_systems``, ``Write``) or namespaced
#: (``mcp__python__execute``). A ``*`` tail is a wildcard over a whole server.
_TOOL_MENTION = re.compile(r"`(?:mcp__[a-z_]+__)?([A-Za-z][A-Za-z0-9_]*|\*)`")


def _persona_templates() -> list[Path]:
    return sorted(CLAUDE_CODE_TEMPLATES.glob("CLAUDE*.md.j2"))


def test_every_tool_a_persona_names_is_registered() -> None:
    registry = registered_tool_names()
    assert len(registry) > 50, "tool registry scan found too few tools to be trusted"
    templates = _persona_templates()
    assert templates, "no persona templates found"

    offenders: list[str] = []
    for path in templates:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _TOOL_MENTION.finditer(line):
                name = match.group(1)
                if name == "*" or name in _HARNESS_TOOLS or name in registry:
                    continue
                offenders.append(f"  {path.name}:{lineno}: {match.group(0)}")
    assert offenders == [], "A persona names a tool no MCP server registers:\n" + "\n".join(
        offenders
    )


def test_tool_mention_pattern_still_matches_its_examples() -> None:
    assert _TOOL_MENTION.search("call `list_systems` yourself").group(1) == "list_systems"
    assert _TOOL_MENTION.search("via `mcp__python__execute`").group(1) == "execute"
    assert _TOOL_MENTION.search("all of `mcp__osprey_workspace__*`").group(1) == "*"
    assert _TOOL_MENTION.search("or `Write` yourself").group(1) == "Write"
    # Not identifiers: a command, a key with a colon, a placeholder.
    assert _TOOL_MENTION.search("`osprey scaffold claim`") is None
    assert _TOOL_MENTION.search("`claude_md_template:`") is None
    assert _TOOL_MENTION.search("`[entry_id]`") is None


# ---------------------------------------------------------------------------
# 5. Every Hot/Cold key setup-mode lists has a manifest entry
# ---------------------------------------------------------------------------

_SETUP_MODE_SKILL = CLAUDE_CODE_TEMPLATES / "claude" / "skills" / "setup-mode" / "SKILL.md.j2"
_HOT_COLD_HEADING = "## Hot vs. Cold Changes"

#: A config key: dotted, lower-case, possibly carrying ``<placeholder>``
#: segments or a trailing ``*``. Rejects ``.mcp.json`` (leading dot) and prose.
_CONFIG_KEY = re.compile(r"^[a-z_]+(?:\.(?:[a-z_]+|<[^>]+>|\*))+$")


def hot_cold_keys() -> list[str]:
    """Every key the Hot/Cold table's first column names, one per cell entry."""
    text = _SETUP_MODE_SKILL.read_text(encoding="utf-8")
    assert _HOT_COLD_HEADING in text, f"{_SETUP_MODE_SKILL.name} lost its Hot/Cold section"
    section = text.split(_HOT_COLD_HEADING, 1)[1]
    keys: list[str] = []
    for line in section.splitlines():
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[1] not in {"Hot", "Cold"}:
            continue
        for token in re.findall(r"`([^`]+)`", cells[0]):
            if _CONFIG_KEY.match(token):
                keys.append(token)
    return keys


def _key_pattern(key: str) -> re.Pattern[str]:
    """``a.<type>.b.*`` -> a regex over concrete manifest keys."""
    parts = []
    for segment in key.split("."):
        if segment == "*":
            parts.append(r".+")
        elif segment.startswith("<"):
            parts.append(r"[^.]+")
        else:
            parts.append(re.escape(segment))
    return re.compile(r"\.".join(parts) + r"$")


def manifest_keys() -> set[str]:
    manifest = yaml.safe_load(CONFIG_KEY_MANIFEST.read_text(encoding="utf-8"))
    keys = manifest["keys"]
    assert isinstance(keys, dict) and keys, "manifest has no `keys` section"
    return set(keys)


def test_every_hot_cold_key_setup_mode_lists_has_a_manifest_entry() -> None:
    keys = hot_cold_keys()
    assert len(keys) >= 5, f"Hot/Cold table parse found too few keys: {keys}"
    known = manifest_keys()
    unknown = [key for key in keys if not any(_key_pattern(key).match(k) for k in known)]
    assert unknown == [], (
        "setup-mode lists config keys the manifest has no reader for "
        f"(scripts/config_key_manifest.yml `keys:`): {unknown}"
    )


def test_hot_cold_key_parser_still_finds_the_table() -> None:
    keys = hot_cold_keys()
    assert "control_system.writes_enabled" in keys
    assert "archiver.*" in keys
    assert not any(key.startswith(".") for key in keys)
    assert _key_pattern("control_system.connector.<type>.writes_enabled").match(
        "control_system.connector.epics.writes_enabled"
    )
    assert _key_pattern("archiver.*").match("archiver.type")
    assert not _key_pattern("archiver.*").match("archiver")
