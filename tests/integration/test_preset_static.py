"""Tier 0 static checks — preset YAML, skill ↔ MCP-tool consistency.

Fast (<5s), no model calls, no subprocess. Run on every push.

Catches drift between:
  • preset YAML and the artifact registry (hooks/rules/skills/agents/output_styles)
  • preset ``web_panels`` and the shared ``BUILTIN_PANELS`` registry
  • SKILL.md ``mcp__server__tool`` references and the actual ``@mcp.tool()``
    function names exposed by the framework MCP servers

MCP-server short names are matched at .mcp.json render time (Tier 1 covers
that).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from osprey.cli.build_profile import load_profile
from osprey.cli.templates.artifact_library import validate_artifacts

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_OSPREY = REPO_ROOT / "src" / "osprey"
PRESETS_DIR = SRC_OSPREY / "profiles" / "presets"
MCP_SERVER_ROOT = SRC_OSPREY / "mcp_server"
SKILL_ROOTS = (
    SRC_OSPREY / "templates" / "skills",
    SRC_OSPREY / "templates" / "claude_code" / "claude" / "skills",
)
COMMANDS_ROOT = SRC_OSPREY / "templates" / "claude_code" / "claude" / "commands"

_MCP_REF_RE = re.compile(r"mcp__([a-zA-Z0-9_-]+)__([a-zA-Z0-9_]+)")


def _all_presets() -> list[Path]:
    """Return every preset YAML shipped with osprey."""
    return sorted(p for p in PRESETS_DIR.glob("*.yml") if not p.name.startswith("_"))


def _collect_framework_tool_names() -> dict[str, set[str]]:
    """Walk ``src/osprey/mcp_server/<server>/tools/`` and collect ``@mcp.tool()`` names.

    Returns a mapping of ``server_name → set of tool function names``. The
    server name is the package directory name; tool names are the function
    names decorated with ``@mcp.tool()``.

    This intentionally excludes ``mcp_server/<server>/server.py`` registration
    code — only ``@mcp.tool()``-decorated functions count as tools.
    """
    server_tools: dict[str, set[str]] = {}
    for server_dir in MCP_SERVER_ROOT.iterdir():
        if not server_dir.is_dir() or server_dir.name.startswith(("_", ".")):
            continue
        tools_dir = server_dir / "tools"
        if not tools_dir.is_dir():
            continue
        names: set[str] = set()
        for py_file in tools_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue
            names.update(_extract_mcp_tool_names(py_file))
        if names:
            server_tools[server_dir.name] = names
    return server_tools


def _extract_mcp_tool_names(py_file: Path) -> set[str]:
    """Parse a Python file and return the set of functions decorated with ``@mcp.tool()``."""
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for deco in node.decorator_list:
            target = deco.func if isinstance(deco, ast.Call) else deco
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "tool"
                and isinstance(target.value, ast.Name)
                and target.value.id in {"mcp", "server"}
            ):
                out.add(node.name)
    return out


def _server_alias_map(server_tools: dict[str, set[str]]) -> dict[str, set[str]]:
    """Map every plausible server-name alias to its tool set.

    The .mcp.json server name (e.g. ``controls``, ``python``, ``channel-finder``)
    differs from the ``mcp_server/`` package directory (e.g. ``control_system``,
    ``python_executor``, ``channel_finder_hierarchical``). Skill files use the
    .mcp.json form (``mcp__controls__channel_read``), so we expose both.
    """
    aliases: dict[str, set[str]] = {}
    for server, tools in server_tools.items():
        aliases.setdefault(server, set()).update(tools)
        # Hyphen variant (channel_finder_hierarchical → channel-finder-hierarchical)
        aliases.setdefault(server.replace("_", "-"), set()).update(tools)
    # Hard-coded aliases from registry/mcp.py FRAMEWORK_SERVERS naming
    if "control_system" in server_tools:
        aliases.setdefault("controls", set()).update(server_tools["control_system"])
    if "python_executor" in server_tools:
        aliases.setdefault("python", set()).update(server_tools["python_executor"])
    # channel-finder is a single .mcp.json server name backed by one of three packages
    cf_union: set[str] = set()
    for pkg in (
        "channel_finder_hierarchical",
        "channel_finder_in_context",
        "channel_finder_middle_layer",
    ):
        if pkg in server_tools:
            cf_union.update(server_tools[pkg])
    if cf_union:
        aliases.setdefault("channel-finder", set()).update(cf_union)
        aliases.setdefault("channel_finder", set()).update(cf_union)
    return aliases


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_path", _all_presets(), ids=lambda p: p.stem)
def test_preset_yaml_parses_and_artifacts_resolve(preset_path: Path) -> None:
    """Each preset must parse, validate, and reference only artifacts that exist."""
    profile = load_profile(preset_path)

    artifacts = {}
    for artifact_type in ("hooks", "rules", "skills", "agents", "output_styles"):
        names = getattr(profile, artifact_type, [])
        if names:
            artifacts[artifact_type] = list(names)
    if artifacts:
        validate_artifacts(artifacts)


@pytest.mark.parametrize("preset_path", _all_presets(), ids=lambda p: p.stem)
def test_preset_web_panels_against_registry(preset_path: Path) -> None:
    """Every ``web_panels`` entry in a preset must resolve to a built-in panel.

    The runtime registry lives in ``osprey.profiles.web_panels`` and is shared
    with both the web terminal and the template-manifest validator. A drifted
    preset entry would render a broken UI tab at runtime.
    """
    from osprey.profiles.web_panels import BUILTIN_PANELS

    profile = load_profile(preset_path)
    unknown = [p for p in profile.web_panels if p not in BUILTIN_PANELS]
    assert not unknown, (
        f"{preset_path.name} declares unknown web_panels: {unknown} "
        f"(valid: {sorted(BUILTIN_PANELS)})"
    )


def _iter_skill_files() -> list[Path]:
    files: list[Path] = []
    for root in SKILL_ROOTS:
        if root.is_dir():
            files.extend(root.rglob("SKILL.md"))
            files.extend(root.rglob("*.md"))
    # Dedup
    return sorted(set(files))


def test_skill_referenced_mcp_tools_exist() -> None:
    """Every ``mcp__<server>__<tool>`` ref in a SKILL must resolve to a real tool.

    Skipped for servers not built into this repo (facility-specific or
    documentation-only references).
    """
    server_tools = _collect_framework_tool_names()
    aliases = _server_alias_map(server_tools)

    orphans: list[str] = []
    for skill_file in _iter_skill_files():
        text = skill_file.read_text(encoding="utf-8", errors="replace")
        for match in _MCP_REF_RE.finditer(text):
            server, tool = match.group(1), match.group(2)
            known_tools = aliases.get(server)
            if known_tools is None:
                continue  # unknown server (facility-specific or doc example)
            if tool not in known_tools:
                rel = skill_file.relative_to(REPO_ROOT)
                orphans.append(f"{rel}: mcp__{server}__{tool} not found")

    assert not orphans, "Skill files reference MCP tools that do not exist:\n  " + "\n  ".join(
        orphans
    )


def test_slash_commands_resolve_targets() -> None:
    """Every ``.claude/commands/*.md`` slash command must reference a real artifact.

    Skipped silently when no commands directory ships with this repo.
    """
    if not COMMANDS_ROOT.is_dir():
        pytest.skip("no slash-command templates in this repo")

    from osprey.cli.templates.artifact_library import list_artifacts

    available_skills = set(list_artifacts("skills"))
    available_agents = set(list_artifacts("agents"))

    orphans: list[str] = []
    for cmd_file in COMMANDS_ROOT.rglob("*.md*"):
        text = cmd_file.read_text(encoding="utf-8", errors="replace")
        # Match common references like @skill:NAME, /agent:NAME, "skill: NAME" frontmatter
        for kind, pat in (
            ("skill", r"@skill:([a-z][a-z0-9_-]+)"),
            ("agent", r"@agent:([a-z][a-z0-9_-]+)"),
        ):
            for m in re.finditer(pat, text):
                ref = m.group(1)
                pool = available_skills if kind == "skill" else available_agents
                if ref not in pool:
                    rel = cmd_file.relative_to(REPO_ROOT)
                    orphans.append(f"{rel}: {kind} '{ref}' not in registry")

    assert not orphans, "Slash commands reference unknown artifacts:\n  " + "\n  ".join(orphans)
