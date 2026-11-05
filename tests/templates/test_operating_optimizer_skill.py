"""The ``operating-optimizer`` skill renders only with the optimizer server, and says what the agent cannot do.

``tests/integration/test_preset_static.py`` resolves ``mcp__`` names in ``*.md``
skills only, not in a ``.md.j2`` like this one, so the check is made here.
"""

from __future__ import annotations

import asyncio
import importlib
import pkgutil
import re

import yaml

from osprey.agent_runner.build_artifacts.catalog import BuildArtifactCatalog
from osprey.cli.templates.manager import TemplateManager

SKILL = "claude_code/claude/skills/operating-optimizer/SKILL.md.j2"


def _render(servers: set[str]) -> str:
    return TemplateManager().jinja_env.get_template(SKILL).render(enabled_servers=servers)


def test_it_is_in_the_catalog():
    artifact = BuildArtifactCatalog.default().get("skills/operating-optimizer")
    assert artifact is not None
    assert artifact.output_path == ".claude/skills/operating-optimizer/SKILL.md"


def test_it_renders_nothing_without_the_optimizer_server():
    assert _render({"controls", "bluesky"}).strip() == ""


def test_it_renders_a_skill_with_the_optimizer_server():
    text = _render({"optimizer"})
    assert text.startswith("---"), "the Jinja guard must not leak into the rendered file"
    front = yaml.safe_load(text.split("---", 2)[1])
    assert front["name"] == "operating-optimizer"
    assert len(text.splitlines()) < 100


def test_it_says_the_agent_cannot_start_resume_or_restore():
    text = re.sub(r"\s+", " ", _render({"optimizer"})).lower()
    assert "cannot start, resume or restore" in text
    assert "nothing is restored" in text


def test_every_tool_it_names_is_an_optimizer_tool():
    from osprey.mcp_server.optimizer import server, tools

    for found in pkgutil.iter_modules(tools.__path__):
        importlib.import_module(f"{tools.__name__}.{found.name}")
    registered = {tool.name for tool in asyncio.run(server.mcp.list_tools())}
    named = set(re.findall(r"mcp__optimizer__(\w+)", _render({"optimizer"})))
    assert named, "the skill names no optimizer tool"
    assert named <= registered
