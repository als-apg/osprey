"""The demo tutorial's channel-finder tools are the ones the demo renders."""

from __future__ import annotations

import re
from pathlib import Path

from osprey.cli.build_profile_presets import _load_preset_raw
from osprey.cli.templates.claude_code import _pv_directory_context, _pv_directory_tools
from osprey.registry.mcp import CHANNEL_FINDER_TOOLS_BY_PIPELINE
from tests.docs.test_facility_graph_pages import tool_counts

DEMO_PAGE = Path(__file__).resolve().parents[2] / "docs" / "source" / "getting-started" / "demo.rst"


def _rendered_tools() -> list[str]:
    """The channel-finder agent's search tools in the demo render."""
    raw, _ = _load_preset_raw("demo")
    ctx = _pv_directory_context(raw["config"])
    assert ctx["pv_directory_configured"], "the demo preset renders no pv-directory tools"
    return list(CHANNEL_FINDER_TOOLS_BY_PIPELINE[raw["channel_finder_mode"]]) + _pv_directory_tools(
        ctx
    )


def _how_it_works(page: str) -> str:
    """The "How it works." paragraph, through the blank line that ends its bullets."""
    start = page.index("**How it works.**")
    in_bullets = False
    lines: list[str] = []
    for line in page[start:].splitlines():
        if line.startswith("- "):
            in_bullets = True
        elif in_bullets and not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


def _bullet_tools(paragraph: str) -> set[str]:
    """Every ``name`` literal in the paragraph's bullets."""
    bullets = paragraph[paragraph.index("\n- ") :]
    return set(re.findall(r"``([a-z_]+)``", bullets))


def _paragraph() -> str:
    return _how_it_works(DEMO_PAGE.read_text(encoding="utf-8"))


def test_the_stated_count_is_the_rendered_channel_finder_tool_count() -> None:
    assert tool_counts(_paragraph()) == [len(_rendered_tools())]


def test_the_bullets_name_exactly_the_rendered_tools() -> None:
    rendered = set(_rendered_tools())
    named = _bullet_tools(_paragraph())
    assert named == rendered, (
        f"missing from the page: {sorted(rendered - named)}; "
        f"on the page but not rendered: {sorted(named - rendered)}"
    )


def test_a_miscounted_or_unnamed_tool_is_caught() -> None:
    paragraph = _paragraph()
    rendered = _rendered_tools()
    assert tool_counts(paragraph.replace("twelve tools", "thirteen tools")) != [len(rendered)]
    assert _bullet_tools(paragraph.replace("``alarm_config``", "the alarm configuration")) != (
        set(rendered)
    )
