"""Docs pages that enumerate a tool list the code owns.

Each page below spells out, by name, a tool list whose source of truth is a
constant in :mod:`osprey.agent_runner.tool_names`. A reader acts on the page, so
each enumeration is held equal to the constant it describes: a tool added to or
dropped from the code without the page following fails here.
"""

import re
from pathlib import Path

from osprey.agent_runner.tool_names import READ_ONLY_DENIED_BUILTINS, WRITE_CAPABLE_BUILTINS

DOCS_SOURCE = Path(__file__).resolve().parents[2] / "docs" / "source"
CLI_AGENT_PAGE = DOCS_SOURCE / "how-to" / "agent-interfaces" / "cli-agent.rst"
PROFILE_PAGE = DOCS_SOURCE / "reference" / "configuration" / "profile.rst"

#: A backticked built-in tool name: capitalised, letters only.
_TOOL = re.compile(r"``([A-Z][A-Za-z]+)``")


def _bullet_block(text: str, prefix: str) -> str:
    """The bullet starting with ``prefix``, up to the next top-level bold bullet."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(prefix)), None)
    if start is None:
        return ""
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("- **")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _difference_message(found: set[str], expected: set[str]) -> str:
    return (
        f"missing from the page: {sorted(expected - found)}; "
        f"on the page but not in the code: {sorted(found - expected)}"
    )


def test_query_page_lists_the_read_only_floor():
    """The read-only guarantee names exactly the built-ins the query floor denies."""
    block = _bullet_block(CLI_AGENT_PAGE.read_text(encoding="utf-8"), "- **Built-in")
    assert block, f"no '- **Built-in' bullet in {CLI_AGENT_PAGE}"

    found = set(_TOOL.findall(block))
    expected = {name for name in READ_ONLY_DENIED_BUILTINS if not name.startswith("mcp__")}

    assert found, f"the '- **Built-in' bullet in {CLI_AGENT_PAGE} names no tool"
    assert found == expected, _difference_message(found, expected)


def test_profile_page_lists_the_write_capable_builtins():
    """The un-gate warning names exactly the built-ins the build's write check covers."""
    text = PROFILE_PAGE.read_text(encoding="utf-8")
    heading = ".. admonition:: You cannot un-gate a tool that can write"
    start = text.find(heading)
    assert start != -1, f"no {heading!r} in {PROFILE_PAGE}"
    end = text.find("osprey build", start)
    assert end != -1, f"no 'osprey build' after {heading!r} in {PROFILE_PAGE}"
    block = text[start:end]

    found = set(_TOOL.findall(block))
    expected = set(WRITE_CAPABLE_BUILTINS)

    assert found, f"the {heading!r} block in {PROFILE_PAGE} names no tool"
    assert found == expected, _difference_message(found, expected)
