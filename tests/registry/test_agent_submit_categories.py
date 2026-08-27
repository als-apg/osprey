"""Every submitting agent names a real category for its answer.

``submit_response`` files an agent's prose under whatever ``data_type`` the
agent's own definition tells it to pass, and defaults to ``agent_response``
("Uncategorized") when the definition names none. That default is a fallback,
not a category: it says nothing about what the artifact holds, and it becomes a
gallery group heading the operator reads.

So the instruction in each agent template is a contract, and this is its gate.
An agent whose ``tools:`` line carries ``submit_response`` must name a
registered, non-generic category — in the bullet form the searcher agents use
(``- `data_type`: "logbook_research"``) or the keyword form the worked examples
use (``data_type="lattice_analysis"``). Templates are scanned as source rather
than rendered: the instruction is static text, and every agent's file is
checked whether or not it is enabled in some profile.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import osprey
from osprey.stores.type_registry import valid_category_keys

AGENTS_DIR = Path(osprey.__file__).parent / "templates" / "claude_code" / "claude" / "agents"

#: The fallback category. Registered (so an undeclared hand-in still files),
#: but never a legitimate declaration.
GENERIC_CATEGORY = "agent_response"

SUBMIT_TOOL = "mcp__osprey_workspace__submit_response"

#: Matches both spellings of the instruction:
#:   - `data_type`: "facility_knowledge"      (bullet, in prose)
#:   data_type="lattice_analysis",            (kwarg, in a worked example)
_DATA_TYPE_RE = re.compile(r'data_type`?\s*[:=]\s*"([^"]+)"')

_TOOLS_LINE_RE = re.compile(r"^tools:.*$", re.MULTILINE)


def _agent_templates() -> list[Path]:
    """Every agent template. Partial dirs (``_shared``, ``_terminology``) are
    directories, so a top-level glob already excludes them."""
    return sorted(AGENTS_DIR.glob("*.md.j2"))


def _submitting_agents() -> list[Path]:
    """Templates whose ``tools:`` line grants ``submit_response``."""
    out = []
    for path in _agent_templates():
        text = path.read_text(encoding="utf-8")
        tools_line = _TOOLS_LINE_RE.search(text)
        if tools_line and SUBMIT_TOOL in tools_line.group(0):
            out.append(path)
    return out


def test_agent_templates_are_discoverable():
    """Guard the guard: a moved template directory must fail loudly, not
    silently parametrize this module to nothing."""
    assert AGENTS_DIR.is_dir(), f"agent templates not at {AGENTS_DIR}"
    assert _agent_templates(), f"no *.md.j2 agent templates under {AGENTS_DIR}"
    assert _submitting_agents(), "no agent template grants submit_response"


@pytest.mark.parametrize("path", _submitting_agents(), ids=lambda p: p.name)
def test_submitting_agent_names_a_registered_category(path: Path):
    text = path.read_text(encoding="utf-8")
    declared = _DATA_TYPE_RE.findall(text)

    assert declared, (
        f"{path.name} grants {SUBMIT_TOOL} but never tells the agent which "
        "data_type to pass, so its answers file under the generic fallback "
        f"'{GENERIC_CATEGORY}' (shown as 'Uncategorized'). Add a data_type "
        "naming a registered category to its submit instructions."
    )

    valid = valid_category_keys()
    for category in declared:
        assert category in valid, (
            f"{path.name} instructs data_type={category!r}, which is not a "
            f"registered category — submit_response refuses it at runtime. "
            f"Valid: {sorted(valid)}"
        )
        assert category != GENERIC_CATEGORY, (
            f"{path.name} instructs the generic fallback {GENERIC_CATEGORY!r}. "
            "That key exists for agents that declare nothing; an agent that "
            "names its category should name what its answer is about."
        )
