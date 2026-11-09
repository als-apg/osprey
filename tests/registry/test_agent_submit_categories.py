"""No submitting agent is told to pass a ``data_type``.

``submit_response`` used to file an agent's answer under a ``data_type``
category named in the agent's own template. Categories are retired: the tool no
longer takes the parameter, so a template that still instructs one tells the
agent to make a call the tool rejects. Templates are scanned as source rather
than rendered: the instruction is static text, and every agent's file is
checked whether or not it is enabled in some profile.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import osprey
from osprey.mcp_server.workspace.tools import submit_response as submit_module

AGENTS_DIR = Path(osprey.__file__).parent / "templates" / "claude_code" / "claude" / "agents"

SUBMIT_TOOL = "mcp__osprey_workspace__submit_response"

#: Matches both spellings of the retired instruction:
#:   - `data_type`: "facility_knowledge"      (bullet, in prose)
#:   data_type="lattice_analysis",            (kwarg, in a worked example)
_DATA_TYPE_RE = re.compile(r'data_type`?\s*[:=]\s*"([^"]+)"')

_TOOLS_LINE_RE = re.compile(r"^tools:.*$", re.MULTILINE)


def _agent_templates() -> list[Path]:
    return sorted(AGENTS_DIR.glob("*.md.j2"))


def _submitting_agents() -> list[Path]:
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


def test_submit_response_takes_no_data_type():
    fn = getattr(submit_module.submit_response, "fn", submit_module.submit_response)
    assert "data_type" not in inspect.signature(fn).parameters


@pytest.mark.parametrize("path", _submitting_agents(), ids=lambda p: p.name)
def test_submitting_agent_names_no_data_type(path: Path):
    declared = _DATA_TYPE_RE.findall(path.read_text(encoding="utf-8"))
    assert not declared, (
        f"{path.name} still instructs data_type={declared!r}; submit_response "
        "no longer takes one, so the call it describes is rejected."
    )
