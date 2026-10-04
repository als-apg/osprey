"""Agentic e2e: the graph answers a sector question by place path and position.

A deployment built from the ``control_assistant`` preset, its store seeded with
the graph view the build writes, and an operator asking which beam position
monitors sit in one section and in what order the beam meets them. The graph
holds the answer as facts — each device's ``placePath``, ``sectionCode`` and
``sPositionM`` — so the agent has only to read them: under ``SR/`` in section
``SECT1`` the monitors are BPM01 to BPM06, at the positions of
:data:`SECTOR_BPMS`.

The deployment, the store and the seeding are the graph smoke's own fixtures
(:mod:`tests.e2e.test_graph_mcp_smoke`), so both questions are asked of a
corpus that reached the store by the same path.

The grade is deterministic: the answer names all six monitors, in position
order. :func:`named_in_order` is that grade, and the offline ``-k floor`` tests
run it against hand-written prose with no Docker, no API key and no agent
session::

    uv run pytest tests/e2e/test_sector_bpms_agentic.py -k floor
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest

from osprey.agent_runner import expected_mcp_servers
from tests.e2e.sdk_helpers import HAS_SDK, is_claude_code_available, render_dir, run_sdk_query
from tests.e2e.test_graph_mcp_smoke import (  # noqa: F401 - fixtures used by name
    graph_project,
    graph_smoke_plugin_dir,
    graph_store_port,
)

logger = logging.getLogger(__name__)

#: The monitors under ``SR/`` whose section code is ``SECT1``, ordered by
#: ``sPositionM`` (metres), as the control-assistant build's graph view holds them.
SECTOR_BPMS: tuple[tuple[str, float], ...] = (
    ("BPM01", 4.800),
    ("BPM02", 6.287),
    ("BPM03", 8.800),
    ("BPM04", 11.177),
    ("BPM05", 13.690),
    ("BPM06", 14.499),
)

#: Operator-style: it names a section and a kind of device, and nothing the
#: agent could pattern-match onto a tool.
OPERATOR_PROMPT = (
    "Which beam position monitors sit in section SECT1 under SR, in the order the "
    "beam meets them, and where along the beam path is each one?"
)

_BPM_NAME = re.compile(r"\bBPM\s*0*(\d+)\b", re.IGNORECASE)


def named_in_order(text: str) -> list[str]:
    """The monitors *text* names, in the order of each one's first mention.

    ``BPM 1``, ``bpm01`` and ``SR/BPM01`` all name ``BPM01``.

    Args:
        text: The agent's prose.

    Returns:
        Each monitor's canonical name, once, in first-mention order.
    """
    seen: list[str] = []
    for match in _BPM_NAME.finditer(text):
        name = f"BPM{int(match.group(1)):02d}"
        if name not in seen:
            seen.append(name)
    return seen


def _expected() -> list[str]:
    return [name for name, _ in SECTOR_BPMS]


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.agentic_benchmark
@pytest.mark.requires_als_apg
@pytest.mark.skipif(not HAS_SDK, reason="claude_agent_sdk not installed")
@pytest.mark.skipif(not is_claude_code_available(), reason="claude CLI not available")
@pytest.mark.flaky(reruns=2)  # multi-step agentic; absorb stochastic misses
@pytest.mark.asyncio
async def test_the_agent_names_the_sector_bpms_by_position(
    graph_project: Path,  # noqa: F811
) -> None:
    """The agent reaches the graph and names BPM01 to BPM06 in position order."""
    render = render_dir(graph_project)
    assert "graph" in expected_mcp_servers(render), (
        "the readiness barrier would not wait for the graph server — .mcp.json "
        f"declares {sorted(expected_mcp_servers(render))}"
    )

    result = await run_sdk_query(graph_project, OPERATOR_PROMPT, max_turns=14, max_budget_usd=1.5)
    prose = "\n".join(result.text_blocks).strip()
    logger.info("tools called: %s", result.tool_names)
    logger.info("prose:\n%s", prose)

    graph_calls = [
        t for t in result.tool_traces if t.name.startswith("mcp__graph__") and not t.is_error
    ]
    assert graph_calls, (
        f"the agent answered without a successful graph call. Tools called: {result.tool_names}"
    )

    named = [name for name in named_in_order(prose) if name in _expected()]
    assert named == _expected(), (
        f"the answer names {named_in_order(prose)}; the section holds {_expected()} "
        "in that order by position"
    )


# ---------------------------------------------------------------------------
# Offline dry-verification of the grade.
# ---------------------------------------------------------------------------

_FLOOR_CASES: list[tuple[str, str, list[str]]] = [
    (
        "table",
        "| SR/BPM01 | 4.800 m |\n| SR/BPM02 | 6.287 m |\n| SR/BPM03 | 8.800 m |\n"
        "| SR/BPM04 | 11.177 m |\n| SR/BPM05 | 13.690 m |\n| SR/BPM06 | 14.499 m |",
        ["BPM01", "BPM02", "BPM03", "BPM04", "BPM05", "BPM06"],
    ),
    (
        "spaced_and_lowercase",
        "bpm 1 first, then BPM 2, bpm03, BPM04, BPM05 and finally bpm6",
        ["BPM01", "BPM02", "BPM03", "BPM04", "BPM05", "BPM06"],
    ),
    (
        "out_of_order",
        "BPM02, BPM01, BPM03",
        ["BPM02", "BPM01", "BPM03"],
    ),
]


@pytest.mark.harness_benchmark
@pytest.mark.parametrize(
    ("prose", "expected"),
    [pytest.param(prose, expected, id=f"floor-{case}") for case, prose, expected in _FLOOR_CASES],
)
def test_floor_named_in_order(prose: str, expected: list[str]) -> None:
    assert named_in_order(prose) == expected
