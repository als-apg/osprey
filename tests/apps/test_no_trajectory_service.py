"""Tripwire: OSPREY ships no trajectory service.

The trajectory measurement is an application a facility builds on OSPREY's
generic mechanism (the application contract and the measurement-kind registry); it is not part
of the framework. This guard fails when ``src/`` names the service, its tools, its panel route,
its launch facts or its files, when the port layout reserves a port for it, or when the framework
registers a ``trajectory`` measurement kind of its own.

Only the exact tokens below match, on identifier boundaries. OSPREY's own words
stay legal: the ``turn_series`` wiring role, the engine's ``trajectory``
response kind, the turn-by-turn vocabulary roles and the approval hook's
setpoint-trajectory preview.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey.facility.kinds import builtin_kinds
from osprey.port_layout import LAYOUT
from tests._text_sweep import SHIPPED_TEXT_SUFFIXES, matching_lines, text_files

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The service's identifiers, as a trajectory application spells them.
TOKENS = (
    "services.trajectory",
    "mcp__trajectory__",
    "/panel/trajectory",
    "TRAJECTORY_SERVICE_KEY",
    "TRAJECTORY_LAUNCH_TOKEN",
    "trajectory_slots",
    "trajectory_catalog",
    "trajectory_rm",
)

#: Every text file under ``src/``, scripts and panel sources included.
_SUFFIXES = (*SHIPPED_TEXT_SUFFIXES, ".js", ".mjs", ".ts", ".html", ".css", ".conf")


def _token_pattern(token: str) -> str:
    """``token`` on identifier boundaries; an MCP prefix is open on its right."""
    head = r"(?<![\w])" if re.match(r"\w", token) else ""
    tail = "" if token.endswith("__") else r"(?![\w])"
    return head + re.escape(token) + tail


PATTERN = re.compile("|".join(_token_pattern(token) for token in TOKENS))


@pytest.mark.parametrize(
    "line",
    [
        "services.trajectory.port: 10082",
        'allow = ["mcp__trajectory__measure"]',
        'href="/panel/trajectory/"',
        "TRAJECTORY_SERVICE_KEY = 'x'",
        "os.environ['TRAJECTORY_LAUNCH_TOKEN']",
        "trajectory_slots:",
        "load(trajectory_catalog)",
        "path = 'trajectory_rm.json'",
    ],
)
def test_pattern_matches_each_token(line: str) -> None:
    assert PATTERN.search(line), line


@pytest.mark.parametrize(
    "line",
    [
        "role: turn_series",
        'response_matrix(kind="trajectory")',
        "bpm_tbt_x: {}",
        "bpm_arm_command: {}",
        "injection_request_command: {}",
        "def _setpoint_trajectory_preview(rows):",
        "tests/simulation/test_pyat_trajectory_rm.py",
        "services.trajectory_plot",
    ],
)
def test_pattern_spares_osprey_words(line: str) -> None:
    assert not PATTERN.search(line), line


def test_src_names_no_trajectory_service() -> None:
    files = text_files(_REPO_ROOT, ("src",), suffixes=_SUFFIXES)
    assert files, "the sweep found no files under src/"
    hits = matching_lines(_REPO_ROOT, files, PATTERN)
    assert hits == [], "\n".join(f"{path}:{number}: {line.strip()}" for path, number, line in hits)


def test_framework_registers_no_trajectory_kind() -> None:
    assert "trajectory" not in builtin_kinds()


def test_layout_reserves_no_trajectory_port() -> None:
    named = [
        slot.name
        for slot in LAYOUT
        if "trajectory" in slot.name or "trajectory" in (slot.config_key or "")
    ]
    assert named == [], f"the port layout reserves a trajectory port: {named}"
