"""The ``canopy-screens`` skill: a build-time guide to the project's CANOPY screen library.

The guide is read from the library the deployment ships, so every library screen is
in it under its ``ref``, and each screen lists the devices it draws or binds by their
labels. The control-assistant build renders it; a project without the CANOPY panel
renders none.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from osprey.integrations.canopy.screen_guide import (
    build_screen_guide,
    compress,
    screen_guide_context,
)
from tests._builds import BuiltProject

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "src/osprey/templates/apps/control_assistant/data/canopy/library"
SKILL = Path(".claude/skills/canopy-screens/SKILL.md")
_RUN = re.compile(r"^(?P<stem>.*?)(?P<first>\d+)–(?P<last>\d+)$")


def _labels(screen, system: str) -> list[str]:
    """The labels a guide screen lists under ``system``."""
    return [label for name, labels in screen.devices if name == system for label in labels]


def _expand(labels: list[str]) -> set[str]:
    """The labels a compressed list stands for."""
    out = set()
    for label in labels:
        match = _RUN.match(label)
        if match is None:
            out.add(label)
            continue
        width = len(match["first"])
        out |= {
            f"{match['stem']}{n:0{width}d}"
            for n in range(int(match["first"]), int(match["last"]) + 1)
        }
    return out


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["BPM 39", "BPM 37", "BPM 38"], ["BPM 37–39"]),
        (["EPU7", "EPU8"], ["EPU7", "EPU8"]),
        ([f"BPM {n:02d}" for n in range(1, 73)], ["BPM 01–72"]),
        (["QF 13", "QF 14", "SEK", "IG ST07A"], ["IG ST07A", "QF 13", "QF 14", "SEK"]),
        (["U1", "U14", "U9", "U10", "U11"], ["U1", "U9", "U10", "U11", "U14"]),
    ],
)
def test_compress_writes_numbered_runs_first_to_last(labels, expected) -> None:
    assert compress(labels) == expected


def test_the_guide_holds_every_library_screen_and_what_it_opens(
    built_control_assistant: BuiltProject,
) -> None:
    guide = build_screen_guide(LIBRARY, built_control_assistant.facility)
    files = sorted((LIBRARY / "screens").rglob("*.screen.json"))
    refs = [f"library/{p.relative_to(LIBRARY / 'screens').as_posix()[:-12]}" for p in files]
    assert [s.ref for s in guide.screens] == refs
    assert not guide.warnings
    by_ref = {s.ref: s for s in guide.screens}
    assert {f"library/sr/sector-{n:02d}" for n in range(1, 13)} <= set(
        by_ref["library/facility/overview"].opens
    )
    assert by_ref["library/sr/sector-07"].opens == (
        "library/sr/_absorbers-07",
        "library/sr/_vacuum-07",
        "library/sr/orbit-feedback",
    )
    assert by_ref["library/sr/_vacuum-07"].is_sub_screen
    assert [system for system, _ in by_ref["library/facility/overview"].devices] == [
        "BR",
        "BTS",
        "GTL",
        "LN",
        "LTB",
        "SR",
    ]


def test_each_sector_lists_its_bpms_pumps_gauges_valves_and_insertion_devices(
    built_control_assistant: BuiltProject,
) -> None:
    """Every such device the records place in a sector is on that sector's screen, by label."""
    facility = built_control_assistant.facility
    guide = {s.ref: s for s in build_screen_guide(LIBRARY, facility).screens}
    classes = {"BeamPositionMonitor", "Pump", "InsertionDevice", "Gauge", "Valve"}
    for n in range(1, 13):
        placed = {
            d["label"].split(" (")[0]
            for d in facility["devices"]
            if d.get("place") == f"SR/SECT{n}" and d.get("class") in classes
        }
        assert placed, n
        shown = _expand(_labels(guide[f"library/sr/sector-{n:02d}"], "SR"))
        assert placed <= shown, (n, sorted(placed - shown))
    assert {"EPU7", "EPU8", "BPM 37", "ION-PUMP 38"} <= _expand(
        _labels(guide["library/sr/sector-07"], "SR")
    )


def test_the_build_renders_the_skill_from_the_library(
    built_control_assistant: BuiltProject,
) -> None:
    text = (built_control_assistant.build_dir / SKILL).read_text(encoding="utf-8")
    assert text.startswith("---\nname: canopy-screens\n")
    for path in sorted((LIBRARY / "screens").rglob("*.screen.json")):
        ref = f"library/{path.relative_to(LIBRARY / 'screens').as_posix()[:-12]}"
        assert f"`{ref}`" in text, ref
    sector = text.split("— `library/sr/sector-07`", 1)[1].split("\n### ", 1)[0]
    assert "- SR: " in sector and "EPU7" in sector and "ION-PUMP 37–42" in sector
    # The bare tools open empty; each seeded view is a collection member's own query.
    assert "`open_tool` with `tool: waveform`" not in text
    assert "Collection **Control assistant**" in text
    assert "Septum pulse (waveform on `pv=SR:MAG:SEPTUM:01:CURRENT:WAVEFORM&persist=10`)" in text


def test_a_project_without_the_canopy_panel_gets_no_guide(tmp_path: Path) -> None:
    assert screen_guide_context({"web": {"panels": {}}}, tmp_path) is None
    assert (
        screen_guide_context({"web": {"panels": {"canopy": {"enabled": False}}}}, tmp_path) is None
    )
