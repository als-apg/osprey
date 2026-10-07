"""No served page formats an instant itself; ``facility-time.js`` does.

Every page OSPREY serves renders instants on the facility clock, in the zone
``system.timezone`` names. The one place a page formats an instant is
``design_system/static/js/facility-time.js`` (``formatFacilityTime`` and its
siblings), which reads the zone stamped on ``<html>``. A page that calls a Date's
locale or field accessors itself renders in the viewer's zone instead.

This guard walks every script and page template under ``src/osprey`` for the
forms that format an instant in the viewer's zone. The pattern is a net, not a
proof: ``d.toLocaleString()`` on a Date variable with no arguments cannot be
told from a number's, so review still owns that form.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[3] / "src" / "osprey"

_SUFFIXES = (".js", ".mjs", ".html", ".html.j2")

#: The forms that format an instant in the viewer's zone. The last alternative
#: is ``toLocaleString`` with a date option object, which leaves number
#: formatting alone.
_INSTANT_FORMATTING = re.compile(
    r"\.(?:toLocaleTimeString|toLocaleDateString|toTimeString|toDateString|toUTCString"
    r"|toGMTString)\("
    r"|\.get(?:UTC)?(?:Hours|Minutes|Seconds|Milliseconds|Date|Day|Month|FullYear)\("
    r"|\bIntl\.DateTimeFormat\b"
    r"|new Date\([^()]*\)\.toLocaleString\("
    r"|\.toLocaleString\(\s*[^)]{0,80}?\{[^}]{0,200}?"
    r"\b(?:year|month|day|weekday|hour|minute|second|timeZone)\b",
    re.S,
)

#: Files allowed to format an instant themselves, with the reason.
_EXEMPT = {
    "interfaces/design_system/static/js/facility-time.js": "the formatter itself",
    "interfaces/web_terminal/static/js/bar-items.js": (
        "the clock: its local and UTC faces read the viewer's and UTC's wall clock, "
        "and its facility face reads fixed en-US fields"
    ),
}


def _served_sources() -> list[Path]:
    paths = []
    for path in sorted(_SRC.rglob("*")):
        if not path.is_file() or not path.name.endswith(_SUFFIXES):
            continue
        rel = path.relative_to(_SRC)
        if "vendor" in rel.parts or path.name.endswith(".min.js"):
            continue
        paths.append(path)
    return paths


def _hits(text: str) -> list[int]:
    """The 1-based line of every match in *text*."""
    return [text.count("\n", 0, m.start()) + 1 for m in _INSTANT_FORMATTING.finditer(text)]


def test_no_served_page_formats_an_instant_itself():
    found = []
    for path in _served_sources():
        rel = path.relative_to(_SRC).as_posix()
        if rel in _EXEMPT:
            continue
        found.extend(f"{rel}:{line}" for line in _hits(path.read_text(encoding="utf-8")))
    assert not found, (
        "These pages format an instant in the viewer's zone; call formatFacilityTime from "
        "/design-system/js/facility-time.js instead:\n  " + "\n  ".join(found)
    )


@pytest.mark.parametrize("rel", sorted(_EXEMPT))
def test_each_exemption_still_formats_an_instant(rel):
    path = _SRC / rel
    assert path.is_file(), f"{rel} is exempt but does not exist"
    assert _hits(path.read_text(encoding="utf-8")), f"{rel} is exempt but formats no instant"


@pytest.mark.parametrize(
    "snippet",
    [
        "d.toLocaleTimeString()",
        "d.toLocaleDateString('en-US', {})",
        "d.getHours()",
        "d.getUTCMinutes()",
        "when.getFullYear()",
        "new Intl.DateTimeFormat(undefined, {})",
        "new Date(ts).toLocaleString()",
        "d.toLocaleString(undefined, {\n  month: 'short',",
    ],
)
def test_the_pattern_sees_every_form_it_forbids(snippet):
    assert _INSTANT_FORMATTING.search(snippet)


@pytest.mark.parametrize(
    "snippet",
    [
        "count.toLocaleString()",
        "value.toLocaleString('en-US')",
        "n.toLocaleString(undefined, {maximumFractionDigits: 2})",
        "${summary.total_points.toLocaleString()}</span>",
    ],
)
def test_the_pattern_leaves_number_formatting_alone(snippet):
    assert not _INSTANT_FORMATTING.search(snippet)
