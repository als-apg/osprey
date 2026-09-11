"""Every bundled font ships with the licence text that permits bundling it.

``src/osprey/interfaces/shared_fonts/`` puts five OFL-licensed families into
every wheel and every image OSPREY builds. The OFL requires the licence to
travel with the binaries, and an institution reviewing a mirrored wheel looks
for exactly that file — so a font added without one is a rollout blocker
discovered by a lawyer rather than by CI.

The check is deliberately coarse: it asks that each family's name appears in
the notice block of ``OFL.txt``, not that a particular sentence does. That is
enough to fail on the thing that actually happens — a new ``.ttf`` dropped in
beside the others — and it does not pin the wording of an upstream copyright
notice.

It is the *notice block* and the *whole* family name on purpose. Searching the
whole file would let a family pass on a word the licence boilerplate happens to
use ("open", "font", "mono"), and matching only the first hyphen-separated
token would cover ``syne-mono`` with a line that names some other Syne face.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

FONT_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "interfaces" / "shared_fonts"
LICENCE = FONT_DIR / "OFL.txt"

#: A bundled file is named ``<family>-<weight>.ttf``; the family is what has a
#: copyright holder.
_FAMILY = re.compile(r"^(?P<family>.+)-\d+$")


def _families() -> set[str]:
    found = set()
    for path in FONT_DIR.glob("*.ttf"):
        match = _FAMILY.match(path.stem)
        assert match, f"{path.name} is not named <family>-<weight>.ttf"
        found.add(match.group("family"))
    return found


def test_the_directory_still_bundles_fonts() -> None:
    """A vacuous pass here would hide the whole check."""
    assert _families(), f"no .ttf files under {FONT_DIR}"


def test_the_licence_file_ships_beside_the_fonts() -> None:
    assert LICENCE.is_file(), (
        f"{LICENCE} is missing: the OFL requires its text to travel with the "
        f"binaries, and the wheel carries only what is under src/osprey."
    )
    assert "SIL OPEN FONT LICENSE Version 1.1" in LICENCE.read_text(encoding="utf-8")


def _notice_block() -> str:
    """The copyright notices, without the licence text they sit above.

    A rule of dashes separates the two halves of ``OFL.txt``.
    """
    text = LICENCE.read_text(encoding="utf-8")
    notices, sep, _licence = text.partition("-----")
    assert sep, f"{LICENCE} has no rule separating the notices from the licence text"
    return notices.lower().replace("-", " ").replace("_", " ")


def test_every_bundled_family_is_named_in_the_licence_file() -> None:
    notices = _notice_block()
    uncovered = sorted(family for family in _families() if family.replace("-", " ") not in notices)
    assert uncovered == [], (
        "font families bundled with no copyright line in OFL.txt: "
        f"{uncovered}. Add the notice carried in the font binary's own name "
        "table (name id 0) before shipping the file."
    )
