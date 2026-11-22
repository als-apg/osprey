"""Write ``src/osprey/facility/panels/metrics.py`` from the viewer's text font.

The panel layout measures text in the font the viewer draws it with. This
script reads that font's advance widths, in thousandths of an em, for the
characters a panel writes, and prints the module:

    uv run python scripts/panel_font_metrics.py \\
        > src/osprey/facility/panels/metrics.py
    uv run ruff format src/osprey/facility/panels/metrics.py

The font file is :data:`FONT_FILE`: the regular face of IBM Plex Sans, as the
npm package ``@ibm/plex-sans`` 1.1.0 ships it (``fonts/complete/woff``), which
is the package the viewer loads its text font from. The font is IBM's
(https://github.com/ibm/plex), under the SIL Open Font License 1.1; the licence
is beside it in ``scripts/panel_font/OFL.txt``. A test holds ``metrics.py`` to
this file, so when the viewer's font changes, replace the file (and its
licence), run the two lines above, and commit the three together.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: The viewer's regular text face, the one file the table is measured from.
FONT_FILE = Path(__file__).parent / "panel_font" / "IBMPlexSans-Regular.woff"

#: The characters measured: printable ASCII, then the marks a panel writes.
CHARACTERS = "".join(chr(i) for i in range(0x20, 0x7F)) + "—–·›≈≠…→←↑↓Σ±°µ×Δ²³Ω"


def measure(path: Path | str = FONT_FILE) -> tuple[str, str, dict[str, int], int, int]:
    """The font's family, its version, its advance width per character, its line
    height (the browser's ``line-height: normal``) and its ascent (the baseline
    under the top of that line), in thousandths of an em."""
    from fontTools.ttLib import TTFont  # a matplotlib dependency, imported where it is used

    font = TTFont(str(path))
    cmap = font.getBestCmap()
    advances = font["hmtx"]
    per_em = font["head"].unitsPerEm
    widths = {
        c: round(advances[cmap[ord(c)]][0] * 1000 / per_em) for c in CHARACTERS if ord(c) in cmap
    }
    hhea = font["hhea"]
    line = round((hhea.ascent - hhea.descent + hhea.lineGap) * 1000 / per_em)
    ascent = round((hhea.ascent + hhea.lineGap / 2) * 1000 / per_em)
    return font["name"].getDebugName(1), font["name"].getDebugName(5), widths, line, ascent


def main(path: Path | str = FONT_FILE) -> str:
    family, version, widths, line, ascent = measure(path)
    rows = "\n".join(f"    {c!r}: {w}," for c, w in widths.items())
    return f'''"""Advance widths of the viewer's text font, for measuring text in a panel.

Written by ``scripts/panel_font_metrics.py`` from {family}, {version}; do not edit.
Each width is in thousandths of an em; a character not listed is measured as
:data:`UNLISTED`, the widest listed one, so an estimate never falls short.
"""

from __future__ import annotations

#: The family the viewer draws a panel's text in.
FONT = {family!r}

#: The unit of :data:`WIDTHS`: thousandths of an em.
PER_EM = 1000

#: Advance width per character, in thousandths of an em.
WIDTHS: dict[str, int] = {{
{rows}
}}

#: The width of a character :data:`WIDTHS` does not list.
UNLISTED = max(WIDTHS.values())

#: The height of one line of text (the browser's ``line-height: normal``), in
#: thousandths of an em: a label's box lower than this the viewer shrinks.
LINE = {line}

#: The baseline under the top of one line of text, in thousandths of an em: a
#: text centred in its box has its baseline ``ASCENT - LINE / 2`` below the middle.
ASCENT = {ascent}
'''


if __name__ == "__main__":
    sys.stdout.write(main(*sys.argv[1:2]))
