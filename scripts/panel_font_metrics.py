"""Write ``src/osprey/facility/panels/metrics.py`` from the viewer's text font.

The panel layout measures text in the font the viewer draws it with. This
script reads that font's advance widths, in thousandths of an em, for the
characters a panel writes, and prints the module. It needs ``fontTools``,
which is not a dependency of OSPREY:

    uv run --with fonttools python scripts/panel_font_metrics.py FONT_FILE \\
        > src/osprey/facility/panels/metrics.py
    uv run ruff format src/osprey/facility/panels/metrics.py

``FONT_FILE`` is the viewer's regular text face (TTF, OTF or WOFF).
"""

from __future__ import annotations

import sys

#: The characters measured: printable ASCII, then the marks a panel writes.
CHARACTERS = "".join(chr(i) for i in range(0x20, 0x7F)) + "—–·›≈…→←↑↓Σ±°µ×Δ²³Ω"


def main(path: str) -> str:
    from fontTools.ttLib import TTFont  # an optional tool, imported where it is used

    font = TTFont(path)
    cmap = font.getBestCmap()
    advances = font["hmtx"]
    per_em = font["head"].unitsPerEm
    family = font["name"].getDebugName(1)
    version = font["name"].getDebugName(5)
    widths = {
        c: round(advances[cmap[ord(c)]][0] * 1000 / per_em) for c in CHARACTERS if ord(c) in cmap
    }
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
'''


if __name__ == "__main__":
    sys.stdout.write(main(sys.argv[1]))
