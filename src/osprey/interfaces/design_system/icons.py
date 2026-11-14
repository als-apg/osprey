"""The names of the design system's chrome icons, for Python readers.

``static/js/icons.js`` draws the icons and is the one place a name is defined.
A config key that names one (``web.panels.<id>.icon``) is judged at build time,
in Python, so the names are read from that file rather than listed a second
time here.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

ICONS_PATH = Path(__file__).parent / "static" / "js" / "icons.js"

_SHAPES_OPENER = "const SHAPES = Object.freeze({"
_SHAPES_CLOSER = "\n});"
_NAME = re.compile(r"""^  ['"]?([\w-]+)['"]?: \[""", re.MULTILINE)


@cache
def icon_names() -> frozenset[str]:
    """Every icon name ``icons.js`` draws: the keys of its ``SHAPES`` table.

    Raises:
        RuntimeError: When the table cannot be found or names nothing, which
            means ``icons.js`` changed shape and this reader did not follow.
    """
    source = ICONS_PATH.read_text(encoding="utf-8")
    start = source.find(_SHAPES_OPENER)
    end = source.find(_SHAPES_CLOSER, start)
    names = (
        frozenset(_NAME.findall(source[start + len(_SHAPES_OPENER) : end]))
        if start >= 0 and end >= 0
        else frozenset()
    )
    if not names:
        raise RuntimeError(f"no icon names found in {ICONS_PATH}: its SHAPES table moved")
    return names
