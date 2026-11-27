"""The demo census's named counts, read by name.

``tests/facility/golden/demo_census.json`` holds each count of the demo
facility once: its ``facility`` block the counts of the built facility file,
its ``facts`` block every other count a test pins. A test module binds its
constant to :func:`census` instead of restating the number, so a change to the
demo's size moves one file.

Import-light on purpose: the census is parsed with the standard library, and
nothing of osprey or of ``tests.facility`` is imported here.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

#: The census file, relative to nothing but this module.
CENSUS_PATH = Path(__file__).resolve().parent / "facility" / "golden" / "demo_census.json"


@cache
def _document() -> dict[str, Any]:
    document: dict[str, Any] = json.loads(CENSUS_PATH.read_text(encoding="utf-8"))
    return document


def census(name: str) -> Any:
    """The census value called ``name``: a ``facility`` count, else a fact.

    Raises:
        KeyError: The census names no such count.
    """
    document = _document()
    for block in ("facility", "facts"):
        if name in document[block]:
            return document[block][name]
    raise KeyError(f"{CENSUS_PATH.name} holds no count called {name!r}")
