"""The scripts and page templates OSPREY serves, for the source checks.

One walk decides what counts as a served script, so every check that refuses a
pattern in served code reads the same set: every ``.js``, ``.mjs``, ``.html``
and ``.html.j2`` below ``src/osprey``, minus vendored and minified files.
"""

from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src" / "osprey"

_SUFFIXES = (".js", ".mjs", ".html", ".html.j2")


def served_sources() -> list[Path]:
    """Every served script and page template below ``src/osprey``, sorted."""
    paths = []
    for path in sorted(SRC.rglob("*")):
        if not path.is_file() or not path.name.endswith(_SUFFIXES):
            continue
        rel = path.relative_to(SRC)
        if "vendor" in rel.parts or path.name.endswith(".min.js"):
            continue
        paths.append(path)
    return paths
