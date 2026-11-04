"""What a repo's facility tree holds: bundled with OSPREY, seeded, imported or authored.

``osprey init`` copies a bundled facility from the package, and every copied text
file states it on its first line (:func:`header_line`). :func:`classify` reads a
facility directory back into those classes, so the facility import can say what
it may replace and what a person wrote.

The invariants:

* The header is the proof. A file that keeps its line is bundled even when edited;
  deleting the line claims the file.
* A file that cannot carry a line is bundled only while its bytes equal the package
  copy of the facility the headers name.
* A tree that was imported into is never wholly bundled.
* The knowledge indexes and the mml mapping prove nothing: the product rewrites the
  former and the import reads the latter, so they are in no class.
"""

from __future__ import annotations

import filecmp
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["HEADER_CAPABLE", "Classification", "bundled_origin", "classify", "header_line"]

#: The suffixes of the files that carry the bundled-origin line.
HEADER_CAPABLE = frozenset({".yaml", ".yml", ".md"})

_ORIGIN_TEXT = "Bundled with OSPREY: the {} facility. Delete this line to make the file yours."
_ORIGIN = re.compile(re.escape(_ORIGIN_TEXT).replace(r"\{\}", r"(\S+)"))
_LAYER_DIR = "imported"
_FRONTMATTER = "---"


def header_line(name: str) -> str:
    """The first line of a file copied from the bundled facility ``name``."""
    return f"# {_ORIGIN_TEXT.format(name)}"


def _first_line(path: Path) -> str:
    """The line a file's header sits on: line 1, or the first inside Markdown frontmatter.

    Only the newline ends a line: ``str.splitlines`` would also end one at a form
    feed or a Unicode line separator and pass a line that runs on.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n", 2)
    if path.suffix == ".md" and lines[0] == _FRONTMATTER and len(lines) > 1:
        return lines[1]
    return lines[0]


def bundled_origin(path: Path) -> str | None:
    """The bundled facility a file's header names, or ``None`` when it has no such line."""
    if path.suffix not in HEADER_CAPABLE:
        return None
    line = _first_line(path)
    if line.startswith("# "):
        text = line.removeprefix("# ")
    elif line.startswith("<!-- ") and line.endswith(" -->"):
        text = line.removeprefix("<!-- ").removesuffix(" -->")
    else:
        return None
    match = _ORIGIN.fullmatch(text)
    return match[1] if match else None


def _is_seeded(path: Path) -> bool:
    """Whether a file opens with the header line the facility import seeds files under."""
    from osprey.facility.layers import HEADER

    return path.suffix in HEADER_CAPABLE and _first_line(path) == HEADER


@dataclass(frozen=True)
class Classification:
    """A facility directory by origin; each class is a sorted tuple of relative POSIX paths.

    Attributes:
        name: The one bundled facility the headers agree on, or ``None``.
        bundled: The files copied from that facility.
        seeded: The files an import seeded.
        layer: The files under ``imported/``.
        authored: Every other file.
    """

    name: str | None
    bundled: tuple[str, ...]
    seeded: tuple[str, ...]
    layer: tuple[str, ...]
    authored: tuple[str, ...]

    @property
    def wholly_bundled(self) -> bool:
        """Whether the tree holds bundled files and nothing else that counts."""
        return bool(self.bundled) and self.authored == self.seeded == self.layer == ()


def _proves_nothing(relative: Path) -> bool:
    from osprey.facility.layers.mml.mapping import MAPPING_FILE

    if relative.as_posix() == MAPPING_FILE:
        return True
    return relative.parts[0] == "knowledge" and relative.name == "index.md"


def classify(facility_dir: Path, template_root: Path) -> Classification:
    """Sort every file under ``facility_dir`` by where it came from.

    Args:
        facility_dir: The ``data/facility`` directory; a missing one is empty.
        template_root: The package's template root, which holds the bundled facilities.

    Returns:
        The classification.
    """
    from osprey.cli.templates.preset_data import FACILITIES_DIRNAME

    files = [
        (path, path.relative_to(facility_dir))
        for path in sorted(facility_dir.rglob("*"))
        if path.is_file() and not _proves_nothing(path.relative_to(facility_dir))
    ]
    origins = {
        relative: bundled_origin(path)
        for path, relative in files
        if relative.parts[0] != _LAYER_DIR
    }
    named = {origin for origin in origins.values() if origin is not None}
    name = named.pop() if len(named) == 1 else None
    package = template_root / FACILITIES_DIRNAME / name if name is not None else None

    classes: dict[str, list[str]] = {"bundled": [], "seeded": [], "layer": [], "authored": []}
    for path, relative in files:
        if relative.parts[0] == _LAYER_DIR:
            found = "layer"
        elif _is_seeded(path):
            found = "seeded"
        elif package is None:
            found = "authored"
        elif origins[relative] == name or (
            path.suffix not in HEADER_CAPABLE
            and (package / relative).is_file()
            and filecmp.cmp(path, package / relative, shallow=False)
        ):
            found = "bundled"
        else:
            found = "authored"
        classes[found].append(relative.as_posix())
    return Classification(
        name=name,
        bundled=tuple(classes["bundled"]),
        seeded=tuple(classes["seeded"]),
        layer=tuple(classes["layer"]),
        authored=tuple(classes["authored"]),
    )
