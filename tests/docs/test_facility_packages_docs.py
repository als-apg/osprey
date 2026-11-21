"""The facility-packages docs draw their pictures as OSPREY diagrams, never ASCII.

Two diagrams carry the end state of the facility packages: the package map
(``_diagrams/facility-packages.html``: the framework, osprey-facility,
osprey-panels and osprey-connectors with the allowed dependency directions)
and today versus target (``_diagrams/facility-packages-today.html``: the
import cycle and the ratchet by edge class and removing phase). Each must be
an accessible OSPREY diagram: the ``osprey-diagram`` wrapper, one ``svg`` with
``role="img"`` labelled by a non-empty ``title`` and ``desc``, and colour only
through the diagram classes and ``var(--od-*)`` tokens, so both themes render.

The two new pages that show them (``architecture/facility-packages.rst`` and
``reference/contracts/facility-json.rst``) carry no box-drawing glyph and no
ASCII-art drawing. A page not written yet is skipped here; the phase check
asserts both exist. Other pages keep their file-tree blocks: the scan is
scoped to these two on purpose.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs" / "source"
DIAGRAMS = DOCS / "_diagrams"

NEW_DIAGRAMS = ("facility-packages.html", "facility-packages-today.html")
NEW_PAGES = (
    "architecture/facility-packages.rst",
    "reference/contracts/facility-json.rst",
)

_SVG = re.compile(r"<svg\b([^>]*)>", re.S)
_ATTR = re.compile(r'\b([\w-]+)="([^"]*)"')
_COLOUR_ATTR = re.compile(r'\b(?:fill|stroke|color|stop-color)="(?!none")[^"]*"')
_COLOUR_LITERAL = re.compile(r"#[0-9a-fA-F]{3,8}\b|\b(?:rgba?|hsla?)\(")
_STYLE = re.compile(r'\bstyle="([^"]*)"')


def _diagram_offences(text: str) -> list[str]:
    """Every way ``text`` falls short of an accessible, themed OSPREY diagram."""
    offences: list[str] = []
    body = text.strip()
    if not body.startswith('<div class="osprey-diagram">') or not body.endswith("</div>"):
        offences.append('not wrapped in <div class="osprey-diagram">')
    svgs = _SVG.findall(body)
    if len(svgs) != 1:
        offences.append(f"expected one <svg>, found {len(svgs)}")
        return offences
    attrs = dict(_ATTR.findall(svgs[0]))
    if attrs.get("role") != "img":
        offences.append('<svg> lacks role="img"')
    ids = attrs.get("aria-labelledby", "").split()
    if len(ids) != 2:
        offences.append("<svg> aria-labelledby does not name a title and a desc")
    else:
        for tag, ident in zip(("title", "desc"), ids, strict=True):
            match = re.search(rf'<{tag} id="{re.escape(ident)}">(.*?)</{tag}>', body, re.S)
            if match is None:
                offences.append(f'no <{tag} id="{ident}">')
            elif not match.group(1).strip():
                offences.append(f"<{tag}> is empty")
    for attr in _COLOUR_ATTR.findall(body):
        offences.append(f"colour attribute {attr}")
    for literal in _COLOUR_LITERAL.findall(body):
        offences.append(f"colour literal {literal}")
    for style in _STYLE.findall(body):
        for decl in filter(None, (d.strip() for d in style.split(";"))):
            if "var(--od-" not in decl:
                offences.append(f"style {decl!r} not through a var(--od-*) token")
    return offences


#: Box-drawing (U+2500-U+257F), block elements and the triangle arrowheads
#: drawings use. A prose arrow (a \u2192 b) is not a drawing and passes.
_GLYPHS = re.compile("[\u2500-\u259f\u25b2-\u25c4]")
_ASCII_ART = (
    re.compile(r"\+[-=]{2,}\+?"),  # +----+ box corners and grid-table rules
    re.compile(r"^\s*\|.*\|\s*$"),  # |  boxed  | rows
    re.compile(r"-{2,}>|<-{2,}|={2,}>|<={2,}"),  # --> and ==> drawn arrows
    re.compile(r"^\s*[|^v]\s*$"),  # a lone vertical connector
)


def _ascii_art_lines(text: str) -> list[str]:
    """Lines of ``text`` that draw: box-drawing glyphs or ASCII-art shapes."""
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        if _GLYPHS.search(line) or any(p.search(line) for p in _ASCII_ART):
            found.append(f"{number}: {line}")
    return found


@pytest.mark.parametrize("name", NEW_DIAGRAMS)
def test_each_new_diagram_is_an_accessible_osprey_diagram(name: str) -> None:
    path = DIAGRAMS / name
    assert path.is_file(), f"missing diagram {path.relative_to(DOCS)}"
    offences = _diagram_offences(path.read_text(encoding="utf-8"))
    assert offences == [], f"{name}:\n" + "\n".join(offences)


@pytest.mark.parametrize("page", NEW_PAGES)
def test_each_new_page_draws_no_ascii(page: str) -> None:
    path = DOCS / page
    if not path.is_file():
        pytest.skip(f"{page} not written yet; the phase check asserts it exists")
    found = _ascii_art_lines(path.read_text(encoding="utf-8"))
    assert found == [], f"{page} draws in text:\n" + "\n".join(found)


GOOD_DIAGRAM = """<div class="osprey-diagram">
<svg viewBox="0 0 10 10" role="img" aria-labelledby="x-title x-desc">
  <title id="x-title">A box</title>
  <desc id="x-desc">One box.</desc>
  <defs><marker id="x-arr"><polygon points="0,0 8,3 0,6" style="fill: var(--od-muted)"/></marker></defs>
  <rect class="nb" x="1" y="1" width="8" height="8" rx="6"/>
  <line class="e" x1="0" y1="0" x2="1" y2="1" fill="none"/>
</svg>
</div>
"""


def test_a_sound_diagram_passes() -> None:
    assert _diagram_offences(GOOD_DIAGRAM) == []


@pytest.mark.parametrize(
    ("broken", "offence"),
    [
        (GOOD_DIAGRAM.replace('<div class="osprey-diagram">', "<div>"), "not wrapped"),
        (GOOD_DIAGRAM.replace(' role="img"', ""), 'role="img"'),
        (GOOD_DIAGRAM.replace(' aria-labelledby="x-title x-desc"', ""), "aria-labelledby"),
        (GOOD_DIAGRAM.replace('<title id="x-title">A box</title>', ""), "<title"),
        (GOOD_DIAGRAM.replace("One box.", " "), "<desc> is empty"),
        (GOOD_DIAGRAM.replace('class="nb"', 'class="nb" fill="#ffffff"'), "colour"),
        (GOOD_DIAGRAM.replace("var(--od-muted)", "rgb(0,0,0)"), "colour literal"),
        (GOOD_DIAGRAM.replace("var(--od-muted)", "red"), "var(--od-*)"),
    ],
    ids=["wrapper", "role", "labelledby", "title", "desc", "fill-attr", "rgb", "style"],
)
def test_a_broken_diagram_is_caught(broken: str, offence: str) -> None:
    offences = _diagram_offences(broken)
    assert any(offence in o for o in offences), offences


@pytest.mark.parametrize(
    "drawing",
    [
        "    records ──► build",
        "    +--------+",
        "    | panels |",
        "    framework --> facility",
        "    facility <-- panels",
        "        |",
        "        v",
        "    description ◄── views",
    ],
)
def test_a_text_drawing_is_caught(drawing: str) -> None:
    assert _ascii_art_lines(f"Title\n=====\n\n{drawing}\n") != []


def test_prose_tables_and_code_are_not_drawings() -> None:
    page = (
        "The facility document\n"
        "=====================\n\n"
        "Readers\n-------\n\n"
        ".. list-table::\n   :header-rows: 1\n\n"
        "   * - Reader\n     - Checks\n"
        "   * - osprey-panels\n     - the header ``osprey.facility.facility/N``\n\n"
        ".. code-block:: text\n\n"
        "   facility.json: header osprey.facility.facility/2, this reader expects /3;\n"
        "   rebuild with --force or pin the reader -- see the contract.\n\n"
        "A reader that meets a newer ``/N`` refuses; ``importer -> root`` names an edge;\n"
        "the build \u2192 facility.json \u2192 the readers.\n"
    )
    assert _ascii_art_lines(page) == []
