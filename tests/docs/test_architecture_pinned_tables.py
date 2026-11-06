"""Three tables the docs retype from code state what the code holds.

- The method table of ``architecture/facility-file.rst`` lists the public surface
  of :class:`osprey.facility.records.FacilityRecords`, each row with the
  parameters of the member it names, and the ``RECORD_KINDS`` / ``LINK_KINDS``
  orders it quotes are the module's own.
- The pin table of ``architecture/virtual-accelerator.rst`` lists every exact
  ``lume-*`` pin of the root and the connectors ``pyproject.toml``, and no other.
- The bibtex block of the landing page cites what ``CITATION.cff``'s
  ``preferred-citation`` cites.

Each comparison is a function returning the differences it finds, so the negative
controls below can plant a drift and watch it reported.
"""

from __future__ import annotations

import inspect
import re
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from osprey.facility import records
from osprey.facility.records import LINK_KINDS, RECORD_KINDS, FacilityRecords

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _REPO_ROOT / "docs" / "source"
_FACILITY_FILE_PAGE = _DOCS / "architecture" / "facility-file.rst"
_VA_PAGE = _DOCS / "architecture" / "virtual-accelerator.rst"
_INDEX_PAGE = _DOCS / "index.rst"
_PYPROJECTS = (
    _REPO_ROOT / "pyproject.toml",
    _REPO_ROOT / "packages" / "osprey-connectors" / "pyproject.toml",
)
_CITATION = _REPO_ROOT / "CITATION.cff"

#: A ``list-table`` row's first cell when it is one RST inline literal.
_FIRST_CELL = re.compile(r"^   \* - ``(?P<cell>[^`]+)``$")
#: A method cell: ``name`` or ``name(param, ...)``.
_MEMBER = re.compile(r"^(?P<name>\w+)(?:\((?P<params>[^)]*)\))?$")
#: An exact requirement, extras kept: ``name[extra,...]==version``, marker ignored.
_PIN = re.compile(r"^(?P<name>lume-[\w.-]+(?:\[[^\]]*\])?)==(?P<version>[^;\s]+)")
#: One ``field = {value},`` line of a bibtex entry.
_BIBTEX_FIELD = re.compile(r"^\s*(?P<field>\w+) = \{(?P<value>.*)\},$", re.MULTILINE)


def _section(text: str, heading: str) -> list[str]:
    """The lines under ``heading``, up to the next ``=``-underlined heading."""
    lines = text.splitlines()
    start = lines.index(heading) + 2
    for n in range(start, len(lines) - 1):
        if lines[n].strip() and lines[n + 1].startswith("==="):
            return lines[start:n]
    return lines[start:]


def _rows(lines: list[str]) -> list[list[str]]:
    """The ``list-table`` rows in ``lines``, each as its cells' first lines."""
    rows: list[list[str]] = []
    for line in lines:
        if line.startswith("   * - "):
            rows.append([line[len("   * - ") :]])
        elif line.startswith("     - ") and rows:
            rows[-1].append(line[len("     - ") :])
    return rows


# --- FacilityRecords ---------------------------------------------------------


def _method_table(text: str) -> dict[str, tuple[str, ...] | None]:
    """The members the method table names: ``None`` for a property, else its params."""
    entries: dict[str, tuple[str, ...] | None] = {}
    for line in _section(text, "Reading the facility file"):
        if not (cell := _FIRST_CELL.match(line)):
            continue
        member = _MEMBER.match(cell["cell"])
        assert member is not None, f"not a member cell: {cell['cell']!r}"
        name = "__init__" if member["name"] == FacilityRecords.__name__ else member["name"]
        params = member["params"]
        entries[name] = (
            None if params is None else tuple(p.strip() for p in params.split(",") if p.strip())
        )
    return entries


def _public_members(cls: type) -> dict[str, tuple[str, ...] | None]:
    """``cls``'s public members and ``__init__``: ``None`` for a property, else its params."""
    members: dict[str, tuple[str, ...] | None] = {}
    for name, value in vars(cls).items():
        if name.startswith("_") and name != "__init__":
            continue
        if isinstance(value, property):
            members[name] = None
            continue
        function = value.__func__ if isinstance(value, classmethod | staticmethod) else value
        if not callable(function):
            continue
        params = list(inspect.signature(function).parameters)
        if not isinstance(value, staticmethod):
            params = params[1:]
        members[name] = tuple(params)
    return members


def _method_drift(page: dict[str, tuple[str, ...] | None], cls: type) -> list[str]:
    code = _public_members(cls)
    drift = [f"the page lists {name}, which {cls.__name__} lacks" for name in page.keys() - code]
    drift += [f"{cls.__name__}.{name} is missing from the page" for name in code.keys() - page]
    drift += [
        f"{name}: the page shows {page[name]}, the code takes {code[name]}"
        for name in page.keys() & code.keys()
        if page[name] != code[name]
    ]
    return sorted(drift)


def _order_cells(text: str) -> dict[str, list[str]]:
    """The kinds each ``RECORD_KINDS`` / ``LINK_KINDS`` order cell lists, in order."""
    found: dict[str, list[str]] = {}
    for row in _rows(_section(text, "Reading the facility file")):
        for cell in row[1:]:
            if order := re.match(r"^``(RECORD_KINDS|LINK_KINDS)``: (.+)\.$", cell):
                found[order[1]] = re.split(r",\s*(?:then\s+)?", order[2])
    return found


def test_the_method_table_lists_the_facility_records_surface() -> None:
    page = _method_table(_FACILITY_FILE_PAGE.read_text(encoding="utf-8"))

    assert _method_drift(page, FacilityRecords) == []


def test_the_method_table_quotes_the_record_and_link_kind_orders() -> None:
    orders = _order_cells(_FACILITY_FILE_PAGE.read_text(encoding="utf-8"))

    assert orders == {
        "RECORD_KINDS": [kind for kind, _key, _id in RECORD_KINDS],
        "LINK_KINDS": list(LINK_KINDS),
    }
    assert {"RECORD_KINDS", "LINK_KINDS"} <= set(records.__all__)


def test_a_row_the_class_lacks_or_a_member_the_page_lacks_is_reported() -> None:
    page = _method_table(_FACILITY_FILE_PAGE.read_text(encoding="utf-8"))

    def nearest(self: FacilityRecords, address: str) -> None: ...

    grown = {**vars(FacilityRecords), "nearest": nearest}
    Grown = type(
        "Grown", (), {k: v for k, v in grown.items() if k not in {"__dict__", "__weakref__"}}
    )

    planted = {**page, "retired": ()}
    planted["record"] = ("kind",)
    drift = _method_drift(planted, Grown)

    assert drift == [
        "Grown.nearest is missing from the page",
        "record: the page shows ('kind',), the code takes ('kind', 'id')",
        "the page lists retired, which Grown lacks",
    ]


# --- LUME pins ---------------------------------------------------------------


def _requirements(pyproject: dict[str, Any]) -> Iterable[str]:
    project = pyproject.get("project", {})
    yield from project.get("dependencies", [])
    for extra in project.get("optional-dependencies", {}).values():
        yield from extra
    for group in pyproject.get("dependency-groups", {}).values():
        yield from (item for item in group if isinstance(item, str))


def _normalised(name: str) -> str:
    """``name[extras]`` with the extras stripped and sorted, so their spelling is no drift."""
    base, _, extras = name.partition("[")
    if not extras:
        return base.strip()
    listed = sorted(extra.strip() for extra in extras.rstrip("]").split(",") if extra.strip())
    return f"{base.strip()}[{','.join(listed)}]"


def test_extras_compare_by_set_not_spelling() -> None:
    assert _normalised("lume-pva-apg[pva, ca]") == _normalised("lume-pva-apg[ca,pva]")
    assert _normalised("lume-pva-apg[ca]") != _normalised("lume-pva-apg[ca,pva]")
    assert _normalised("lume-base") == "lume-base"


def _code_pins() -> dict[str, set[str]]:
    """Every exact ``lume-*`` pin of the two pyproject files, by name with extras."""
    pins: dict[str, set[str]] = {}
    for path in _PYPROJECTS:
        pyproject = tomllib.loads(path.read_text(encoding="utf-8"))
        for requirement in _requirements(pyproject):
            if pin := _PIN.match(requirement.replace(" ", "")):
                pins.setdefault(_normalised(pin["name"]), set()).add(pin["version"])
    return pins


def _pin_table(text: str) -> dict[str, set[str]]:
    """The package and pin cells of the LUME pin table."""
    table: dict[str, set[str]] = {}
    for row in _rows(_section(text, "The LUME stack and its pins")):
        package, pin = (re.fullmatch(r"``([^`]+)``", cell) for cell in row[:2])
        if package and pin:
            table.setdefault(_normalised(package[1]), set()).add(pin[1])
    return table


def _pin_drift(page: dict[str, set[str]], code: dict[str, set[str]]) -> list[str]:
    drift = [
        f"{name}: the page pins {sorted(page.get(name, set()))}, the code {sorted(code.get(name, set()))}"
        for name in sorted(page.keys() | code.keys())
        if page.get(name) != code.get(name)
    ]
    drift += [
        f"{name} is pinned to more than one version: {sorted(v)}"
        for name, v in code.items()
        if len(v) > 1
    ]
    return drift


def test_the_pin_table_states_the_lume_pins() -> None:
    page = _pin_table(_VA_PAGE.read_text(encoding="utf-8"))

    assert page, "the LUME pin table was not found"
    assert _pin_drift(page, _code_pins()) == []


def test_a_moved_or_added_pin_is_reported() -> None:
    page = _pin_table(_VA_PAGE.read_text(encoding="utf-8"))
    code = {**_code_pins(), "lume-model": {"1.0.0"}}
    code["lume-base"] = {"9.9.9"}

    assert _pin_drift(page, code) == [
        f"lume-base: the page pins {sorted(page['lume-base'])}, the code ['9.9.9']",
        "lume-model: the page pins [], the code ['1.0.0']",
    ]


# --- Citation ----------------------------------------------------------------


def _bibtex(text: str) -> tuple[str, dict[str, str]]:
    """The landing page's bibtex entry: its key and its fields."""
    start = text.index(".. code-block:: bibtex")
    entry = text[start : text.index("\n      }\n", start)]
    key = re.search(r"@article\{(?P<key>[^,]+),", entry)
    assert key is not None, "no @article entry in the bibtex block"
    return key["key"], {m["field"]: m["value"] for m in _BIBTEX_FIELD.finditer(entry)}


def _cff_fields(cff: dict[str, Any]) -> dict[str, str]:
    """The bibtex fields ``CITATION.cff``'s ``preferred-citation`` implies."""
    cited = cff.get("preferred-citation", cff)
    return {
        "author": " and ".join(
            f"{a['family-names']}, {a['given-names']}" for a in cited["authors"]
        ),
        "title": str(cited["title"]),
        "journal": str(cited["journal"]),
        "volume": str(cited["volume"]),
        "number": str(cited["issue"]),
        "pages": str(cited["start"]),
        "year": str(cited["year"]),
        "month": f"{int(cited['month']):02d}",
        "doi": str(cited["doi"]),
        "url": str(cited["url"]),
    }


def _citation_drift(key: str, fields: dict[str, str], cff: dict[str, str]) -> list[str]:
    drift = [f"the entry key {key!r} is not the DOI {cff['doi']!r}"] if key != cff["doi"] else []
    drift += [
        f"{field}: the page says {fields.get(field)!r}, CITATION.cff {cff.get(field)!r}"
        for field in sorted(fields.keys() | cff.keys())
        if fields.get(field) != cff.get(field)
    ]
    return drift


def test_the_landing_page_cites_what_citation_cff_cites() -> None:
    text = _INDEX_PAGE.read_text(encoding="utf-8")
    key, fields = _bibtex(text)
    cff = _cff_fields(yaml.safe_load(_CITATION.read_text(encoding="utf-8")))

    assert _citation_drift(key, fields, cff) == []
    assert f"<{cff['url']}>" in text, "the citation dropdown's paper link is not the DOI URL"


def test_a_citation_that_drifts_is_reported() -> None:
    key, fields = _bibtex(_INDEX_PAGE.read_text(encoding="utf-8"))
    cff = _cff_fields(yaml.safe_load(_CITATION.read_text(encoding="utf-8")))
    planted = {**fields, "pages": "016104"}
    del planted["volume"]

    assert _citation_drift(key, planted, cff) == [
        "pages: the page says '016104', CITATION.cff '016103'",
        "volume: the page says None, CITATION.cff '4'",
    ]
