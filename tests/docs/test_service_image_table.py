"""The service-image table states what the compose templates render.

Every ``image:`` line in a shipped service template resolves through an
``OSPREY_<SVC>_IMAGE`` variable, and its packaged default is either an upstream
pin spelled in the template or an OSPREY-built name from
:func:`~osprey.deployment.compose_generator.resolve_image_defaults`. The
configuration reference lists those images one row each, and the mirroring
guide shows a mirror for every upstream pin. This module holds both pages to
the templates, so a service added, renamed or re-pinned fails here instead of
leaving a page a row behind; and neither page spells how many images there
are, because a count is the one fact the table already states.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from osprey.deployment.compose_generator import _OSPREY_IMAGE_SUFFIXES

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES = _REPO_ROOT / "src" / "osprey" / "templates" / "services"
_REFERENCE = _REPO_ROOT / "docs" / "source" / "reference" / "configuration" / "config.rst"
_MIRROR_GUIDE = (
    _REPO_ROOT / "docs" / "source" / "how-to" / "deploy-project" / "compose-templates.rst"
)

_IMAGE_LINE = re.compile(
    r"^\s*image: \$\{(?P<env>OSPREY_\w+_IMAGE):-\{\{ (?P<expr>.*) \}\}\}\s*$", re.MULTILINE
)
_BUILT = re.compile(r"default\(osprey_images\.(?P<key>\w+)\)")
_PIN = re.compile(r"default\('(?P<ref>[^']+)'")
_ROW = re.compile(
    r"^   \* - (?P<service>.+)\n"
    r"     - ``(?P<env>OSPREY_\w+_IMAGE)``\n"
    r"     - ``(?P<key>[\w.]+)``\n"
    r"     - (?P<default>.+)$",
    re.MULTILINE,
)
_COUNT_WORDS = (
    "two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    "fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
)
#: A number word up to three words ahead of "images" or "pins".
_SPELLED_COUNT = re.compile(
    rf"\b(?:{_COUNT_WORDS})(?:[\s*]+[\w-]+){{0,3}}?[\s*]+(?:images|pins)\b", re.IGNORECASE
)


def _template_defaults() -> dict[str, set[tuple[str, str]]]:
    """Each override variable's packaged defaults: ``("built", key)`` or ``("pin", ref)``."""
    found: dict[str, set[tuple[str, str]]] = {}
    for path in sorted(_TEMPLATES.glob("*/docker-compose.yml.j2")):
        for match in _IMAGE_LINE.finditer(path.read_text(encoding="utf-8")):
            if built := _BUILT.search(match["expr"]):
                default = ("built", built["key"])
            else:
                pin = _PIN.search(match["expr"])
                assert pin is not None, f"{path.name}: no packaged default in {match[0]!r}"
                default = ("pin", pin["ref"])
            found.setdefault(match["env"], set()).add(default)
    return found


def _table_rows() -> list[dict[str, str]]:
    """The rows of the configuration reference's service-image table."""
    text = _REFERENCE.read_text(encoding="utf-8")
    start = text.index(".. list-table::", text.index(".. _deployment-image-overrides:\n"))
    lines = text[start:].splitlines()[1:]
    end = next(i for i, line in enumerate(lines) if line and not line.startswith(" "))
    return [m.groupdict() for m in _ROW.finditer("\n".join(lines[:end]) + "\n")]


def _mirror_example() -> dict:
    """The ``config.yml`` the mirroring guide shows for one mirror."""
    text = _MIRROR_GUIDE.read_text(encoding="utf-8")
    marker = text.index("# config.yml — all four channels, one mirror")
    block = text[text.rindex(".. code-block:: yaml", 0, marker) :].splitlines()[2:]
    end = next(i for i, line in enumerate(block) if line and not line.startswith("   "))
    return yaml.safe_load("\n".join(line[3:] for line in block[:end]))


def test_every_overridable_image_has_one_row() -> None:
    """The table's variables are exactly the ones the templates' image lines read."""
    rows = _table_rows()
    assert len(rows) == len({row["service"] for row in rows})
    assert {row["env"] for row in rows} == set(_template_defaults())


def test_each_rows_packaged_default_is_its_templates_fallback() -> None:
    """An upstream-pin row is pinned in its template; any other row names the built image."""
    defaults = _template_defaults()
    for row in _table_rows():
        ((kind, value),) = defaults[row["env"]]
        if row["default"] == "upstream pin":
            assert kind == "pin", f"{row['service']}: the template builds {value}"
        else:
            assert kind == "built", f"{row['service']}: the template pins {value}"
            assert row["default"] == f"``<project>{_OSPREY_IMAGE_SUFFIXES[value]}``"


def test_every_osprey_built_image_is_one_a_template_renders() -> None:
    """The templates fall back to every suffix-map image and to no other built name."""
    found = _template_defaults().values()
    built = {value for defaults in found for kind, value in defaults if kind == "built"}
    assert built == set(_OSPREY_IMAGE_SUFFIXES)


def test_the_mirror_example_mirrors_every_upstream_pin() -> None:
    """A host that copies the example pulls no upstream pin from its publisher."""
    example = _mirror_example()
    for row in _table_rows():
        if row["default"] != "upstream pin":
            continue
        node = example
        for part in row["key"].split("."):
            assert isinstance(node, dict) and part in node, f"the example omits {row['key']}"
            node = node[part]


def test_neither_page_spells_an_image_count() -> None:
    """The table states how many images there are; prose that repeats it goes stale."""
    spelled = {
        page.name: hits
        for page in (_REFERENCE, _MIRROR_GUIDE)
        if (hits := _SPELLED_COUNT.findall(page.read_text(encoding="utf-8")))
    }
    assert spelled == {}
