"""The dashboard's JS settings schema and figure catalog track their Python owner.

Two tables the browser holds restate facts the server owns:

* ``settings.js``'s ``SETTINGS_FIELDS`` carries a min/max per field, which is
  what the number inputs are bounded by. The server clamps independently
  (``state._VALIDATION_RANGES``), so a widened browser bound does not break —
  it silently clamps, and the operator sees a value they did not ask for come
  back. A narrowed one simply makes a legal value unreachable.
* the figure catalog is written four times over — ``state.ALL_FIGURES``,
  ``app.FIGURE_BUILDERS``, ``app.js``, and the ``data-figure`` cells of
  ``index.html``. A figure in three of the four renders an empty cell, a dead
  slot, or nothing at all, with no error anywhere.

Both are read out of the JS/HTML source, so this fails the moment one side
moves alone.
"""

import json
import re
from pathlib import Path

from osprey.interfaces.lattice_dashboard.app import FIGURE_BUILDERS
from osprey.interfaces.lattice_dashboard.state import (
    _VALIDATION_RANGES,
    ALL_FIGURES,
    DEFAULT_SETTINGS,
)

_STATIC = (
    Path(__file__).parents[3] / "src" / "osprey" / "interfaces" / "lattice_dashboard" / "static"
)
_SETTINGS_JS = _STATIC / "js" / "settings.js"
_APP_JS = _STATIC / "js" / "app.js"
_INDEX_HTML = _STATIC / "index.html"

_FIELD_RE = re.compile(
    r"\{\s*key:\s*'(?P<key>\w+)'.*?min:\s*(?P<min>-?[\d.]+),\s*max:\s*(?P<max>-?[\d.]+)"
)


def _js_settings_fields() -> dict[str, dict[str, tuple[float, float]]]:
    """``{group: {key: (min, max)}}`` as ``SETTINGS_FIELDS`` declares it."""
    source = _SETTINGS_JS.read_text(encoding="utf-8")
    body = re.search(r"export const SETTINGS_FIELDS = \{(.*?)\n\};", source, re.DOTALL)
    assert body, f"SETTINGS_FIELDS object literal not found in {_SETTINGS_JS}"

    groups: dict[str, dict[str, tuple[float, float]]] = {}
    current: str | None = None
    for line in body.group(1).splitlines():
        group_start = re.match(r"\s{2}(\w+): \{", line)
        if group_start:
            current = group_start.group(1)
            groups[current] = {}
            continue
        field = _FIELD_RE.search(line)
        if field and current is not None:
            groups[current][field.group("key")] = (
                float(field.group("min")),
                float(field.group("max")),
            )
    return groups


def _js_all_figures() -> list[str]:
    source = _APP_JS.read_text(encoding="utf-8")
    names: list[str] = []
    for const in ("FAST_FIGURES", "VERIFICATION_FIGURES"):
        match = re.search(rf"const {const} = (\[[^\]]*\]);", source)
        assert match, f"{const} not found in {_APP_JS}"
        names.extend(json.loads(match.group(1).replace("'", '"')))
    return names


def test_the_js_schema_declares_the_server_validation_ranges() -> None:
    js = _js_settings_fields()
    assert {group: dict(fields) for group, fields in js.items()} == {
        group: {key: (float(lo), float(hi)) for key, (lo, hi) in fields.items()}
        for group, fields in _VALIDATION_RANGES.items()
    }


def test_the_js_schema_covers_exactly_the_settings_the_server_defaults() -> None:
    js = _js_settings_fields()
    assert js.keys() == DEFAULT_SETTINGS.keys()
    for group, defaults in DEFAULT_SETTINGS.items():
        assert js[group].keys() == defaults.keys(), group


def test_every_holder_of_the_figure_catalog_agrees() -> None:
    html_figures = re.findall(r'data-figure="([\w-]+)"', _INDEX_HTML.read_text(encoding="utf-8"))
    assert set(ALL_FIGURES) == set(FIGURE_BUILDERS)
    assert set(ALL_FIGURES) == set(_js_all_figures())
    assert set(ALL_FIGURES) == set(html_figures)
    assert len(html_figures) == len(set(html_figures)), "a figure has two cells"
