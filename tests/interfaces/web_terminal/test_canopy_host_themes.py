"""Browser suite: the CANOPY host adapter resolved by a real engine, per theme.

``tests/interfaces/design_system/test_freshness.py`` reads ``canopy-host.css``
as text: it proves every contract name is declared and every token the sheet
reads exists in ``tokens.css``. It cannot prove what a browser makes of the
two sheets together. A custom property whose ``var()`` chain ends nowhere is
valid CSS and computes to nothing, a ``color-mix()`` the engine rejects does
the same, and a per-theme override that loses the cascade leaves the hovered
accent indistinguishable from the accent at rest.

So this suite loads the two sheets the way an embedded CANOPY page does: a
bare page that links ``tokens.css`` and then ``canopy-host.css`` by
``file://``, with one anchor element carrying ``data-canopy-host-root``. No
app server is involved; nothing here depends on how the sheets are served
(``test_canopy_proxy.py`` covers that). For each theme the id goes on both
``<html>`` and the anchor, as the page does it, and the anchor's computed
style is read.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from osprey.integrations.canopy import HOST_CONTRACT_NAMES
from osprey.interfaces.design_system.generator.build import DEFAULT_STATIC_DIR
from tests.interfaces.design_system.test_freshness import THEME_IDS, _token_blocks

if TYPE_CHECKING:
    from pathlib import Path

    from playwright.sync_api import Browser

pytestmark = [pytest.mark.browser, pytest.mark.slow]

_CSS_DIR = DEFAULT_STATIC_DIR / "css"

#: Link order matters: the adapter reads the tokens, and CANOPY links it last.
_SHEETS = (_CSS_DIR / "tokens.css", _CSS_DIR / "canopy-host.css")

_ANCHOR = "[data-canopy-host-root]"
_HOST_PREFIX = "--canopy-host-"

#: Puts the theme id where the embedded page puts it, then reads every
#: contract property off the anchor. The two accent values are also painted
#: onto a reference element and read back as its ``color``: custom properties
#: compute to their own token text, so ``var(--color-accent)`` spelled two
#: ways, or a ``color-mix()`` that lands on the accent, only compares as a
#: used color.
_READ_CONTRACT = """
([theme, names, prefix, anchorSelector]) => {
  const anchor = document.querySelector(anchorSelector);
  document.documentElement.setAttribute('data-theme', theme);
  anchor.setAttribute('data-theme', theme);

  const style = getComputedStyle(anchor);
  const values = {};
  for (const name of names) {
    values[name] = style.getPropertyValue(prefix + name).trim();
  }

  // A declaration the engine cannot use falls back to the inherited color.
  // Reading under two different inherited colors tells a real color, which
  // ignores both, from that fallback, which follows them.
  const reference = document.getElementById('reference');
  const paint = (name) => {
    reference.style.color = 'var(' + prefix + name + ')';
    const used = [];
    for (const inherited of ['rgb(1, 2, 3)', 'rgb(3, 2, 1)']) {
      anchor.style.color = inherited;
      used.push(getComputedStyle(reference).color);
    }
    anchor.style.color = '';
    return used;
  };
  return {
    values: values,
    accent: paint('accent'),
    accentHover: paint('accent-hover'),
  };
}
"""


def _write_probe(directory: Path) -> Path:
    """Write the bare page into ``directory`` and return its path."""
    links = "\n".join(f'<link rel="stylesheet" href="{sheet.as_uri()}">' for sheet in _SHEETS)
    probe = directory / "probe.html"
    probe.write_text(
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        "<title>CANOPY host adapter probe</title>\n"
        f"{links}\n"
        "</head>\n"
        "<body>\n"
        '<div data-canopy-host-root><span id="reference">reference</span></div>\n'
        "</body>\n"
        "</html>\n",
        encoding="utf-8",
    )
    return probe


def test_the_theme_list_is_the_eight_shipped_themes() -> None:
    """The parametrization below covers eight distinct themes, not fewer."""
    assert len(set(THEME_IDS)) == len(THEME_IDS) == 8


@pytest.mark.parametrize("theme", THEME_IDS)
def test_host_contract_resolves_on_a_bare_page(
    chromium_browser: Browser, tmp_path: Path, theme: str
) -> None:
    """Every contract name computes to a value, and hover moves the accent."""
    for sheet in _SHEETS:
        assert sheet.is_file(), f"missing stylesheet: {sheet}"
    probe = _write_probe(tmp_path)

    page = chromium_browser.new_page()
    try:
        page.goto(probe.as_uri())  # page-ready: a file:// probe, not an OSPREY page
        # A sheet that failed to load leaves every property empty, which would
        # read as 27 unresolved names; say what actually happened instead.
        loaded = page.evaluate("Array.from(document.styleSheets, (sheet) => sheet.href)")
        assert loaded == [sheet.as_uri() for sheet in _SHEETS]

        result = page.evaluate(
            _READ_CONTRACT, [theme, list(HOST_CONTRACT_NAMES), _HOST_PREFIX, _ANCHOR]
        )
    finally:
        page.close()

    assert set(result["values"]) == set(HOST_CONTRACT_NAMES)
    unresolved = sorted(name for name, value in result["values"].items() if not value)
    assert not unresolved, f"theme {theme}: {_HOST_PREFIX}{{{', '.join(unresolved)}}} resolve empty"

    accent = result["accent"]
    hover = result["accentHover"]
    assert accent[0] == accent[1], f"theme {theme}: the accent is not a usable color: {accent}"
    assert hover[0] == hover[1], f"theme {theme}: accent-hover is not a usable color: {hover}"
    assert hover[0] != accent[0], (
        f"theme {theme}: accent-hover equals the accent ({accent[0]}), so hover would not show"
    )


#: What the bar names compute to in every theme. Custom properties compute to
#: their token text, so these are the token values as ``tokens.css`` spells them.
_BAR_GEOMETRY: dict[str, str] = {
    # The pane header's height, ``--wt-pane-head-height``, in every theme.
    "bar-h": "40px",
    "bar-icon": "14px",
    "tab-min-w": "96px",
    "tab-max-w": "220px",
    "tab-underline": "2px",
    "control-h-xs": "24px",
    "weight-medium": "500",
    "z-dropdown": "100",
}


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@pytest.mark.parametrize("theme", THEME_IDS)
def test_bar_names_resolve_to_the_shared_values(
    chromium_browser: Browser, tmp_path: Path, theme: str
) -> None:
    """The bar geometry is theme-independent; the dropdown shadow is the theme's own."""
    probe = _write_probe(tmp_path)

    page = chromium_browser.new_page()
    try:
        page.goto(probe.as_uri())  # page-ready: a file:// probe, not an OSPREY page
        result = page.evaluate(
            _READ_CONTRACT, [theme, list(HOST_CONTRACT_NAMES), _HOST_PREFIX, _ANCHOR]
        )
    finally:
        page.close()

    values = result["values"]
    resolved = {name: values[name] for name in _BAR_GEOMETRY}
    assert resolved == _BAR_GEOMETRY, f"theme {theme}"

    expected_shadow = _token_blocks()[theme]["shadow-dropdown"]
    assert _normalized(values["shadow-dropdown"]) == _normalized(expected_shadow), (
        f"theme {theme}: --canopy-host-shadow-dropdown is {values['shadow-dropdown']!r}, "
        f"the theme's --shadow-dropdown is {expected_shadow!r}"
    )
