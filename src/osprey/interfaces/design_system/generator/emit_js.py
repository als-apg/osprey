"""Emit ``tokens.js`` and ``theme-boot.js`` from a validated DTCG :class:`~.model.TokenTree`.

Both outputs are theme-*registry* artifacts only: neither contains a single
color. Per the proposal's OC-2 decision (superseding an earlier draft of
the design spec's §3.3), the color-bearing exports once envisioned for
``tokens.js`` (``XTERM_PALETTES``, ``CHART_THEMES``, ``CHART_SERIES``,
``HLJS_THEMES``) do not exist here — ``theme-manager.js``'s computed-style
bridges (``xtermPalette()``/``chartTheme()``/``chartSeries()``) read
``--ansi-*``/``--chart-*`` custom properties out of ``tokens.css`` at
runtime instead, and the highlight.js stylesheet swap uses server-rendered
``data-href-dark/light`` attributes built from ``code.*`` via
``vendor_url()``. So the only thing either JS file needs from the token
tree is theme *identity* (id/label/mode) — never a resolved token value.

This module assumes ``tree`` has already passed
:func:`osprey.interfaces.design_system.generator.validate.assert_valid`:
it trusts ``tree.theme_metadata`` to have well-formed ``id``/``label``/
``mode`` strings for every theme and does not re-validate them.

Two artifacts:

- ``tokens.js`` — an ES module exporting ``THEMES`` (the ordered theme
  manifest) and ``DEFAULTS`` (which theme id ``auto`` resolves to per OS
  color-scheme preference).
- ``theme-boot.js`` — a tiny classic (non-module) script, meant to be the
  first thing loaded in every ``<head>``, that applies ``data-theme``
  synchronously before first paint (the FOUC guard). It cannot ``import``
  ``tokens.js`` (module scripts are deferred, which would let the
  pre-theme flash it exists to prevent slip through), so it duplicates
  ``THEMES``'/``DEFAULTS``' data as inline literals baked from the same
  ``tree``.

Determinism rules mirror ``emit_css.py``: ``\\n`` line endings only, no
timestamps, deterministic ordering (manifest order = ``tree.theme_metadata``
iteration order, itself the sorted-by-filename order established by
``model.py`` — never re-sorted here), exactly one trailing newline, and no
trailing whitespace on any line.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from osprey.interfaces.design_system.generator.model import TokenTree

__all__ = [
    "GENERATED_HEADER_LINES",
    "STORAGE_KEY",
    "ThemeManifestEntry",
    "build_theme_manifest",
    "build_theme_defaults",
    "render_tokens_js",
    "render_theme_boot_js",
]

#: localStorage key theme-manager.js (hub role) persists the user's choice
#: under. Duplicated here (not imported from anywhere) because it must be
#: baked into theme-boot.js as a literal; kept as a named constant so the
#: two occurrences below can't drift from each other.
STORAGE_KEY = "osprey-theme"

#: Shared do-not-edit preamble for both generated files, as ``//`` line
#: comments (both outputs are plain JS/ESM, so ``//`` works for either).
#: Wording matches ``emit_css.py``'s header verbatim (modulo comment
#: syntax) so all three generated files read as one family. No timestamp:
#: freshness is verified by content diff, not by date.
GENERATED_HEADER_LINES: tuple[str, ...] = (
    "// AUTO-GENERATED — DO NOT EDIT.",
    "// Source: src/osprey/interfaces/design_system/tokens/",
    "// Regenerate with: python -m osprey.interfaces.design_system.generator.build",
)


@dataclass(frozen=True)
class ThemeManifestEntry:
    """One theme's public identity, as exposed to runtime JS.

    Attributes:
        id: The theme's slug (``$extensions.id``), e.g. ``"dark"``.
        label: The theme's display name (``$extensions.label``).
        mode: ``"dark"`` or ``"light"`` (``$extensions.mode``).
    """

    id: str
    label: str
    mode: str


def build_theme_manifest(tree: TokenTree) -> list[ThemeManifestEntry]:
    """Build the ordered ``THEMES`` manifest from a validated tree's theme metadata.

    Order is exactly ``tree.theme_metadata``'s iteration order — the
    sorted-by-filename order ``model.py`` established when loading
    ``tokens/themes/*.json`` (e.g. ``dark`` before ``light``). Never
    re-sorted here, so a new skin's manifest position is controlled purely
    by its filename.

    Args:
        tree: A token tree that has already passed
            :func:`~osprey.interfaces.design_system.generator.validate.assert_valid`.

    Returns:
        One :class:`ThemeManifestEntry` per theme, in manifest order.
    """
    return [
        ThemeManifestEntry(id=metadata["id"], label=metadata["label"], mode=metadata["mode"])
        for metadata in tree.theme_metadata.values()
    ]


def build_theme_defaults(entries: Sequence[ThemeManifestEntry]) -> dict[str, str]:
    """Map each distinct mode to the theme id ``auto`` resolves to for it.

    The first manifest entry for a given mode wins (manifest order, so
    file order decides ties) — this is what lets a future third skin
    sharing a mode (e.g. a second dark-family theme) coexist with the
    registry without becoming an ambiguous ``auto`` target.

    Args:
        entries: The ordered manifest, as returned by
            :func:`build_theme_manifest`.

    Returns:
        A mapping from mode (``"dark"``/``"light"``) to a theme id.
        Contains only the modes actually present in ``entries`` — e.g. a
        light-only registry would yield ``{"light": ...}`` with no
        ``"dark"`` key.
    """
    defaults: dict[str, str] = {}
    for entry in entries:
        defaults.setdefault(entry.mode, entry.id)
    return defaults


def _indent_continuation(text: str, indent: str) -> str:
    """Prefix every line but the first of ``text`` with ``indent``.

    For embedding a multi-line ``json.dumps(..., indent=2)`` literal after
    an inline prefix like ``"  var DEFAULTS = "`` so its continuation
    lines land at the surrounding block's indent level instead of at
    column 0.

    Args:
        text: The (possibly multi-line) literal to embed.
        indent: The whitespace prefix to add to every line after the first.

    Returns:
        ``text`` unchanged if it has only one line; otherwise every line
        after the first gets ``indent`` prepended.
    """
    lines = text.split("\n")
    return lines[0] + "".join(f"\n{indent}{line}" for line in lines[1:])


def _render(header_lines: tuple[str, ...], *body_blocks: str) -> str:
    """Join a header and body blocks into hook-clean generated file content.

    Args:
        header_lines: Leading ``//`` comment lines.
        *body_blocks: Remaining top-level blocks, each already terminated
            without a trailing newline; blank-line-separated in the output.

    Returns:
        The full file content: ``\\n``-joined, exactly one trailing
        newline, no trailing whitespace on any line (every piece passed in
        is a plain literal with no line ever ending in a space).
    """
    parts = ["\n".join(header_lines), *body_blocks]
    return "\n\n".join(parts) + "\n"


def render_tokens_js(tree: TokenTree) -> str:
    """Render ``tokens.js``: the theme registry, nothing else.

    Args:
        tree: A token tree that has already passed
            :func:`~osprey.interfaces.design_system.generator.validate.assert_valid`.

    Returns:
        The complete ``tokens.js`` ES module source.
    """
    entries = build_theme_manifest(tree)
    defaults = build_theme_defaults(entries)

    themes_json = json.dumps(
        [{"id": e.id, "label": e.label, "mode": e.mode} for e in entries],
        indent=2,
        ensure_ascii=True,
    )
    defaults_json = json.dumps(defaults, indent=2, ensure_ascii=True)

    body = (
        "// Theme registry only: no color palettes here (see module docstring\n"
        "// in generator/emit_js.py for why). Consumers read colors from\n"
        "// tokens.css via theme-manager.js's computed-style bridges.\n"
        f"export const THEMES = {themes_json};\n\n"
        f"export const DEFAULTS = {defaults_json};"
    )
    return _render(GENERATED_HEADER_LINES, body)


def render_theme_boot_js(tree: TokenTree) -> str:
    """Render ``theme-boot.js``: the pre-paint FOUC guard.

    Resolution order, matching the design spec: read ``?theme=``, then
    ``localStorage['osprey-theme']``, each validated against the baked-in
    id list (plus the special ``'auto'`` value); the first candidate that
    validates wins, and anything left over (missing or unknown/legacy)
    falls all the way through to ``'auto'``. ``'auto'`` resolves via
    ``matchMedia('(prefers-color-scheme: dark)')`` against ``DEFAULTS``.
    ``data-theme`` is set synchronously as the script's last statement, so
    it must be loaded as a blocking, non-module, non-deferred ``<script>``
    first in ``<head>`` — no other script in this design system may run
    before it.

    Args:
        tree: A token tree that has already passed
            :func:`~osprey.interfaces.design_system.generator.validate.assert_valid`.

    Returns:
        The complete ``theme-boot.js`` classic-script source.
    """
    entries = build_theme_manifest(tree)
    defaults = build_theme_defaults(entries)
    valid_ids = [entry.id for entry in entries]

    valid_ids_json = json.dumps(valid_ids, ensure_ascii=True)
    defaults_json = _indent_continuation(json.dumps(defaults, indent=2, ensure_ascii=True), "  ")
    storage_key_json = json.dumps(STORAGE_KEY, ensure_ascii=True)

    body = f"""\
// Applies data-theme before first paint. Deliberately NOT an ES module —
// module scripts are deferred, which would let a pre-theme flash slip
// through. Duplicates THEMES/DEFAULTS identity from tokens.js as inline
// literals for the same reason: this script must not import anything.
(function () {{
  "use strict";

  var STORAGE_KEY = {storage_key_json};
  var VALID_IDS = {valid_ids_json};
  var DEFAULTS = {defaults_json};

  function isKnownId(value) {{
    return value === "auto" || VALID_IDS.indexOf(value) !== -1;
  }}

  function resolveAuto() {{
    var prefersDark = true;
    try {{
      prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    }} catch (error) {{
      prefersDark = true;
    }}
    return prefersDark ? DEFAULTS.dark : DEFAULTS.light;
  }}

  function readQueryTheme() {{
    try {{
      return new URLSearchParams(window.location.search).get("theme");
    }} catch (error) {{
      return null;
    }}
  }}

  function readStoredTheme() {{
    try {{
      return window.localStorage.getItem(STORAGE_KEY);
    }} catch (error) {{
      return null;
    }}
  }}

  var candidate = "auto";
  var queryTheme = readQueryTheme();
  var storedTheme = readStoredTheme();
  if (isKnownId(queryTheme)) {{
    candidate = queryTheme;
  }} else if (isKnownId(storedTheme)) {{
    candidate = storedTheme;
  }}

  var resolved = candidate === "auto" ? resolveAuto() : candidate;
  if (!resolved && VALID_IDS.length > 0) {{
    resolved = VALID_IDS[0];
  }}
  if (resolved) {{
    document.documentElement.setAttribute("data-theme", resolved);
  }}
}})();\
"""
    return _render(GENERATED_HEADER_LINES, body)
