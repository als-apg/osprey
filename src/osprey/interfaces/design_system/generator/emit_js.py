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
  manifest, each entry carrying its declared ``family``), ``DEFAULTS``
  (a per-family ``{family: {mode: id}}`` map — which theme id ``auto``
  resolves to per OS color-scheme preference, grouped by
  ``$extensions.family`` so a family sharing a mode with another family
  can never hijack its default), and ``DEFAULT_FAMILY`` (the first family
  declared in the manifest — the single fallback ``theme-manager.js``
  reads instead of re-deriving it from ``DEFAULTS``; see
  :func:`_default_family`).
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
from osprey.interfaces.design_system.generator.validate import VALID_THEME_MODES

__all__ = [
    "GENERATED_HEADER_LINES",
    "STORAGE_KEY",
    "ThemeManifestEntry",
    "ThemeFamilyDefaultsError",
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
        family: The ``{light, dark}`` pair this theme belongs to
            (``$extensions.family``), e.g. the built-in ``"osprey"``
            family. Groups themes for ``auto`` default resolution (see
            :func:`build_theme_defaults`) and, separately, selects the
            theme's required WCAG gate tuple
            (:func:`~osprey.interfaces.design_system.generator.validate.gates_for_family`).
    """

    id: str
    label: str
    mode: str
    family: str


class ThemeFamilyDefaultsError(ValueError):
    """A theme family's declared modes can't yield an unambiguous ``auto`` default.

    Raised by :func:`build_theme_defaults` when a family declares more
    than one theme for the same mode (an ambiguous default) or fewer than
    all of :data:`~osprey.interfaces.design_system.generator.validate.VALID_THEME_MODES`
    (an incomplete default). Fails closed rather than silently picking a
    winner by manifest/file order.
    """


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
        ThemeManifestEntry(
            id=metadata["id"],
            label=metadata["label"],
            mode=metadata["mode"],
            family=metadata["family"],
        )
        for metadata in tree.theme_metadata.values()
    ]


def build_theme_defaults(entries: Sequence[ThemeManifestEntry]) -> dict[str, dict[str, str]]:
    """Build the per-family ``{family: {mode: id}}`` map ``auto`` resolves against.

    Each theme declares its own ``family`` and ``mode``
    (``$extensions.family``/``$extensions.mode``); this groups manifest
    entries by family and, within each family, by mode. The result is
    "author-declared" — derived purely from what each theme file states
    about itself — rather than picked by manifest/file order, so a new
    theme file can never silently hijack another family's (or its own
    family's) ``auto`` target by sorting before it.

    Args:
        entries: The ordered manifest, as returned by
            :func:`build_theme_manifest`.

    Returns:
        A mapping from family name to a ``{mode: id}`` mapping. Contains
        only the families actually present in ``entries``.

    Raises:
        ThemeFamilyDefaultsError: If a family declares two themes for the
            same mode (an ambiguous ``auto`` target) or is missing a
            theme for one of :data:`~osprey.interfaces.design_system.generator.validate.VALID_THEME_MODES`
            (an incomplete ``auto`` target). Fails closed rather than
            silently picking a winner.
    """
    defaults: dict[str, dict[str, str]] = {}
    for entry in entries:
        family_defaults = defaults.setdefault(entry.family, {})
        if entry.mode in family_defaults:
            raise ThemeFamilyDefaultsError(
                f"theme family {entry.family!r} declares more than one theme for "
                f"mode {entry.mode!r}: {family_defaults[entry.mode]!r} and "
                f"{entry.id!r} are both candidates for its 'auto' default"
            )
        family_defaults[entry.mode] = entry.id

    for family, modes in defaults.items():
        missing = sorted(VALID_THEME_MODES - modes.keys())
        if missing:
            raise ThemeFamilyDefaultsError(
                f"theme family {family!r} is missing a theme for mode(s) "
                f"{missing!r}: 'auto' needs one theme per mode to resolve"
            )
    return defaults


def _default_family(defaults: dict[str, dict[str, str]]) -> str | None:
    """The fallback family for ``auto`` when no better signal is available.

    The first family declared in the manifest (insertion order, itself
    manifest/filename order -- never re-sorted). Shared by
    :func:`render_tokens_js` (which exports it as ``DEFAULT_FAMILY`` for
    ``theme-manager.js`` to read) and :func:`render_theme_boot_js` (which
    bakes it as a literal, for the same fallback role -- see that
    function's docstring), so the two generated files can never disagree
    on what "first family" means.

    Args:
        defaults: The per-family ``{family: {mode: id}}`` map, as returned
            by :func:`build_theme_defaults`.

    Returns:
        The first family key, or ``None`` if ``defaults`` is empty.
    """
    return next(iter(defaults), None)


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
    default_family = _default_family(defaults)

    themes_json = json.dumps(
        [{"id": e.id, "label": e.label, "mode": e.mode, "family": e.family} for e in entries],
        indent=2,
        ensure_ascii=True,
    )
    defaults_json = json.dumps(defaults, indent=2, ensure_ascii=True)
    default_family_json = json.dumps(default_family, ensure_ascii=True)

    body = (
        "// Theme registry only: no color palettes here (see module docstring\n"
        "// in generator/emit_js.py for why). Consumers read colors from\n"
        "// tokens.css via theme-manager.js's computed-style bridges.\n"
        f"export const THEMES = {themes_json};\n\n"
        f"export const DEFAULTS = {defaults_json};\n\n"
        "// The first family declared in the manifest -- the single fallback\n"
        "// theme-manager.js reads instead of re-deriving it from DEFAULTS.\n"
        f"export const DEFAULT_FAMILY = {default_family_json};"
    )
    return _render(GENERATED_HEADER_LINES, body)


def render_theme_boot_js(tree: TokenTree) -> str:
    """Render ``theme-boot.js``: the pre-paint FOUC guard.

    Resolution order, matching the design spec (finding I4): read
    ``?theme=``, then ``localStorage['osprey-theme']``, then the
    server-rendered ``<html data-theme>`` attribute, each validated
    against the baked-in id list (the query/storage rungs also accept the
    special ``'auto'`` value); the first candidate that validates wins,
    and anything left over (missing or unknown/legacy) falls all the way
    through to ``'auto'``. ``'auto'`` resolves via
    ``matchMedia('(prefers-color-scheme: dark)')`` against ``DEFAULTS``.
    ``data-theme`` is set synchronously as the script's last statement, so
    it must be loaded as a blocking, non-module, non-deferred ``<script>``
    first in ``<head>`` — no other script in this design system may run
    before it.

    Server-attribute contract (for whoever renders ``<html>`` server-side,
    e.g. Task 1.10's web_terminal server): the boot script reads
    ``document.documentElement.getAttribute("data-theme")`` — i.e. the
    ``data-theme`` attribute on the ``<html>`` element. It is treated as a
    candidate only when it is a non-null string present in the baked
    ``VALID_IDS`` list (a concrete theme id the server resolved from
    config — never the literal ``"auto"``); anything else (missing
    attribute, unknown/stale id) is ignored and resolution falls through
    to the next rung. Critically, the script never unconditionally
    overwrites this attribute: the final ``setAttribute`` call only fires
    when the resolved id differs from what was already there, so a
    correctly server-rendered attribute causes neither a flash nor a
    redundant DOM write.

    ``DEFAULTS`` is the per-family ``{family: {mode: id}}`` map (see
    :func:`build_theme_defaults`). This script has no independent notion
    of "which family is active" — it derives the family ``auto`` should
    resolve within from ``FAMILY_BY_ID``, an ``{id: family}`` map baked
    from the same manifest: when the server ``data-theme`` attribute is a
    valid concrete id, that id's family wins (so ``auto`` stays inside the
    family the server already committed to, even if a literal ``"auto"``
    from ``?theme=``/``localStorage`` ends up being the actual candidate);
    otherwise ``DEFAULT_FAMILY`` — the first family declared in the
    manifest (manifest/filename order — never re-sorted) — is the
    deterministic fallback, reached only when no server attribute is
    present/valid (e.g. before Task 1.10 wires up server rendering, or if
    it's ever omitted).

    Args:
        tree: A token tree that has already passed
            :func:`~osprey.interfaces.design_system.generator.validate.assert_valid`.

    Returns:
        The complete ``theme-boot.js`` classic-script source.
    """
    entries = build_theme_manifest(tree)
    defaults = build_theme_defaults(entries)
    valid_ids = [entry.id for entry in entries]
    family_by_id = {entry.id: entry.family for entry in entries}
    # Fallback for when the server data-theme attribute is absent/invalid;
    # see docstring above and _default_family's own docstring.
    default_family = _default_family(defaults)

    valid_ids_json = json.dumps(valid_ids, ensure_ascii=True)
    defaults_json = _indent_continuation(json.dumps(defaults, indent=2, ensure_ascii=True), "  ")
    family_by_id_json = _indent_continuation(
        json.dumps(family_by_id, indent=2, ensure_ascii=True), "  "
    )
    storage_key_json = json.dumps(STORAGE_KEY, ensure_ascii=True)
    default_family_json = json.dumps(default_family, ensure_ascii=True)

    body = f"""\
// Applies data-theme before first paint. Deliberately NOT an ES module —
// module scripts are deferred, which would let a pre-theme flash slip
// through. Duplicates THEMES/DEFAULTS identity from tokens.js as inline
// literals for the same reason: this script must not import anything.
(function () {{
  "use strict";

  const STORAGE_KEY = {storage_key_json};
  const VALID_IDS = {valid_ids_json};
  // Per-family {{mode: id}} map: DEFAULTS[family][mode]. Typed as a
  // Record (not the narrower literal shape object-literal inference would
  // give it) because resolveAuto() below indexes it with a general
  // `string` family, not just the exact DEFAULT_FAMILY literal.
  /** @type {{Record<string, {{dark?: string, light?: string}}>}} */
  const DEFAULTS = {defaults_json};
  // id -> family, so a valid server-rendered data-theme id can supply the
  // family 'auto' resolves within instead of DEFAULT_FAMILY. See the
  // render_theme_boot_js docstring in generator/emit_js.py.
  /** @type {{Record<string, string>}} */
  const FAMILY_BY_ID = {family_by_id_json};
  // Fallback family for 'auto' when no server data-theme attribute is
  // present/valid: the first family declared in the manifest.
  const DEFAULT_FAMILY = {default_family_json};

  /** @param {{string|null}} value @returns {{value is string}} */
  function isValidId(value) {{
    return value !== null && VALID_IDS.indexOf(value) !== -1;
  }}

  /** @param {{string|null}} value @returns {{value is string}} */
  function isKnownId(value) {{
    return value !== null && (value === "auto" || isValidId(value));
  }}

  /** @param {{string}} family */
  function resolveAuto(family) {{
    let prefersDark = true;
    try {{
      prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    }} catch {{
      prefersDark = true;
    }}
    const familyDefaults = DEFAULTS[family] || {{}};
    return prefersDark ? familyDefaults.dark : familyDefaults.light;
  }}

  function readQueryTheme() {{
    try {{
      return new URLSearchParams(window.location.search).get("theme");
    }} catch {{
      return null;
    }}
  }}

  function readStoredTheme() {{
    try {{
      return window.localStorage.getItem(STORAGE_KEY);
    }} catch {{
      return null;
    }}
  }}

  // The server-rendered rung (finding I4): whatever data-theme already
  // sits on <html> when this script runs, e.g. stamped by the web server
  // from config (Task 1.10). Read once so both the resolution candidate
  // below and the no-clobber check at the end use the exact same value.
  function readServerTheme() {{
    try {{
      return document.documentElement.getAttribute("data-theme");
    }} catch {{
      return null;
    }}
  }}

  const queryTheme = readQueryTheme();
  const storedTheme = readStoredTheme();
  const serverTheme = readServerTheme();
  // auto's family: the valid server theme's declared family wins over
  // DEFAULT_FAMILY, even if the final candidate below turns out to be a
  // literal "auto" from ?theme=/localStorage rather than serverTheme
  // itself — see docstring. (isValidId is called inline, not via a
  // stored boolean, so its type-predicate narrows serverTheme for the
  // FAMILY_BY_ID lookup.)
  const familyForAuto = isValidId(serverTheme) ? FAMILY_BY_ID[serverTheme] : DEFAULT_FAMILY;

  let candidate = "auto";
  if (isKnownId(queryTheme)) {{
    candidate = queryTheme;
  }} else if (isKnownId(storedTheme)) {{
    candidate = storedTheme;
  }} else if (isValidId(serverTheme)) {{
    candidate = serverTheme;
  }}

  let resolved = candidate === "auto" ? resolveAuto(familyForAuto) : candidate;
  if (!resolved && VALID_IDS.length > 0) {{
    resolved = VALID_IDS[0];
  }}
  // No-clobber: only touch the DOM when the resolved id actually differs
  // from what's already there, so a correct server-rendered attribute
  // causes neither a flash nor a redundant write.
  if (resolved && resolved !== serverTheme) {{
    document.documentElement.setAttribute("data-theme", resolved);
  }}
}})();\
"""
    # theme-boot.js is generated JS. It is type-checked under checkJs like the rest
    # of the fleet; the emitted isKnownId type predicate makes it strict-clean, so it
    # carries @ts-check rather than an opt-out.
    header_lines = (
        "// @ts-check",
        *GENERATED_HEADER_LINES,
    )
    return _render(header_lines, body)
