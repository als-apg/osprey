"""Fail-closed static validation for a **panel** directory.

A *panel* is a directory bundling an HTML entry point plus a
``manifest.json`` — the self-contained mini-app the web terminal mounts
alongside the chat surface. Where ``manifest.py`` validates only the JSON
descriptor's *schema*, this module validates the whole **directory**: it
runs the cheap, decidable static checks that decide whether a panel is a
well-formed, theme-aware surface before anything tries to serve it.

The checks are deliberately limited to what is decidable from the source
alone (this is why they are cheap and never flake):

1. **Manifest present + schema-valid.** ``manifest.json`` must exist in
   the panel dir and pass ``manifest.py``'s :func:`validate_manifest`;
   every :class:`~.manifest.ManifestError` is folded into this module's
   own error type. The manifest's declared ``entry`` file must also
   actually exist on disk (a filesystem fact the schema validator, being
   pure, cannot check).
2. **Design-system linked.** The entry HTML must opt into shared theming
   the way every standalone surface does — by referencing the pre-paint
   boot script *and* the token stylesheet: a
   ``<script src="/design-system/js/theme-boot.js">`` and a
   ``<link ... href="/design-system/css/tokens.css">`` (see the reference
   head in ``osprey.interfaces.artifacts.app``). Missing either is an
   error.
3. **Token-only (no raw hex colors).** The panel's HTML/CSS/JS must style
   through ``var(--…)`` design tokens, never raw ``#rgb``/``#rrggbb``
   color literals. Each hex-color-shaped literal is flagged with its file
   and line (see :data:`_HEX_COLOR_PATTERN` for exactly what counts as a
   color shape, and the false-positive caveat documented there).

This module deliberately does **not** attempt to prove a panel honors the
runtime ``?theme=`` query parameter — that is undecidable statically and
belongs to the separate runtime/browser check. Everything here is
decidable, and every check *fails closed*: :func:`assert_valid_panel`
raises a bundled :class:`PanelValidationError` if *any* check fails; no
check is weakened to let a panel pass.

The idiom mirrors the token validator (``generator/validate.py``) and the
sibling manifest validator (``panels/manifest.py``): a :class:`StrEnum` of
machine-readable rule ids, a frozen :class:`PanelError` rendering
``"{source}: {message}"``, a :class:`PanelValidationError` bundling
*every* failure, and a :func:`validate_panel` that runs every check
without short-circuiting so a caller sees the complete set in one pass.

Stdlib-only (``json``, ``re``, ``dataclasses``, ``enum``, ``pathlib``,
``typing``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from osprey.interfaces.design_system.panels.manifest import (
    ManifestError,
    PanelManifestError,
    parse_manifest,
)

__all__ = [
    "MANIFEST_FILENAME",
    "PanelRule",
    "PanelError",
    "PanelValidationError",
    "validate_panel",
    "assert_valid_panel",
]

#: The manifest file every panel directory must contain.
MANIFEST_FILENAME = "manifest.json"

#: The two shared design-system references a themed surface must load: the
#: pre-paint boot script (resolves the theme before first paint, avoiding a
#: flash of the wrong theme) and the token stylesheet (the ``var(--…)``
#: custom properties everything styles against). Matched on the load-bearing
#: ``src``/``href`` so attribute order and quote style don't matter; the
#: ``[^>]*`` guards keep each match inside a single tag.
_THEME_BOOT_SCRIPT_PATTERN = re.compile(
    r"<script\b[^>]*\bsrc\s*=\s*['\"]/design-system/js/theme-boot\.js['\"][^>]*>",
    re.IGNORECASE,
)
_TOKENS_CSS_LINK_PATTERN = re.compile(
    r"<link\b[^>]*\bhref\s*=\s*['\"]/design-system/css/tokens\.css['\"][^>]*>",
    re.IGNORECASE,
)
_FONTS_CSS_LINK_PATTERN = re.compile(
    r"<link\b[^>]*\bhref\s*=\s*['\"]/static/fonts/fonts\.css['\"][^>]*>",
    re.IGNORECASE,
)

#: A raw hex *color* literal: ``#`` followed by exactly 3, 4, 6, or 8 hex
#: digits (the CSS ``#rgb``/``#rgba``/``#rrggbb``/``#rrggbbaa`` shapes),
#: mirroring ``generator/validate.py``'s ``_SHORT_OR_FULL_HEX_PATTERN`` but
#: adapted from a whole-value ``^...$`` match to an embedded scan of raw
#: HTML/CSS/JS text:
#:
#: - The alternation is ordered longest-first and the trailing
#:   ``(?![0-9a-fA-F])`` forbids a following hex digit, so a wrong-length
#:   run like ``#12345`` (5 digits) matches nothing rather than being
#:   mis-read as ``#1234``/``#123`` — exactly the lengths ``validate.py``
#:   accepts as colors, and no others.
#: - The leading ``(?<![&\w])`` skips a ``#`` that is part of a word (so a
#:   ``#`` embedded in an identifier is ignored) and, crucially, a ``#``
#:   preceded by ``&`` — i.e. an HTML numeric character reference such as
#:   ``&#160;`` (nbsp) or ``&#8212;`` (em dash), whose digits are otherwise
#:   hex-shaped and would false-positive.
#:
#: Residual false-positive caveat: because this is a raw text scan (not the
#: type-scoped check ``validate.py`` can do over ``$type: "color"`` token
#: values), a URL *fragment* whose name happens to be all hex digits and
#: exactly 3/4/6/8 long — e.g. ``href="#abc"`` or ``href="#deadbeef"`` —
#: will be flagged as a color. Fragments with any non-hex character
#: (``href="#section"``, ``href="#top"``) are correctly ignored. This is an
#: accepted, fail-closed trade-off: the fix is to rename such a fragment (or
#: the author confirms it is genuinely not a color), never to loosen the
#: token-only rule.
_HEX_COLOR_PATTERN = re.compile(
    r"(?<![&\w])#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})"
    r"(?![0-9a-fA-F])"
)

#: File suffixes scanned (alongside the entry HTML) for raw hex colors.
_HEX_SCAN_SUFFIXES = (".css", ".js")


class PanelRule(StrEnum):
    """Machine-readable identifier for each kind of panel validation failure."""

    #: The panel directory has no ``manifest.json``.
    MANIFEST_MISSING = "manifest_missing"
    #: ``manifest.json`` is not valid JSON, or fails the manifest schema
    #: (each underlying :class:`~.manifest.ManifestError` is folded in).
    MANIFEST_INVALID = "manifest_invalid"
    #: The manifest's declared ``entry`` file does not exist in the dir.
    ENTRY_MISSING = "entry_missing"
    #: The entry HTML does not reference the design-system token stylesheet.
    MISSING_DESIGN_SYSTEM_LINK = "missing_design_system_link"
    #: The entry HTML does not reference the pre-paint theme-boot script.
    MISSING_THEME_BOOT = "missing_theme_boot"
    #: The entry HTML does not reference the shared web-font stylesheet, so the
    #: panel (and everything copied from it) falls back to system fonts.
    MISSING_FONT_LINK = "missing_font_link"
    #: A raw hex color literal appears where a ``var(--…)`` token belongs.
    RAW_HEX_COLOR = "raw_hex_color"


@dataclass(frozen=True)
class PanelError:
    """A single, located panel validation failure.

    Attributes:
        rule: Which check produced this error.
        message: Human-readable description of the failure.
        source: Where the failure is, for error messages — a file path
            string, or ``"path:line"`` for a located line (raw hex colors),
            or the panel directory itself for a whole-panel failure (a
            missing manifest).
    """

    rule: PanelRule
    message: str
    source: str

    def __str__(self) -> str:
        return f"{self.source}: {self.message}"


class PanelValidationError(ValueError):
    """Raised by :func:`assert_valid_panel`, bundling every :class:`PanelError`.

    Attributes:
        errors: Every panel failure, in the order they were found.
    """

    def __init__(self, errors: Sequence[PanelError]) -> None:
        self.errors = list(errors)
        super().__init__("\n".join(str(error) for error in self.errors))


def _from_manifest_error(error: ManifestError) -> PanelError:
    """Fold a manifest-schema :class:`ManifestError` into a :class:`PanelError`.

    The manifest error already carries its own ``source`` (the manifest
    file path) and rendered message, so this preserves both under this
    module's single :attr:`PanelRule.MANIFEST_INVALID` rule.
    """
    return PanelError(
        rule=PanelRule.MANIFEST_INVALID,
        message=error.message,
        source=error.source,
    )


def _check_manifest(panel_dir: Path) -> tuple[list[PanelError], Path | None]:
    """Validate the panel's manifest and resolve its entry file path.

    Args:
        panel_dir: The panel directory.

    Returns:
        A ``(errors, entry_path)`` pair. ``errors`` holds every manifest
        failure (missing file, bad JSON, schema violations, missing entry
        file). ``entry_path`` is the resolved entry HTML path *if* the
        manifest was schema-valid and its entry file exists, else ``None``
        — signalling downstream HTML checks to skip (their target is
        unknown or absent, already reported here).
    """
    manifest_path = panel_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return (
            [
                PanelError(
                    rule=PanelRule.MANIFEST_MISSING,
                    message=f"panel is missing its {MANIFEST_FILENAME!r} descriptor",
                    source=str(panel_dir),
                )
            ],
            None,
        )

    source = str(manifest_path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (
            [
                PanelError(
                    rule=PanelRule.MANIFEST_INVALID,
                    message=f"manifest is not valid JSON: {exc}",
                    source=source,
                )
            ],
            None,
        )

    try:
        manifest = parse_manifest(data, source=source)
    except PanelManifestError as exc:
        return [_from_manifest_error(error) for error in exc.errors], None

    # Schema-valid: the entry field is a known non-empty string, so we can
    # resolve it and check the file actually exists on disk.
    entry_path = panel_dir / manifest.entry
    if not entry_path.is_file():
        return (
            [
                PanelError(
                    rule=PanelRule.ENTRY_MISSING,
                    message=(f"manifest 'entry' {manifest.entry!r} does not exist in the panel"),
                    source=source,
                )
            ],
            None,
        )
    return [], entry_path


def _check_design_system_linked(entry_path: Path, entry_html: str) -> list[PanelError]:
    """Require the entry HTML to load the token stylesheet and boot script.

    Args:
        entry_path: The entry HTML file, for error attribution.
        entry_html: Its full text.

    Returns:
        One error per missing reference (stylesheet and/or boot script).
    """
    errors: list[PanelError] = []
    if not _TOKENS_CSS_LINK_PATTERN.search(entry_html):
        errors.append(
            PanelError(
                rule=PanelRule.MISSING_DESIGN_SYSTEM_LINK,
                message=(
                    "entry HTML must link the design-system token stylesheet "
                    '(<link href="/design-system/css/tokens.css">)'
                ),
                source=str(entry_path),
            )
        )
    if not _THEME_BOOT_SCRIPT_PATTERN.search(entry_html):
        errors.append(
            PanelError(
                rule=PanelRule.MISSING_THEME_BOOT,
                message=(
                    "entry HTML must load the pre-paint theme boot script "
                    '(<script src="/design-system/js/theme-boot.js">)'
                ),
                source=str(entry_path),
            )
        )
    if not _FONTS_CSS_LINK_PATTERN.search(entry_html):
        errors.append(
            PanelError(
                rule=PanelRule.MISSING_FONT_LINK,
                message=(
                    "entry HTML must link the shared web-font stylesheet "
                    '(<link href="/static/fonts/fonts.css">) so the panel uses the '
                    "brand typeface instead of falling back to system fonts"
                ),
                source=str(entry_path),
            )
        )
    return errors


def _check_no_raw_hex_colors(path: Path, text: str) -> list[PanelError]:
    """Flag every raw hex color literal in one file's text.

    Args:
        path: The file being scanned, for error attribution.
        text: Its full text.

    Returns:
        One :class:`PanelRule.RAW_HEX_COLOR` error per hex-color-shaped
        literal, each sourced as ``"path:line"`` (1-based line number).
    """
    errors: list[PanelError] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for match in _HEX_COLOR_PATTERN.finditer(line):
            literal = match.group(0)
            errors.append(
                PanelError(
                    rule=PanelRule.RAW_HEX_COLOR,
                    message=(
                        f"raw hex color {literal!r} — panels must style through "
                        "var(--…) design tokens, not literal colors"
                    ),
                    source=f"{path}:{line_number}",
                )
            )
    return errors


def _hex_scan_files(panel_dir: Path, entry_path: Path | None) -> list[Path]:
    """Collect the files to scan for raw hex colors, in a stable order.

    The entry HTML (if resolved) plus every sibling ``.css``/``.js`` under
    the panel dir. Deduplicated and sorted so error order is deterministic.
    """
    files: set[Path] = set()
    if entry_path is not None:
        files.add(entry_path)
    for suffix in _HEX_SCAN_SUFFIXES:
        files.update(panel_dir.rglob(f"*{suffix}"))
    return sorted(files)


def validate_panel(panel_dir: str | Path) -> list[PanelError]:
    """Run every panel check and collect every failure.

    Never stops at the first failure — a caller (the panel skill, the
    reference-panel test) sees the complete set in one pass, mirroring the
    token/manifest validators' contract.

    Args:
        panel_dir: The panel directory: an HTML entry point plus a
            ``manifest.json``.

    Returns:
        Every :class:`PanelError` found, in check order. Empty if the panel
        is fully valid. When the manifest is missing/invalid or its entry
        file is absent, the HTML-dependent checks (design-system link,
        theme boot) are skipped for the unknown entry — but sibling
        ``.css``/``.js`` files are still scanned for raw hex colors, since
        that check does not depend on the manifest.
    """
    panel_dir = Path(panel_dir)
    errors: list[PanelError] = []

    manifest_errors, entry_path = _check_manifest(panel_dir)
    errors.extend(manifest_errors)

    if entry_path is not None:
        entry_html = entry_path.read_text(encoding="utf-8")
        errors.extend(_check_design_system_linked(entry_path, entry_html))

    for path in _hex_scan_files(panel_dir, entry_path):
        errors.extend(_check_no_raw_hex_colors(path, path.read_text(encoding="utf-8")))

    return errors


def assert_valid_panel(panel_dir: str | Path) -> None:
    """Validate a panel directory and raise if anything failed.

    The fail-closed door: any check failing raises, and the raised error
    carries *every* failure, not just the first.

    Args:
        panel_dir: The panel directory.

    Raises:
        PanelValidationError: If :func:`validate_panel` found any failures.
    """
    errors = validate_panel(panel_dir)
    if errors:
        raise PanelValidationError(errors)


def _main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: validate a panel directory and report the result.

    Usage: ``python -m osprey.interfaces.design_system.panels.validator <panel_dir>``.

    Exits ``0`` when the panel passes, ``1`` after printing every failure
    (one ``"{source}: {message}"`` line per :class:`PanelError`), and ``2``
    on a usage error. Runnable so the panel-authoring skill and the build
    gates can invoke the validator directly rather than only via the
    :func:`assert_valid_panel` import form.
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(
            "usage: python -m osprey.interfaces.design_system.panels.validator"
            " <panel_dir>",
            file=sys.stderr,
        )
        return 2

    errors = validate_panel(args[0])
    if errors:
        for error in errors:
            print(str(error), file=sys.stderr)
        print(f"PANEL INVALID: {len(errors)} error(s) in {args[0]}", file=sys.stderr)
        return 1

    print(f"PANEL OK: {args[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
