# ruff: noqa: T201 -- the panel validator command reports on stdout and stderr
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
   every :class:`~.manifest.ManifestFinding` is folded into this module's
   own error type. The manifest's declared ``entry`` file must also
   actually exist on disk (a filesystem fact the schema validator, being
   pure, cannot check).
2. **Design-system linked.** The entry HTML must opt into shared theming
   the way every standalone surface does — by referencing the pre-paint
   boot script *and* the three design-system stylesheets: a
   ``<script src="/design-system/js/theme-boot.js">`` and a
   ``<link ... href="/design-system/css/<sheet>">`` for each of
   :data:`PANEL_STYLESHEETS` (tokens, base, primitives). Each one missing
   is an error.
3. **Page-ready mark.** The entry HTML takes the late route to the page-ready
   mark: ``data-page-ready="pending"`` on its ``<html>``, the writer's tag
   ``<script type="module" src="/design-system/js/page-ready.js">`` ahead of its
   other module and deferred scripts, and a script it loads calling
   ``markPageReady()``, as ``panels/page_ready.py`` reads it.
4. **Token-only (no raw hex colors).** The panel's HTML/CSS/JS must style
   through ``var(--…)`` design tokens, never raw ``#rgb``/``#rrggbb``
   color literals. Third-party builds under the panel's ``vendor/``
   directory are not panel source and are skipped. Each hex-color-shaped literal is flagged with its file
   and line (see :data:`_HEX_COLOR_PATTERN` for exactly what counts as a
   color shape, and the false-positive caveat documented there).
5. **Classes have a rule (report-only).** Every class the panel's own HTML or
   JS sets has a rule in a ``<style>`` block of its HTML, one of its own
   ``.css`` files, or a design-system sheet the entry page links.

Each rule states its :class:`PanelConsequence`: a finding of a rule that
refuses stops the panel, and a finding of a rule that reports is printed or
logged and stops nothing. Checks 1-4 refuse; check 5 reports.

This module deliberately does **not** attempt to prove a panel honors the
runtime ``?theme=`` query parameter — that is undecidable statically and
belongs to the separate runtime/browser check. Everything here is
decidable, and every check whose rule refuses *fails closed*:
:func:`assert_valid_panel` raises a bundled :class:`PanelValidationError`
if *any* of them fails; no check is weakened to let a panel pass.

The idiom is the design system's shared fail-closed one
(``design_system/errors.py``, also used by ``generator/validate.py`` and
the sibling ``panels/manifest.py``): a :class:`StrEnum` of
machine-readable rule ids, a frozen :class:`PanelFinding` record rendering
``"{source}: {message}"`` (a finding, not a throwable — see that module's
naming rule), a :class:`PanelValidationError` bundling *every* finding,
and a :func:`validate_panel` that runs every check
without short-circuiting so a caller sees the complete set in one pass.
:func:`validate_entry` is the same rule set minus the manifest, for a page
that has none (an application's panel).

Stdlib-only (``json``, ``re``, ``dataclasses``, ``enum``, ``pathlib``,
``typing``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from osprey.interfaces.design_system.errors import BundledValidationError, SourcedFinding
from osprey.interfaces.design_system.panels.manifest import (
    ManifestFinding,
    PanelManifestError,
    parse_manifest,
)
from osprey.interfaces.design_system.panels.page_ready import DESIGN_SYSTEM_STATIC, route_problem

__all__ = [
    "MANIFEST_FILENAME",
    "PANEL_STYLESHEETS",
    "missing_stylesheets",
    "PanelConsequence",
    "PanelRule",
    "PanelFinding",
    "PanelValidationError",
    "refusing",
    "validate_entry",
    "validate_panel",
    "assert_valid_panel",
]

#: The manifest file every panel directory must contain.
MANIFEST_FILENAME = "manifest.json"

#: The design-system stylesheets every panel's entry HTML links, as the framing
#: serves them: the tokens (the ``var(--…)`` custom properties everything styles
#: against), the base element styles, and the shared primitives. ``osprey app
#: check`` holds an application's panel to the same list.
PANEL_STYLESHEETS = tuple(
    f"/design-system/css/{sheet}" for sheet in ("tokens.css", "base.css", "primitives.css")
)

#: The shared design-system references a themed surface must load: the
#: pre-paint boot script (resolves the theme before first paint, avoiding a
#: flash of the wrong theme) and each of :data:`PANEL_STYLESHEETS`. Matched on
#: the load-bearing ``src``/``href`` so attribute order and quote style don't
#: matter; the ``[^>]*`` guards keep each match inside a single tag.
_THEME_BOOT_SCRIPT_PATTERN = re.compile(
    r"<script\b[^>]*\bsrc\s*=\s*['\"]/design-system/js/theme-boot\.js['\"][^>]*>",
    re.IGNORECASE,
)
_STYLESHEET_LINK_PATTERNS = {
    sheet: re.compile(
        rf"<link\b[^>]*\bhref\s*=\s*['\"]{re.escape(sheet)}['\"][^>]*>", re.IGNORECASE
    )
    for sheet in PANEL_STYLESHEETS
}


def missing_stylesheets(html: str) -> list[str]:
    """Each of :data:`PANEL_STYLESHEETS` that *html* does not link, in that order."""
    return [
        sheet for sheet, pattern in _STYLESHEET_LINK_PATTERNS.items() if not pattern.search(html)
    ]


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
#: File suffixes whose markup or script sets classes.
_CLASS_SETTING_SUFFIXES = (".html", ".js", ".mjs")
_JS_SUFFIXES = (".js", ".mjs")

#: A directory segment (relative to the panel dir) whose files are third-party
#: builds fetched by ``osprey vendor fetch`` rather than panel source; they
#: carry their own colour literals and are not scanned.
_VENDOR_SEGMENT = "vendor"


class PanelConsequence(StrEnum):
    """What a finding of a rule does to its panel."""

    #: A finding of this rule stops the panel: discovery does not serve it,
    #: :func:`assert_valid_panel` raises, a check fails.
    REFUSE = "refuse"
    #: A finding of this rule is printed or logged and stops nothing.
    REPORT = "report"


class PanelRule(StrEnum):
    """Machine-readable identifier for each kind of panel finding, with its consequence.

    Each member is declared as ``(id, consequence)``; the id is the member's value.
    """

    consequence: PanelConsequence

    def __new__(cls, value: str, consequence: PanelConsequence) -> PanelRule:
        member = str.__new__(cls, value)
        member._value_ = value
        member.consequence = consequence
        return member

    #: The panel directory has no ``manifest.json``.
    MANIFEST_MISSING = "manifest_missing", PanelConsequence.REFUSE
    #: ``manifest.json`` is not valid JSON, or fails the manifest schema
    #: (each underlying :class:`~.manifest.ManifestFinding` is folded in).
    MANIFEST_INVALID = "manifest_invalid", PanelConsequence.REFUSE
    #: The manifest's declared ``entry`` file does not exist in the dir.
    ENTRY_MISSING = "entry_missing", PanelConsequence.REFUSE
    #: The entry HTML does not reference one of the design-system stylesheets.
    MISSING_DESIGN_SYSTEM_LINK = "missing_design_system_link", PanelConsequence.REFUSE
    #: The entry HTML does not reference the pre-paint theme-boot script.
    MISSING_THEME_BOOT = "missing_theme_boot", PanelConsequence.REFUSE
    #: The entry HTML does not reference the shared web-font stylesheet, so the
    #: panel (and everything copied from it) falls back to system fonts.
    MISSING_FONT_LINK = "missing_font_link", PanelConsequence.REFUSE
    #: A raw hex color literal appears where a ``var(--…)`` token belongs.
    RAW_HEX_COLOR = "raw_hex_color", PanelConsequence.REFUSE
    #: The entry page never reaches the page-ready mark by the late route.
    PAGE_READY = "page_ready", PanelConsequence.REFUSE
    #: The panel's HTML or JS sets a class that no stylesheet of the panel or
    #: linked design-system sheet has a rule for.
    UNDEFINED_CLASS = "undefined_class", PanelConsequence.REPORT


@dataclass(frozen=True)
class PanelFinding(SourcedFinding):
    """A single, located panel validation failure (a record, not a throwable).

    Attributes:
        rule: Which check produced this finding.
        message: Human-readable description of the failure.
        source: Where the failure is, for error messages — a file path
            string, or ``"path:line"`` for a located line (raw hex colors),
            or the panel directory itself for a whole-panel failure (a
            missing manifest).
    """

    rule: PanelRule


class PanelValidationError(BundledValidationError[PanelFinding]):
    """Raised by :func:`assert_valid_panel`, bundling every refusing :class:`PanelFinding`.

    Attributes:
        errors: Every refusing panel finding, in the order they were found.
    """


def refusing(findings: Iterable[PanelFinding]) -> list[PanelFinding]:
    """Those of *findings* whose rule refuses the panel, in order."""
    return [f for f in findings if f.rule.consequence is PanelConsequence.REFUSE]


def _from_manifest_finding(finding: ManifestFinding) -> PanelFinding:
    """Fold a manifest-schema :class:`ManifestFinding` into a :class:`PanelFinding`.

    The manifest finding already carries its own ``source`` (the manifest
    file path) and rendered message, so this preserves both under this
    module's single :attr:`PanelRule.MANIFEST_INVALID` rule.
    """
    return PanelFinding(
        rule=PanelRule.MANIFEST_INVALID,
        message=finding.message,
        source=finding.source,
    )


def _check_manifest(panel_dir: Path) -> tuple[list[PanelFinding], Path | None]:
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
                PanelFinding(
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
                PanelFinding(
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
        return [_from_manifest_finding(error) for error in exc.errors], None

    # Schema-valid: the entry field is a known non-empty string, so we can
    # resolve it and check the file actually exists on disk.
    entry_path = panel_dir / manifest.entry
    if not entry_path.is_file():
        return (
            [
                PanelFinding(
                    rule=PanelRule.ENTRY_MISSING,
                    message=(f"manifest 'entry' {manifest.entry!r} does not exist in the panel"),
                    source=source,
                )
            ],
            None,
        )
    return [], entry_path


def _check_design_system_linked(entry_path: Path, entry_html: str) -> list[PanelFinding]:
    """Require the entry HTML to load the design-system stylesheets and boot script.

    Args:
        entry_path: The entry HTML file, for error attribution.
        entry_html: Its full text.

    Returns:
        One error per missing reference (each stylesheet and the boot script).
    """
    errors: list[PanelFinding] = [
        PanelFinding(
            rule=PanelRule.MISSING_DESIGN_SYSTEM_LINK,
            message=(
                f'entry HTML must link {sheet} (<link href="{sheet}">); a panel links '
                f"{', '.join(PANEL_STYLESHEETS)}"
            ),
            source=str(entry_path),
        )
        for sheet in missing_stylesheets(entry_html)
    ]
    if not _THEME_BOOT_SCRIPT_PATTERN.search(entry_html):
        errors.append(
            PanelFinding(
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
            PanelFinding(
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


def _check_page_ready(entry_path: Path, entry_html: str) -> list[PanelFinding]:
    """Require the entry HTML to take the late route to the page-ready mark.

    Args:
        entry_path: The entry HTML file; its relative script addresses resolve from here.
        entry_html: Its full text.

    Returns:
        One :class:`PanelRule.PAGE_READY` finding naming what the page lacks, or none.
    """
    problem = route_problem(entry_html, entry_path, panel=True)
    if problem is None:
        return []
    return [PanelFinding(rule=PanelRule.PAGE_READY, message=problem, source=str(entry_path))]


def _check_no_raw_hex_colors(path: Path, text: str) -> list[PanelFinding]:
    """Flag every raw hex color literal in one file's text.

    Args:
        path: The file being scanned, for error attribution.
        text: Its full text.

    Returns:
        One :class:`PanelRule.RAW_HEX_COLOR` error per hex-color-shaped
        literal, each sourced as ``"path:line"`` (1-based line number).
    """
    errors: list[PanelFinding] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for match in _HEX_COLOR_PATTERN.finditer(line):
            literal = match.group(0)
            errors.append(
                PanelFinding(
                    rule=PanelRule.RAW_HEX_COLOR,
                    message=(
                        f"raw hex color {literal!r} — panels must style through "
                        "var(--…) design tokens, not literal colors"
                    ),
                    source=f"{path}:{line_number}",
                )
            )
    return errors


def _own_files(panel_dir: Path, suffixes: Iterable[str]) -> set[Path]:
    """The panel's own files with one of *suffixes*: every one under *panel_dir*
    except those under a ``vendor/`` segment (see :data:`_VENDOR_SEGMENT`)."""
    return {
        path
        for suffix in suffixes
        for path in panel_dir.rglob(f"*{suffix}")
        if _VENDOR_SEGMENT not in path.relative_to(panel_dir).parts[:-1]
    }


def _hex_scan_files(panel_dir: Path, entry_path: Path | None) -> list[Path]:
    """Collect the files to scan for raw hex colors, in a stable order.

    The entry HTML (if resolved) plus the panel's own ``.css``/``.js`` files
    (:func:`_own_files`). Deduplicated and sorted so error order is deterministic.
    """
    files = _own_files(panel_dir, _HEX_SCAN_SUFFIXES)
    if entry_path is not None:
        files.add(entry_path)
    return sorted(files)


_LINK = re.compile(r"<link\b[^>]*>", re.IGNORECASE)
_STYLE_BLOCK = re.compile(r"<style\b[^>]*>(.*?)</style\s*>", re.IGNORECASE | re.DOTALL)
_HREF = re.compile(r"""\bhref\s*=\s*["']([^"']*)["']""", re.IGNORECASE)
_DESIGN_SYSTEM_CSS = re.compile(r"^/design-system/css/([\w.-]+\.css)$")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_JS_LINE_COMMENT = re.compile(r"^\s*//.*$", re.MULTILINE)
_CSS_STRING = re.compile(r"""(["'])(?:\\.|(?!\1).)*\1""")
_CSS_CLASS = re.compile(r"\.(-?[A-Za-z_][\w-]*)")
#: Where markup or JS sets a class. A class only read (``classList.contains``) or taken
#: off (``classList.remove``) styles nothing; a state that carries no style belongs in a
#: ``data-`` attribute or ``hidden``, not in a class.
_MARKUP_CLASS = re.compile(r"""\bclass\s*=\s*\\?(["'])(.*?)\\?\1""", re.DOTALL)
_JS_CLASS = (
    _MARKUP_CLASS,
    re.compile(r"""\.className\s*=\s*(["'`])(.*?)\1"""),
    re.compile(r"""\.classList\.(?:add|toggle)\(\s*(["'`])(.*?)\1"""),
    re.compile(r"""[{,]\s*["']?class["']?\s*:\s*(["'`])(.*?)\1"""),
)
_CLASS_NAME = re.compile(r"^-?[A-Za-z_][\w-]*$")


def _class_rule_text(path: Path) -> str:
    """One file's text as the class rule reads it: a byte that is not UTF-8 becomes a
    replacement character, so a rule that only reports never raises on a file's bytes."""
    return path.read_text(encoding="utf-8", errors="replace")


def _class_rule_files(panel_dir: Path, suffixes: Iterable[str]) -> set[Path]:
    """The panel's own regular files with one of *suffixes* (:func:`_own_files`)."""
    return {path for path in _own_files(panel_dir, suffixes) if path.is_file()}


def _defined_classes(entry_html: str, panel_dir: Path, html_files: Iterable[Path]) -> set[str]:
    """The classes a selector names in the panel's own CSS files, the ``<style>`` blocks
    of its own HTML, or a design-system sheet the entry page links."""
    css = [_class_rule_text(path) for path in sorted(_class_rule_files(panel_dir, (".css",)))]
    for tag in _LINK.findall(entry_html):
        href = _HREF.search(tag)
        sheet_name = _DESIGN_SYSTEM_CSS.match(href.group(1)) if href is not None else None
        if sheet_name is not None:
            sheet = DESIGN_SYSTEM_STATIC / "css" / sheet_name.group(1)
            if sheet.is_file():
                css.append(sheet.read_text(encoding="utf-8"))
    for path in html_files:
        css += _STYLE_BLOCK.findall(_class_rule_text(path))
    defined: set[str] = set()
    for text in css:
        defined.update(_CSS_CLASS.findall(_CSS_STRING.sub("", _BLOCK_COMMENT.sub("", text))))
    return defined


def _used_classes(path: Path) -> set[str]:
    """The literal classes one HTML or JS file sets."""
    text = _class_rule_text(path)
    patterns: Sequence[re.Pattern[str]] = (_MARKUP_CLASS,)
    if path.name.endswith(_JS_SUFFIXES):
        text = _JS_LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))
        patterns = _JS_CLASS
    return {
        name
        for pattern in patterns
        for match in pattern.finditer(text)
        for name in match.group(2).split()
        if _CLASS_NAME.match(name)
    }


def _check_classes_defined(
    entry_path: Path, entry_html: str, panel_dir: Path
) -> list[PanelFinding]:
    """Report each of the panel's own HTML and JS files that sets a class with no rule.

    Args:
        entry_path: The panel's entry HTML; the design-system sheets it links count.
        entry_html: Its full text.
        panel_dir: The panel directory whose own files set and define classes.

    Returns:
        One :class:`PanelRule.UNDEFINED_CLASS` finding per file, in file order.
    """
    setting = _class_rule_files(panel_dir, _CLASS_SETTING_SUFFIXES)
    setting.add(entry_path)
    html_files = sorted(path for path in setting if path.suffix == ".html" or path == entry_path)
    defined = _defined_classes(entry_html, panel_dir, html_files)
    undefined = [(path, sorted(_used_classes(path) - defined)) for path in sorted(setting)]
    return [
        PanelFinding(
            rule=PanelRule.UNDEFINED_CLASS,
            message=(
                "no rule in the panel's or the linked design-system stylesheets for "
                f"class {', '.join(names)}"
            ),
            source=str(path),
        )
        for path, names in undefined
        if names
    ]


def validate_entry(entry_path: Path, panel_dir: Path) -> list[PanelFinding]:
    """Run every rule that reads a panel's entry page and its own files, without a manifest.

    The design-system references, the page-ready mark, raw hex colours over the
    entry page and every ``.css``/``.js`` file of the panel, then a rule for every
    class the panel's own HTML and JS set. Never stops at the first failure.

    Args:
        entry_path: The panel's entry HTML file.
        panel_dir: The panel directory whose own files the file rules read.

    Returns:
        Every :class:`PanelFinding`, in check order. Empty if the page passes.
    """
    entry_html = entry_path.read_text(encoding="utf-8")
    errors = [
        *_check_design_system_linked(entry_path, entry_html),
        *_check_page_ready(entry_path, entry_html),
    ]
    for path in _hex_scan_files(panel_dir, entry_path):
        errors.extend(_check_no_raw_hex_colors(path, path.read_text(encoding="utf-8")))
    errors.extend(_check_classes_defined(entry_path, entry_html, panel_dir))
    return errors


def validate_panel(panel_dir: str | Path) -> list[PanelFinding]:
    """Run every panel check and collect every failure.

    Never stops at the first failure — a caller (the panel skill, the
    reference-panel test) sees the complete set in one pass, mirroring the
    token/manifest validators' contract.

    Args:
        panel_dir: The panel directory: an HTML entry point plus a
            ``manifest.json``.

    Returns:
        Every :class:`PanelFinding`, in check order. Empty if the panel
        is fully valid. When the manifest is missing/invalid or its entry
        file is absent, the HTML-dependent checks (design-system link,
        theme boot, page-ready mark, class rules) are skipped for the unknown entry — but sibling
        ``.css``/``.js`` files are still scanned for raw hex colors, since
        that check does not depend on the manifest.
    """
    panel_dir = Path(panel_dir)
    errors: list[PanelFinding] = []

    manifest_errors, entry_path = _check_manifest(panel_dir)
    errors.extend(manifest_errors)

    if entry_path is not None:
        errors.extend(validate_entry(entry_path, panel_dir))
        return errors

    for path in _hex_scan_files(panel_dir, None):
        errors.extend(_check_no_raw_hex_colors(path, path.read_text(encoding="utf-8")))

    return errors


def assert_valid_panel(panel_dir: str | Path) -> None:
    """Validate a panel directory and raise when a finding refuses it.

    The fail-closed door: a finding of a rule that refuses raises, and the raised
    error carries *every* refusing finding, not just the first. A report-only
    finding never raises.

    Args:
        panel_dir: The panel directory.

    Raises:
        PanelValidationError: If :func:`validate_panel` found a refusing finding.
    """
    refused = refusing(validate_panel(panel_dir))
    if refused:
        raise PanelValidationError(refused)


def _main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: validate a panel directory and report the result.

    Usage: ``python -m osprey.interfaces.design_system.panels.validator <panel_dir>``.

    Prints every finding on stderr, one ``"{source}: {message}"`` line each; a
    report-only one ends in ``[<rule>: report]``. Exits ``1`` when a finding
    refuses the panel, ``0`` otherwise, and ``2`` on a usage error. Runnable so
    the panel-authoring skill and the build gates can invoke the validator
    directly rather than only via the :func:`assert_valid_panel` import form.
    """
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(
            "usage: python -m osprey.interfaces.design_system.panels.validator <panel_dir>",
            file=sys.stderr,
        )
        return 2

    findings = validate_panel(args[0])
    for finding in findings:
        if finding.rule.consequence is PanelConsequence.REFUSE:
            print(str(finding), file=sys.stderr)
        else:
            print(
                f"{finding} [{finding.rule.value}: {finding.rule.consequence.value}]",
                file=sys.stderr,
            )
    refused = refusing(findings)
    if refused:
        print(f"PANEL INVALID: {len(refused)} error(s) in {args[0]}", file=sys.stderr)
        return 1

    print(f"PANEL OK: {args[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
