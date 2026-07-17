"""Fleet-wide hygiene scanner: hardcoded colors, ``var(--x)`` integrity, and
stray token-defining blocks.

This module is the "hygiene" leg of the test pyramid described in the
frontend-design-system PLAN (Task 1.11). It scans every ``.css``/``.js``/
``.html`` asset under ``src/osprey/interfaces/`` plus the single dispatch
dashboard file (the design system is mounted there too, see Task 4.1) for
three independent kinds of drift:

(a) Hardcoded color literals (hex, ``rgb()``/``rgba()``, ``hsl()``/
    ``hsla()``) that should have been expressed as ``var(--token)``
    references instead. This used to be a ratchet against
    ``hygiene_baseline.json`` while the fleet migration (PLAN Phase 2/3)
    was in progress; Task 4.2 (hygiene-zero-flip) deleted that baseline and
    flipped this to a strict zero-tolerance check now that every interface
    has migrated. From this commit on, every in-scope file must have zero
    non-allowlisted literals — a genuinely justified, permanent survivor
    (a print stylesheet that must stay light-on-white, a fixed categorical
    color with no fleet-wide semantic equivalent, etc.) goes on the
    commented allowlist described below instead of a token.

(b) ``var(--name)`` reference integrity: every custom-property reference
    in the same in-scope assets must resolve to either a name the token
    generator emits (the authoritative set is read straight off the
    committed ``tokens.css``) or a name defined locally within the same
    interface's own asset set — either a CSS/inline-style
    ``--name: ...`` declaration or a JS ``element.style.setProperty
    ('--name', ...)`` call. This has never been a ratchet: it must hold at
    every commit. ``_KNOWN_DANGLING_VARS`` is the allowlist for any
    pre-existing dangling reference that isn't in scope to fix immediately
    (empty as of the hygiene-zero-flip commit — every migration task that
    had an entry cleared it when it fixed the underlying reference). That
    allowlist is checked in both directions: no unexplained new dangling
    ref may appear, and no allowlisted entry may go stale.

(c) Stray token-defining blocks: no ``:root { ... }`` or
    ``[data-theme=...] { ... }`` rule outside ``design_system/static/`` may
    declare custom properties (``--name: value;``). Every interface used to
    ship its own such block (that's what the fleet migration eliminated);
    now the shared ``tokens.css`` is the only legitimate place one exists.
    A ``[data-theme=...]`` rule that only overrides ordinary CSS properties
    (e.g. disabling a dark-only glow effect in light mode) is unaffected by
    this check — it's the act of *defining a custom property* in one of
    these blocks that's disallowed, not the selector itself.

(d) Scale-literal drift: bare numeric/keyword literals for font-size,
    font-weight, line-height, the border-radius family, and
    transition/animation durations that should instead be expressed as
    ``var(--text-*``/``--weight-*``/``--leading-*``/``--radius-*``/
    ``--duration-*)`` references into the scales the token generator now
    emits (see the ``feat(design-system): emit type/spacing/radius/weight/
    leading/z/duration scales`` commit). Unlike (a)-(c) this is not
    fleet-wide: it is scoped per-interface by ``SCALE_ENFORCED_INTERFACES``,
    which is empty as of this commit (this task only adds the scanner and
    its self-tests) and gets flipped to name the migrated interfaces once
    their CSS actually moves onto the scale variables — the same
    ratchet-then-flip shape checks (a)/(c) went through, just starting from
    an empty rather than a baseline-file ratchet. Spacing (margin/padding/
    gap) and z-index are deliberately excluded from this check even for an
    enforced interface: spacing literals are far denser throughout existing
    CSS than the other five properties, and z-index frequently encodes
    legitimate intra-component micro-stacking (a dropdown one layer above
    its trigger, a tooltip above both) that a shared scale can't cleanly
    express. Both may still migrate opportunistically, and the emitted
    ``--space-*``/``--z-*`` scales remain the preferred spelling for new
    code — they're just not enforced here.

Checks (a) and (c) share the same allowlist idea in spirit — a literal,
commented exception list — but check (a)'s allowlist is *in the scanned
files themselves* (an inline marker comment, since ownership of those
files belongs to the migration tasks, not this one) while check (b)'s
lives in this module (there is nothing sensible to "comment out" in a way
that survives a source edit for a missing declaration). Check (d) follows
check (a)'s in-file convention, with its own ``hygiene-allow-scale``
marker (see below) distinct from ``hygiene-allow-color``.
"""

from __future__ import annotations

import re
from pathlib import Path

import osprey.interfaces.design_system as design_system_pkg

_INTERFACES_ROOT = Path(design_system_pkg.__file__).parents[1]
_REPO_ROOT = Path(design_system_pkg.__file__).parents[4]
_DASHBOARD_HTML = _REPO_ROOT / "src" / "osprey" / "dispatch" / "dashboard.html"
_DESIGN_SYSTEM_STATIC = _INTERFACES_ROOT / "design_system" / "static"
_TOKENS_CSS = _DESIGN_SYSTEM_STATIC / "css" / "tokens.css"

#: Generated artifacts excluded from BOTH checks — these ARE color/token
#: definitions by design, not consumers of them (see PLAN Task 1.11).
_EXCLUDED_GENERATED_FILES = frozenset(
    {
        _INTERFACES_ROOT / "design_system" / "static" / "css" / "tokens.css",
        _INTERFACES_ROOT / "design_system" / "static" / "js" / "tokens.js",
        _INTERFACES_ROOT / "design_system" / "static" / "js" / "theme-boot.js",
    }
)

#: A dispatch dashboard "interface" isn't a real subdirectory of
#: ``src/osprey/interfaces/`` — it's mounted standalone (Task 4.1) — but it
#: needs its own local-definition scope, distinct from every real interface.
_DISPATCH_GROUP = "__dispatch__"


def _in_scope_files() -> list[Path]:
    """Every asset both hygiene checks scan: see the module docstring for scope."""
    files: list[Path] = []
    for pattern in ("*.css", "*.js", "*.html"):
        for path in _INTERFACES_ROOT.rglob(pattern):
            if "vendor" in path.parts:
                continue
            if ".min." in path.name:
                continue
            if path in _EXCLUDED_GENERATED_FILES:
                continue
            files.append(path)
    files.append(_DASHBOARD_HTML)
    return sorted(files)


def _interface_group(path: Path) -> str:
    """The asset-set a file's local ``var()`` definitions/references belong to.

    Every real interface is its own group (``ariel``, ``artifacts``, ...,
    keyed by its top-level directory name under ``src/osprey/interfaces/``);
    the standalone dispatch dashboard is its own single-file group.
    """
    if path == _DASHBOARD_HTML:
        return _DISPATCH_GROUP
    return path.relative_to(_INTERFACES_ROOT).parts[0]


def _relpath(path: Path) -> str:
    """POSIX-style path relative to the repo root, as stored in the baseline JSON."""
    return path.relative_to(_REPO_ROOT).as_posix()


# --- Check (a): hardcoded-color strict zero-tolerance ------------------------------

#: Matches a hex color (#rgb/#rrggbb/#rrggbbaa, with word-boundary guards so
#: e.g. a URL fragment like "#deadbeef-section" is still counted only once,
#: not double-matched) or an rgb()/rgba()/hsl()/hsla() function call. The
#: negative lookbehind on ``#`` excludes HTML numeric character entities
#: (``&#9998;``, ``&#039;``, ...) — their digits are frequently valid hex
#: too (e.g. ``&#128203;`` contains only 0-9), but a literal ``#`` preceded
#: by ``&`` is never a CSS color.
_COLOR_RE = re.compile(
    r"(?<!&)#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})\b"
    r"|\b(?:rgba?|hsla?)\([^)]*\)"
)

#: Commented allowlist mechanism (PLAN Task 1.11 requirement) for a literal
#: that is a genuine, permanent survivor rather than unmigrated debt: a
#: print stylesheet that must stay light-on-white regardless of theme, a
#: fixed categorical color with no fleet-wide semantic equivalent (a JSON
#: syntax-highlight hue, a per-server legend color), a scanner false
#: positive (an HTML entity or issue number the regex can't distinguish
#: from a real color in context), or similar. A line containing this
#: marker is never counted; ``-start``/``-end`` variants bracket a
#: multi-line block (both boundary lines are themselves exempt).
_ALLOW_LINE_MARKER = "hygiene-allow-color"
_ALLOW_BLOCK_START_MARKER = "hygiene-allow-color-start"
_ALLOW_BLOCK_END_MARKER = "hygiene-allow-color-end"


def _count_hardcoded_colors(text: str) -> int:
    """Count non-allowlisted hardcoded-color occurrences in one file's text.

    A ``#`` percent-encoded as ``%23`` inside a ``url()``/``data:`` URI (the
    common shape for an inline-SVG icon's ``fill``/``stroke``) is decoded
    before matching, so an encoded hex color is caught exactly like a literal
    one — otherwise a themed-looking asset can smuggle a fixed color past the
    literal-``#`` regex.
    """
    count = 0
    in_allowed_block = False
    for line in text.splitlines():
        if _ALLOW_BLOCK_START_MARKER in line:
            in_allowed_block = True
            continue
        if _ALLOW_BLOCK_END_MARKER in line:
            in_allowed_block = False
            continue
        if in_allowed_block or _ALLOW_LINE_MARKER in line:
            continue
        count += len(_COLOR_RE.findall(line.replace("%23", "#")))
    return count


def test_scanner_decodes_percent_encoded_hash() -> None:
    """A ``%23``-encoded hex color in a data-URI is counted like a literal one.

    Regression guard for the scanner scope hole: an inline-SVG icon whose
    ``fill``/``stroke`` is written ``%23abcdef`` (URL-encoded ``#``) used to
    pass the literal-``#`` regex untouched.
    """
    encoded = "background-image: url(\"data:image/svg+xml,...fill='%2394a3b8'...\");"
    assert _count_hardcoded_colors(encoded) == 1
    # The inline allow-marker still suppresses it, same as a literal color.
    assert _count_hardcoded_colors(encoded + " /* hygiene-allow-color: x */") == 0


def test_hardcoded_color_zero_tolerance() -> None:
    """No in-scope file may contain a non-allowlisted hardcoded-color literal.

    This was a ratchet against ``hygiene_baseline.json`` during the fleet
    migration (PLAN Phase 2/3); Task 4.2 (hygiene-zero-flip) deleted that
    baseline once every interface finished migrating. A literal that's a
    deliberate, permanent exception (not migration debt) belongs on the
    inline ``hygiene-allow-color`` allowlist instead of a token — see the
    module docstring and the marker's own docstring above.
    """
    offenders = []
    for path in _in_scope_files():
        n = _count_hardcoded_colors(path.read_text(encoding="utf-8"))
        if n:
            offenders.append(f"{_relpath(path)}: {n} hardcoded color(s)")
    assert not offenders, (
        "Hardcoded color literal(s) found — use a design token instead, or if "
        "this is a deliberate, permanent exception (print stylesheet, fixed "
        "categorical color, scanner false positive), mark it with a trailing "
        "`/* hygiene-allow-color: <reason> */` comment (or a "
        "hygiene-allow-color-start/-end block for a multi-line span):\n" + "\n".join(offenders)
    )


# --- Check (b): var(--x) integrity --------------------------------------------------

_VAR_DECLARATION_RE = re.compile(r"--([a-zA-Z0-9-]+)\s*:")
_VAR_CALL_START_RE = re.compile(r"var\(")
_SET_PROPERTY_RE = re.compile(r"\.setProperty\(\s*['\"]--([a-zA-Z0-9-]+)['\"]")

#: Known pre-existing dangling ``var()`` references at the commit this test
#: was authored against, as ``(relative_path, var_name)`` pairs. Each is a
#: real bug (the referenced custom property is never defined anywhere in
#: its interface, either as a CSS declaration or a JS
#: ``element.style.setProperty(...)`` call, AND the reference itself
#: provides no literal fallback — so the CSS property becomes invalid, not
#: just differently colored). None of these are in Task 1.11's scope to
#: fix; each is owned by the migration task noted below and MUST be
#: removed from this set in the same commit that fixes it — a stale entry
#: fails the "no longer dangling" half of the assertion below.
_KNOWN_DANGLING_VARS: frozenset[tuple[str, str]] = frozenset(
    {
        # Not a real CSS/JS reference at all: osprey-theme-switcher.js's module
        # docstring illustrates the general `var(--name)` syntax with a literal
        # ellipsis placeholder, "var(--…)" -- the scanner's balanced-paren
        # extraction faithfully (and here, spuriously) parses that prose as a
        # call for a property literally named "…". There is no such property,
        # declared or otherwise, because there is no such call; this is a false
        # positive on documentation text, not a dangling style reference (the
        # docstring was authored by rewrite-switcher-family-picker, Task 1.9).
        (
            "src/osprey/interfaces/design_system/static/js/components/osprey-theme-switcher.js",
            "…",
        ),
    }
)


def _declared_names(text: str) -> set[str]:
    """Every ``--name`` a CSS custom-property declaration (incl. inline
    ``style="--name: ..."`` attributes) defines anywhere in ``text``."""
    return set(_VAR_DECLARATION_RE.findall(text))


def _js_set_property_names(text: str) -> set[str]:
    """Every ``--name`` a JS ``element.style.setProperty('--name', ...)`` call
    defines anywhere in ``text`` — a legitimate runtime-only local definition
    that a pure textual ``--name:`` declaration scan would miss."""
    return set(_SET_PROPERTY_RE.findall(text))


def _extract_var_calls(text: str) -> list[tuple[str, bool]]:
    """Every ``var(...)`` call in ``text`` — including ones nested inside
    another call's fallback argument — as ``(name, has_fallback)`` pairs.

    Nested calls are discovered independently: scanning for every literal
    ``var(`` substring (not just top-level ones) means a nested call like
    the ``--surface-raised`` one inside ``var(--surface-subtle,
    var(--surface-raised))`` is extracted as a call in its own right, with
    its own name and fallback-presence — so a broken nested fallback is
    still caught, just attributed to the inner name (the actual root
    cause) rather than the outer one.
    """
    calls: list[tuple[str, bool]] = []
    for match in _VAR_CALL_START_RE.finditer(text):
        depth = 1
        i = match.end()
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        content = text[match.end() : i - 1]

        comma_depth = 0
        comma_index = None
        for j, ch in enumerate(content):
            if ch == "(":
                comma_depth += 1
            elif ch == ")":
                comma_depth -= 1
            elif ch == "," and comma_depth == 0:
                comma_index = j
                break

        if comma_index is None:
            name_part, has_fallback = content, False
        else:
            name_part = content[:comma_index]
            has_fallback = content[comma_index + 1 :].strip() != ""

        name_part = name_part.strip()
        if name_part.startswith("--") and len(name_part) > 2:
            calls.append((name_part[2:], has_fallback))
    return calls


def test_var_integrity() -> None:
    """Every ``var(--name)`` reference resolves to a real declaration.

    "Resolves" means: ``name`` is either a generator-emitted token (read
    from the committed ``tokens.css``), a name declared locally within the
    same interface's own asset set (CSS declaration or JS
    ``setProperty``), or the reference itself carries a fallback value
    (in which case a broken *nested* fallback, if any, is still caught
    independently — see :func:`_extract_var_calls`).

    Unlike the hardcoded-color check, this one is not a ratchet: any
    dangling reference must either not exist, or be explicitly listed
    (with a rationale and an owning task) in ``_KNOWN_DANGLING_VARS``.
    """
    files = _in_scope_files()
    emitted_names = _declared_names(_TOKENS_CSS.read_text(encoding="utf-8"))

    texts: dict[Path, str] = {path: path.read_text(encoding="utf-8") for path in files}

    local_names_by_group: dict[str, set[str]] = {}
    for path, text in texts.items():
        group = _interface_group(path)
        names = local_names_by_group.setdefault(group, set())
        names |= _declared_names(text)
        names |= _js_set_property_names(text)

    dangling: set[tuple[str, str]] = set()
    for path, text in texts.items():
        group = _interface_group(path)
        valid_names = emitted_names | local_names_by_group.get(group, set())
        for name, has_fallback in _extract_var_calls(text):
            if name in valid_names or has_fallback:
                continue
            dangling.add((_relpath(path), name))

    unexplained = sorted(dangling - _KNOWN_DANGLING_VARS)
    assert not unexplained, (
        "New dangling var() reference(s) found (no generator-emitted name, no local "
        "declaration/setProperty within the interface, and no fallback value) — "
        "either fix the reference or, if it's a pre-existing bug out of scope for "
        "this change, add it to _KNOWN_DANGLING_VARS with a rationale and owning "
        f"task: {unexplained}"
    )

    stale_allowlist = sorted(_KNOWN_DANGLING_VARS - dangling)
    assert not stale_allowlist, (
        "_KNOWN_DANGLING_VARS entries that are no longer dangling — the fix landed, "
        f"so remove these entries: {stale_allowlist}"
    )


# --- Check (c): no token-DUPLICATING blocks outside design_system/static/ ---------

#: A `:root { ... }` or `[data-theme=...] { ... }` rule header, capturing its
#: body up to the first unnested `}` — these blocks never legitimately nest
#: further rules, only property declarations, so a non-greedy match to the
#: first `}` is safe.
_TOKEN_BLOCK_RE = re.compile(r"(:root|\[data-theme=[^\]]*\])[^{}]*\{([^{}]*)\}")


def test_no_token_defining_blocks_outside_design_system() -> None:
    """No interface may re-declare a canonical token or theme-switch a color.

    Every interface used to ship a `:root {}` PLUS a `[data-theme=...] {}`
    pair shadowing fleet-wide names (`--bg-primary`, `--text-primary`, ...)
    with its own per-theme hardcoded values — that duplication (and the
    resulting light/dark drift it caused) is exactly what the fleet
    migration eliminated; `design_system/static/css/tokens.css` is now the
    only legitimate place those names are declared. Concretely, two
    patterns are still disallowed here:

    - A `[data-theme=...] {}` block declaring ANY custom property. Every
      interface now expresses theme-varying local values as a `color-mix()`
      composite of a canonical bg/accent/etc. token (see e.g. lattice
      dashboard's `--surface-card`) instead of a per-theme override block,
      so no legitimate reason for one remains.
    - A plain `:root {}` declaring a name `tokens.css` ALSO emits — that's
      a shadow of the canonical cascade, which is what silently kept a
      migrated interface dark-only in a previous incarnation of this bug.

    A plain `:root {}` declaring names with NO canonical equivalent (a
    spacing/radius/transition scale, a genuinely local one-off extension
    color like `--verify-accent`) is exactly the sanctioned pattern for
    "no fleet-wide equivalent" tokens described throughout the migration
    and is NOT flagged. Likewise a `[data-theme=...] {}` rule that only
    overrides ordinary CSS properties (no custom-property declarations in
    its body) — e.g. disabling a dark-only glow effect in light mode — is
    unaffected; see the module docstring.
    """
    emitted_names = _declared_names(_TOKENS_CSS.read_text(encoding="utf-8"))

    offenders = []
    for path in _in_scope_files():
        if _DESIGN_SYSTEM_STATIC in path.parents:
            continue
        text = path.read_text(encoding="utf-8")
        for selector, body in _TOKEN_BLOCK_RE.findall(text):
            declared = _declared_names(body)
            if not declared:
                continue
            if selector.startswith("[data-theme") or (declared & emitted_names):
                offenders.append(_relpath(path))
                break
    assert not offenders, (
        "Local :root {}/[data-theme=...] {} block(s) either theme-switching a "
        "custom property or shadowing a canonical tokens.css name found "
        "outside design_system/static/ — colors must come from the shared "
        "tokens.css, with theme-varying local extensions expressed as a "
        "color-mix() composite instead of a per-theme override block: "
        + ", ".join(sorted(offenders))
    )


# --- Check (d): scale-literal ratchet (SCALE_ENFORCED_INTERFACES starts empty) -----

#: Commented allowlist mechanism, mirroring `hygiene-allow-color` above: a
#: genuine, deliberate exception (a literal that predates the scale and is
#: out of scope for the interface's current migration step, or a scanner
#: false positive) goes on this marker instead of a token. A line
#: containing this marker is never counted; `-start`/`-end` variants
#: bracket a multi-line span (both boundary lines are themselves exempt) —
#: same semantics as the color check's markers, just a distinct name so a
#: color exception and a scale exception can't be mistaken for each other.
_ALLOW_SCALE_LINE_MARKER = "hygiene-allow-scale"
_ALLOW_SCALE_BLOCK_START_MARKER = "hygiene-allow-scale-start"
_ALLOW_SCALE_BLOCK_END_MARKER = "hygiene-allow-scale-end"

#: Declaration properties whose values encode a typography/radius/timing
#: scale value: font-size, font-weight, line-height, the border-radius
#: family (border-radius plus any physical or logical corner longhand, e.g.
#: border-top-left-radius/border-start-end-radius), and the
#: duration-bearing transition/animation properties. Spacing
#: (margin/padding/gap) and z-index are deliberately absent — see the
#: module docstring for why.
_SCALE_DECLARATION_RE = re.compile(
    r"\b(font-size|font-weight|line-height|"
    r"border(?:-\w+-\w+)?-radius|"
    r"transition-duration|animation-duration|animation-delay|transition|animation)"
    r"\s*:\s*([^;{}]+)"
)


def _is_scale_literal(prop: str, value: str) -> bool:
    """Whether the already-stripped ``value`` is a bare scale literal for ``prop``.

    Each property has its own notion of "bare literal" versus a legitimate
    pass-through (a ``var()`` reference, a keyword, an outright ``0``, a
    percentage for radius, ...) — see the PLAN task and the module
    docstring's check (d) paragraph for the rationale behind the
    per-property split.
    """
    if prop == "font-size":
        return bool(re.search(r"\d", value)) and not value.startswith("var(")
    if prop == "font-weight":
        return bool(re.match(r"\s*(\d{3}|bold|normal)\b", value))
    if prop == "line-height":
        return bool(re.search(r"\d", value)) and "var(" not in value
    if prop.endswith("radius"):
        return any(
            re.match(r"^\d+(\.\d+)?px$", component) and component != "0px"
            for component in value.split()
        )
    # transition, animation, transition-duration, animation-duration, animation-delay:
    # strip actual var() calls first so a var()-supplied duration mixed with
    # other shorthand keywords (color, ease, ...) still passes, then look for
    # a bare mN/s literal in whatever's left.
    without_var_calls = re.sub(r"var\([^)]*\)", "", value)
    return bool(re.search(r"\d+(\.\d+)?m?s\b", without_var_calls))


def _count_scale_literals(text: str) -> list[str]:
    """Every non-allowlisted scale-literal declaration in one file's text.

    Line-based, mirroring :func:`_count_hardcoded_colors`: honors both the
    single-line ``hygiene-allow-scale`` marker and the
    ``hygiene-allow-scale-start``/``-end`` block markers. Returns the
    matched ``"property: value"`` text for each hit (not just a count) so a
    failing assertion can list the actual offending declarations.
    """
    hits: list[str] = []
    in_allowed_block = False
    for line in text.splitlines():
        if _ALLOW_SCALE_BLOCK_START_MARKER in line:
            in_allowed_block = True
            continue
        if _ALLOW_SCALE_BLOCK_END_MARKER in line:
            in_allowed_block = False
            continue
        if in_allowed_block or _ALLOW_SCALE_LINE_MARKER in line:
            continue
        for prop, raw_value in _SCALE_DECLARATION_RE.findall(line):
            value = raw_value.strip()
            if _is_scale_literal(prop, value):
                hits.append(f"{prop}: {value}")
    return hits


def test_scale_literal_scanner() -> None:
    assert _count_scale_literals("a { font-size: 11px; }") == ["font-size: 11px"]
    assert _count_scale_literals("a { font-size: var(--text-base); }") == []
    assert _count_scale_literals("a { font-size: inherit; }") == []
    assert _count_scale_literals("a { font-weight: 600; }") == ["font-weight: 600"]
    assert _count_scale_literals("a { font-weight: bold; }") == ["font-weight: bold"]
    assert _count_scale_literals("a { line-height: 1.5; }") == ["line-height: 1.5"]
    assert _count_scale_literals("a { border-radius: 3px; }") == ["border-radius: 3px"]
    assert _count_scale_literals("a { border-radius: 50%; }") == []
    assert _count_scale_literals("a { border-radius: 0; }") == []
    assert (
        _count_scale_literals("a { border-radius: var(--radius-sm) 0 0 var(--radius-sm); }") == []
    )
    assert _count_scale_literals("a { transition: color 0.15s ease; }") == [
        "transition: color 0.15s ease"
    ]
    assert _count_scale_literals("a { transition: color var(--duration-fast) ease; }") == []
    assert _count_scale_literals("a { font-size: 7px; } /* hygiene-allow-scale: x */") == []


#: Interfaces where check (d) is enforced with zero tolerance. Empty as of
#: this commit — this task ships only the scanner and its self-tests, not
#: a CSS migration; a later task (web-terminal scale migration) flips this
#: to {"web_terminal", "design_system"} once their CSS actually moves onto
#: the scale variables.
SCALE_ENFORCED_INTERFACES: frozenset[str] = frozenset()


def test_scale_literal_zero_tolerance() -> None:
    """No ``.css`` file in a scale-enforced interface may contain a bare
    scale literal (see the module docstring's check (d) paragraph for what
    counts and why spacing/z-index are excluded from enforcement).

    A ratchet, not a fleet-wide rule (unlike checks (a)-(c)): scoped by
    ``SCALE_ENFORCED_INTERFACES``, empty until the interfaces it will name
    finish their scale migration.
    """
    offenders = []
    for path in _in_scope_files():
        if path.suffix != ".css" or _interface_group(path) not in SCALE_ENFORCED_INTERFACES:
            continue
        for hit in _count_scale_literals(path.read_text(encoding="utf-8")):
            offenders.append(f"{_relpath(path)}: {hit}")
    assert not offenders, (
        "Scale literal(s) in a scale-enforced interface — use var(--text-*/"
        "--weight-*/--leading-*/--radius-*/--duration-*) or mark a deliberate "
        "exception with `/* hygiene-allow-scale: <reason> */`:\n" + "\n".join(offenders)
    )
