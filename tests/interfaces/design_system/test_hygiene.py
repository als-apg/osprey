"""Fleet-wide hygiene scanner: hardcoded colors, ``var(--x)`` integrity,
stray token-defining blocks, and scale-literal drift.

This module is the "hygiene" leg of the test pyramid described in the
frontend-design-system PLAN (Task 1.11). It scans every ``.css``/``.js``/
``.html`` asset under ``src/osprey/interfaces/`` plus the single dispatch
dashboard file (the design system is mounted there too, see Task 4.1) for
each of the independent kinds of drift enumerated below:

(a) Hardcoded color literals (hex, ``rgb()``/``rgba()``, ``hsl()``/
    ``hsla()``) that should have been expressed as ``var(--token)``
    references instead. This is a strict zero-tolerance check (Task 4.2,
    hygiene-zero-flip): there is no ``hygiene_baseline.json`` ratchet to
    grade against, and every in-scope file must have zero
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
    which names the interfaces that have completed their CSS migration onto
    the scale variables (that set is the authoritative membership list —
    interfaces join it as they finish their respective migrations, so it is
    deliberately not re-enumerated here). An interface migrates
    its CSS onto the emitted scales, then adds its directory name to lock the
    migration in place and prevent new scale literals. Spacing (margin/padding/
    gap) and z-index are deliberately excluded from this check even for an
    enforced interface: spacing literals are far denser throughout existing
    CSS than the other five properties, and z-index frequently encodes
    legitimate intra-component micro-stacking (a dropdown one layer above
    its trigger, a tooltip above both) that a shared scale can't cleanly
    express. Both may still migrate opportunistically, and the emitted
    ``--space-*``/``--z-*`` scales remain the preferred spelling for new
    code — they're just not enforced here.

(e) ARIEL's retired parallel scale layer: standalone ARIEL used to ship its
    own ``--ariel-text-*``/``--ariel-space-*``/``--ariel-radius-*``/
    ``--ariel-leading-*``/``--ariel-z-*`` scales plus an unprefixed
    ``--transition-*`` trio, and its JS wrote scale literals into inline
    ``style=``/``cssText`` strings. The ariel-design-parity migration
    replaced all of it with the fleet scales; this check keeps the layer
    from growing back in the two places check (d) structurally cannot see —
    custom-property *definitions* (check (d) only reads declaration
    *values*) and ``.js`` files (check (d) scans ``.css`` only). The
    ``--ariel-score-*`` badge colors are exempt throughout: they are
    generator-emitted extension tokens declared in ``tokens.css``, not part
    of the retired local layer.

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
        # Prose false positive in the bluesky panel: its header comment
        # describes token usage as "var(--…)" — the scanner's balanced-paren
        # extraction spuriously parses that prose as a reference to a property
        # literally named "…". This file was always clean — it
        # only entered the fleet-wide scan when the package moved from
        # services/ to interfaces/ (layering fix).
        ("src/osprey/interfaces/bluesky_web/panels/bluesky/panel.css", "…"),
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


# --- Check (d): scale-literal ratchet (per-interface enforcement) ------------------

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


#: Interfaces where check (d) is enforced with zero tolerance. An interface
#: migrates its CSS onto the emitted scales (--text-*, --weight-*, --leading-*,
#: --radius-*, --duration-*), then adds its directory name to this set to lock
#: the migration in place and prevent new scale literals. Interfaces not yet
#: listed are subject to the color-hygiene check (a) only.
SCALE_ENFORCED_INTERFACES: frozenset[str] = frozenset({"web_terminal", "design_system", "ariel"})


def test_scale_literal_zero_tolerance() -> None:
    """No ``.css`` file in a scale-enforced interface may contain a bare
    scale literal (see the module docstring's check (d) paragraph for what
    counts and why spacing/z-index are excluded from enforcement).

    A ratchet, not a fleet-wide rule (unlike checks (a)-(c)): scoped by
    ``SCALE_ENFORCED_INTERFACES``. An interface is added to the set once its
    CSS migration onto the scale variables is complete.
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


# --- Check (e): ARIEL's retired parallel scale layer stays retired -----------------

#: The interface directory name whose local scale layer this check guards.
_ARIEL_GROUP = "ariel"

#: A custom-property DECLARATION of one of the retired local scale families:
#: the five ``--ariel-<family>-*`` scales plus the unprefixed
#: ``--transition-*`` trio the migration folded into ``--duration-*``. The
#: trailing ``:`` is what makes this a definition rather than a reference —
#: ``var(--transition-fast)`` and ``var(--transition-fast, 150ms)`` both lack
#: it, so a surviving *usage* is left to check (b) (it dangles once the
#: definition is gone) and only the definition is reported here.
#: ``--ariel-score-*`` is absent from the family list by design: those badge
#: colors are generator-emitted extension tokens, not part of the retired
#: layer.
_ARIEL_SCALE_DEFINITION_RE = re.compile(
    r"--(?:ariel-(?:text|space|radius|leading|z)|transition)-[a-zA-Z0-9-]+\s*:"
)

#: A ``var(--ariel-...)`` reference in ARIEL JS. The negative lookahead keeps
#: the sanctioned ``--ariel-score-*`` extension tokens usable.
_ARIEL_VAR_USE_RE = re.compile(r"var\(\s*--ariel-(?!score-)[a-zA-Z0-9-]+")

#: A CSS comment, stripped (newline-for-newline, so line numbers survive)
#: before the definition scan: prose recording what a value *used* to be
#: spelled cannot define anything.
_CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

#: The scale-bearing declarations worth catching inside a JS string literal:
#: font-size, font-weight, and the border-radius family. Spacing is excluded
#: for the same reason check (d) excludes it in CSS. The value stops at a
#: declaration separator or at either quote character, so a declaration that
#: runs to the end of an unterminated ``style="..."`` attribute inside a
#: template literal is still captured cleanly.
_JS_SCALE_DECLARATION_RE = re.compile(
    r"\b(font-size|font-weight|border(?:-\w+-\w+)?-radius)\s*:\s*([^;{}\"'`]+)"
)

#: A number carrying a typographic length unit. Percentages, unitless ``0``,
#: and CSS-wide keywords (``inherit``, ``none``, ``transparent``, ...) are
#: deliberately not matched — none of them encode a scale step.
_LENGTH_LITERAL_RE = re.compile(r"\d+(?:\.\d+)?(?:px|rem|em)\b")

#: Characters after which a ``/`` is a division operator rather than the
#: start of a regex literal (identifier/literal enders). Everything else —
#: ``=``, ``(``, ``,``, ``:``, ``return``'s trailing space, ... — precedes a
#: regex literal.
_DIVISION_AFTER = ")]}"


def _mask_non_string_regions(text: str) -> str:
    """Blank every character of JS ``text`` that is not inside a string literal.

    Returns a same-length, same-line-numbering copy in which the contents of
    every ``'``/``"``/`` ` `` literal survive and everything else — code,
    line and block comments, regex literals — becomes a space. Scanning the
    mask instead of the raw text is what makes check (e)'s "inside a string
    literal" scope real: an inline style is always string content, while a
    comment merely *mentioning* ``font-size: 12px`` is not a style at all.

    Regex literals get their own skip arm rather than being left to the
    string scanner, because they routinely contain a lone quote character
    (``components.js``'s ``.replace(/"/g, '&quot;')``) that would otherwise
    open a string and desynchronize the rest of the file. Template-literal
    bodies are kept whole, interpolations included: an inline style built by
    interpolation is still an inline style.
    """
    out = ["\n" if ch == "\n" else " " for ch in text]
    i, n = 0, len(text)
    prev = ""
    while i < n:
        ch = text[i]
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        if ch == "/" and not (prev.isalnum() or prev in _DIVISION_AFTER):
            j = i + 1
            in_class = False
            while j < n and text[j] != "\n":
                c = text[j]
                if c == "\\":
                    j += 2
                    continue
                if c == "[":
                    in_class = True
                elif c == "]":
                    in_class = False
                elif c == "/" and not in_class:
                    break
                j += 1
            prev = "/"
            i = j + 1
            continue
        if ch in "'\"`":
            quote = ch
            j = i + 1
            while j < n:
                c = text[j]
                if c == "\\":
                    j += 2
                    continue
                if c == quote:
                    break
                out[j] = c
                j += 1
            prev = quote
            i = j + 1
            continue
        if not ch.isspace():
            prev = ch
        i += 1
    return "".join(out)


def _is_js_scale_literal(prop: str, value: str) -> bool:
    """Whether an inline-style ``prop: value`` pair carries a bare scale literal.

    Mirrors :func:`_is_scale_literal`'s per-property split, restricted to the
    three properties inline styles actually reach for: ``var()`` references
    are stripped first so a token-supplied value mixed with other components
    still passes, and the leftovers are judged per property.
    """
    without_var_calls = re.sub(r"var\([^)]*\)", "", value)
    if prop == "font-weight":
        return bool(re.search(r"\b\d{3}\b", without_var_calls)) or bool(
            re.match(r"\s*(bold|bolder|lighter|normal)\b", without_var_calls)
        )
    if prop == "font-size":
        return bool(_LENGTH_LITERAL_RE.search(without_var_calls))
    return any(
        _LENGTH_LITERAL_RE.fullmatch(component) and component not in ("0px", "0rem", "0em")
        for component in without_var_calls.split()
    )


def _js_scale_literals(text: str) -> list[tuple[int, str]]:
    """Every non-allowlisted inline-style scale literal in one JS file's text.

    Returns ``(line_number, "property: value")`` pairs. Honors the same
    ``hygiene-allow-scale`` line and ``-start``/``-end`` block markers as
    check (d) — the marker is read off the *original* line, so it works
    whether it sits in a comment beside the code or inside the string itself.
    """
    hits: list[tuple[int, str]] = []
    masked_lines = _mask_non_string_regions(text).splitlines()
    in_allowed_block = False
    source_lines = text.splitlines()
    for lineno, (line, masked) in enumerate(zip(source_lines, masked_lines, strict=True), start=1):
        if _ALLOW_SCALE_BLOCK_START_MARKER in line:
            in_allowed_block = True
            continue
        if _ALLOW_SCALE_BLOCK_END_MARKER in line:
            in_allowed_block = False
            continue
        if in_allowed_block or _ALLOW_SCALE_LINE_MARKER in line:
            continue
        for prop, raw_value in _JS_SCALE_DECLARATION_RE.findall(masked):
            value = raw_value.strip()
            if _is_js_scale_literal(prop, value):
                hits.append((lineno, f"{prop}: {value}"))
    return hits


def _ariel_files(suffix: str) -> list[Path]:
    """Every in-scope ARIEL asset with the given suffix."""
    return [
        path
        for path in _in_scope_files()
        if path.suffix == suffix and _interface_group(path) == _ARIEL_GROUP
    ]


def test_js_string_mask() -> None:
    """The mask keeps string contents and drops code, comments, and regexes."""
    assert _mask_non_string_regions("x.style.cssText = 'font-size: 12px;';").strip() == (
        "font-size: 12px;"
    )
    # A comment mentioning a declaration is not an inline style.
    assert _mask_non_string_regions("// font-size: 12px is deliberate").strip() == ""
    assert _mask_non_string_regions("/* font-size: 12px */").strip() == ""
    # A regex literal carrying a lone quote must not open a string and swallow
    # the rest of the file (components.js's escapeHtml chain does exactly this).
    masked = _mask_non_string_regions("s.replace(/\"/g, '&quot;');\nq = 'font-size: 9px';")
    assert masked.splitlines()[1].strip() == "font-size: 9px"
    # Line numbering is preserved across a multi-line template literal.
    assert _js_scale_literals('const t = `\n  <div style="font-size: 32px">\n`;') == [
        (2, "font-size: 32px")
    ]


def test_js_scale_literal_scanner() -> None:
    assert _js_scale_literals("a = 'font-size: 32px;'") == [(1, "font-size: 32px")]
    assert _js_scale_literals("a = 'font-size: 0.85rem;'") == [(1, "font-size: 0.85rem")]
    assert _js_scale_literals("a = 'font-size: var(--text-4xl);'") == []
    assert _js_scale_literals("a = 'font-size: inherit;'") == []
    assert _js_scale_literals("a = 'font-weight: 600;'") == [(1, "font-weight: 600")]
    assert _js_scale_literals("a = 'font-weight: bold;'") == [(1, "font-weight: bold")]
    assert _js_scale_literals("a = 'font-weight: var(--weight-semibold);'") == []
    assert _js_scale_literals("a = 'border-radius: 4px;'") == [(1, "border-radius: 4px")]
    assert _js_scale_literals("a = 'border-radius: 50%;'") == []
    assert _js_scale_literals("a = 'border-radius: 0;'") == []
    assert _js_scale_literals("a = 'border-radius: var(--radius-md) 0 0 var(--radius-md);'") == []
    # Spacing is exempt here exactly as it is in check (d).
    assert _js_scale_literals("a = 'width: 100px; margin: 8px 0;'") == []
    assert _js_scale_literals("a = 'font-size: 9px;' // hygiene-allow-scale: x") == []


def test_ariel_retired_name_scanners() -> None:
    """The definition and reference regexes hit definitions only, score never."""
    assert _ARIEL_SCALE_DEFINITION_RE.findall("  --ariel-text-sm: 14px;") == ["--ariel-text-sm:"]
    assert _ARIEL_SCALE_DEFINITION_RE.findall("  --transition-fast: 150ms;") == [
        "--transition-fast:"
    ]
    # A surviving *reference* is check (b)'s business, not this one's.
    assert _ARIEL_SCALE_DEFINITION_RE.findall("  color: var(--transition-fast);") == []
    assert _ARIEL_SCALE_DEFINITION_RE.findall("  color: var(--ariel-text-sm, 14px);") == []
    # The sanctioned extension tokens are exempt on both sides.
    assert _ARIEL_SCALE_DEFINITION_RE.findall("  --ariel-score-high-bg: red;") == []
    assert _ARIEL_VAR_USE_RE.findall("color: var(--ariel-score-high);") == []
    assert _ARIEL_VAR_USE_RE.findall("color: var(--ariel-space-3);") == ["var(--ariel-space-3"]


def test_ariel_scale_layer_stays_retired() -> None:
    """ARIEL's retired local scale layer may not grow back.

    Three prohibitions, none of which check (d) can express (it reads
    declaration values in ``.css`` files only):

    - No ``--ariel-(text|space|radius|leading|z)-*`` or ``--transition-*``
      custom property is *defined* anywhere in ARIEL's CSS. Redefining the
      retired scale is how a parallel vocabulary comes back even while every
      individual declaration value dutifully reads ``var(--something)``.
    - No ``var(--ariel-...)`` reference survives in ARIEL's JS, where a
      dangling reference is otherwise easy to miss.
    - No font-size/font-weight/border-radius literal appears inside a JS
      string literal — the inline ``style=``/``cssText`` strings that are
      check (d)'s blind spot.

    ``--ariel-score-*`` is exempt in all three: those badge colors are
    generator-emitted extension tokens (they are declared in ``tokens.css``
    like any other token), not remnants of the local layer. Only the third
    prohibition carries a ``hygiene-allow-scale`` escape, mirroring check
    (d); the first two have none, because a retired name always has a fleet
    spelling to migrate to.
    """
    offenders: list[str] = []

    for path in _ariel_files(".css"):
        text = path.read_text(encoding="utf-8")
        without_comments = _CSS_COMMENT_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
        for lineno, line in enumerate(without_comments.splitlines(), start=1):
            for match in _ARIEL_SCALE_DEFINITION_RE.findall(line):
                offenders.append(f"{_relpath(path)}:{lineno}: retired scale definition {match}")

    for path in _ariel_files(".js"):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in _ARIEL_VAR_USE_RE.findall(line):
                offenders.append(f"{_relpath(path)}:{lineno}: retired scale reference {match})")
        for lineno, hit in _js_scale_literals(text):
            offenders.append(f"{_relpath(path)}:{lineno}: inline-style scale literal {hit}")

    assert not offenders, (
        "ARIEL's retired --ariel-*/--transition-* scale layer is growing back — "
        "use the fleet scales instead (var(--text-*/--weight-*/--leading-*/"
        "--radius-*/--space-*/--z-*/--duration-*)); --ariel-score-* is the only "
        "sanctioned --ariel- name:\n" + "\n".join(offenders)
    )
