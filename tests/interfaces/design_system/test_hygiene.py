"""Fleet-wide hygiene scanner: hardcoded colors and ``var(--x)`` integrity.

This module is the "hygiene" leg of the test pyramid described in the
frontend-design-system PLAN (Task 1.11). It scans every ``.css``/``.js``/
``.html`` asset under ``src/osprey/interfaces/`` plus the single dispatch
dashboard file (the design system is mounted there too, see Task 4.1) for
two independent kinds of drift:

(a) Hardcoded color literals (hex, ``rgb()``/``rgba()``, ``hsl()``/
    ``hsla()``) that should have been expressed as ``var(--token)``
    references instead. Since the fleet migration (PLAN Phase 2/3) removes
    these incrementally, file-by-file, this check is a *ratchet*: each
    in-scope file's live count may only stay the same or go down relative
    to ``hygiene_baseline.json``, and a file with no baseline entry must
    have zero literals. Task 4.2 (hygiene-zero-flip) deletes the baseline
    entirely and flips this to a strict zero-tolerance check once the
    fleet migration finishes — see that task before changing this ratchet
    logic.

(b) ``var(--name)`` reference integrity: every custom-property reference
    in the same in-scope assets must resolve to either a name the token
    generator emits (the authoritative set is read straight off the
    committed ``tokens.css``) or a name defined locally within the same
    interface's own asset set — either a CSS/inline-style
    ``--name: ...`` declaration or a JS ``element.style.setProperty
    ('--name', ...)`` call. This is NOT a ratchet: it must hold at every
    commit. A handful of genuinely dangling references already exist at
    the time this test was authored (pre-existing bugs, not something
    Task 1.11 is scoped to fix) — see ``_KNOWN_DANGLING_VARS`` below for
    exactly which ones and why, with the owning migration task for each.
    That allowlist is checked in both directions: no unexplained new
    dangling ref may appear, and no allowlisted entry may go stale (i.e.
    every migration task that fixes one of these must delete its entry
    here in the same commit).

Both checks share the same allowlist idea in spirit — a literal, commented
exception list — but check (a)'s allowlist is *in the scanned files
themselves* (an inline marker comment, since ownership of those files
belongs to the migration tasks, not this one) while check (b)'s lives in
this module (there is nothing sensible to "comment out" in a way that
survives a source edit for a missing declaration).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import osprey.interfaces.design_system as design_system_pkg

_INTERFACES_ROOT = Path(design_system_pkg.__file__).parents[1]
_REPO_ROOT = Path(design_system_pkg.__file__).parents[4]
_DASHBOARD_HTML = _REPO_ROOT / "src" / "osprey" / "dispatch" / "dashboard.html"
_TOKENS_CSS = _INTERFACES_ROOT / "design_system" / "static" / "css" / "tokens.css"
_BASELINE_PATH = Path(__file__).parent / "hygiene_baseline.json"

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


# --- Check (a): hardcoded-color ratchet -------------------------------------------

#: Matches a hex color (#rgb/#rrggbb/#rrggbbaa, with word-boundary guards so
#: e.g. a URL fragment like "#deadbeef-section" is still counted only once,
#: not double-matched) or an rgb()/rgba()/hsl()/hsla() function call.
_COLOR_RE = re.compile(
    r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})\b"
    r"|\b(?:rgba?|hsla?)\([^)]*\)"
)

#: Commented allowlist mechanism (PLAN Task 1.11 requirement; no current use
#: — the first consumer is expected to be Task 3.2's `@media print` rule,
#: which intentionally stays light-on-white regardless of theme). A line
#: containing this marker is never counted; ``-start``/``-end`` variants
#: bracket a multi-line block (both boundary lines are themselves exempt).
_ALLOW_LINE_MARKER = "hygiene-allow-color"
_ALLOW_BLOCK_START_MARKER = "hygiene-allow-color-start"
_ALLOW_BLOCK_END_MARKER = "hygiene-allow-color-end"


def _count_hardcoded_colors(text: str) -> int:
    """Count non-allowlisted hardcoded-color occurrences in one file's text."""
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
        count += len(_COLOR_RE.findall(line))
    return count


def test_hardcoded_color_ratchet() -> None:
    """Per-file hardcoded-color counts may only decrease from the baseline.

    A file with no baseline entry must have zero hardcoded-color literals.
    This is the migration's regression guard: every fleet-migration task
    updates ``hygiene_baseline.json`` downward as it removes literals: see
    Task 4.2 (hygiene-zero-flip), which deletes the baseline file and flips
    this test to strict zero-tolerance once every task has landed.
    """
    files = _in_scope_files()
    baseline: dict[str, int] = json.loads(_BASELINE_PATH.read_text())

    current: dict[str, int] = {}
    for path in files:
        n = _count_hardcoded_colors(path.read_text(encoding="utf-8"))
        if n:
            current[_relpath(path)] = n

    in_scope_relpaths = {_relpath(path) for path in files}
    stale_entries = sorted(set(baseline) - in_scope_relpaths)
    assert not stale_entries, (
        "hygiene_baseline.json has entries for files that are no longer in scope "
        "(deleted, renamed, or moved out of src/osprey/interfaces/) — remove them: "
        f"{stale_entries}"
    )

    regressions = []
    for relpath, baseline_count in baseline.items():
        live_count = current.get(relpath, 0)
        if live_count > baseline_count:
            regressions.append(
                f"{relpath}: {live_count} hardcoded colors, baseline allows only "
                f"{baseline_count} (counts may only decrease — did you add a new "
                "literal instead of using a design token?)"
            )
    assert not regressions, "\n".join(regressions)

    unbaselined = []
    for relpath, live_count in current.items():
        if relpath not in baseline:
            unbaselined.append(
                f"{relpath}: {live_count} hardcoded colors but no baseline entry "
                "(new files must use design tokens, not literals — see "
                "tests/interfaces/design_system/hygiene_baseline.json)"
            )
    assert not unbaselined, "\n".join(unbaselined)


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
        # ariel's spacing scale only goes up to --space-16; --space-32 was
        # never added. `padding-right` computes to an invalid value.
        # Owned by: migrate-ariel.
        ("src/osprey/interfaces/ariel/static/css/components.css", "space-32"),
        # A second, fallback-less `--amber` usage distinct from the
        # already-documented entries.js:584 phantom (`var(--amber,
        # #f59e0b)`, which degrades safely via its literal fallback and is
        # tracked separately in PROPOSAL.md as a rename to --color-amber).
        # This one has no fallback at all, so `border-left` is invalid.
        # Owned by: migrate-ariel.
        ("src/osprey/interfaces/ariel/static/js/entries.js", "amber"),
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
