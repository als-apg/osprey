"""The freshness gate: the real, committed generated artifacts must never drift.

Unlike every other test in this package, this one is deliberately NOT
hermetic — it is the drift gate CI (and ``build --check``) rely on, so it
must exercise the real ``tokens/`` source tree against the real,
committed ``static/css/tokens.css``, ``static/js/tokens.js``, and
``static/js/theme-boot.js``. If a token source changes without
regenerating these three files, this test (and the equivalent
``build --check`` invocation) fails.

This module also carries the two other things Task 1.7 is responsible
for finalizing, now that the real build has actually run against the
real token tree:

- the five CSS custom property names ``base.css`` (hand-written, a
  different task's file) hard-depends on existing;
- the rename-mapping table from every name the web-terminal
  ``variables.css``/``theme-light.css`` vocabulary defines to whatever
  the generator actually emits for it — verbatim, or a documented
  rename. Downstream migration tasks (2.2 onward) delete those two files
  and need this mapping to know what to point every ``var(--old-name)``
  call site at. See the inline ``_DOCUMENTED_RENAMES`` table below for the
  four names Task 1.1 originally missed (surfaced only once the real
  build ran and the real CSS files were diffed against it) and the
  ``--crt-*``/``--welcome-*`` -> ``--wt-crt-*``/``--wt-welcome-*``
  namespacing already planned in Task 1.1.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

import osprey.interfaces.design_system as design_system_pkg
from osprey.interfaces.design_system.generator.build import (
    DEFAULT_STATIC_DIR,
    DEFAULT_TOKENS_DIR,
    build_artifacts,
    check_artifacts,
)
from osprey.interfaces.design_system.generator.emit_css import emit_css
from osprey.interfaces.design_system.generator.model import load_token_tree

_VARIABLES_CSS = (
    Path(design_system_pkg.__file__).parents[1]
    / "web_terminal"
    / "static"
    / "css"
    / "variables.css"
)
_THEME_LIGHT_CSS = (
    Path(design_system_pkg.__file__).parents[1]
    / "web_terminal"
    / "static"
    / "css"
    / "theme-light.css"
)

#: Every name the OLD web-terminal ``variables.css``/``theme-light.css``
#: vocabulary defines that the generator does NOT emit verbatim, mapped to
#: the name it emits instead. Two groups:
#:
#: 1. ``--crt-*``/``--welcome-*`` -> ``--wt-crt-*``/``--wt-welcome-*``:
#:    planned in Task 1.1 (these were always going to be namespaced
#:    extension tokens, not semantic ones).
#: 2. ``--header-height``, ``--status-bar-height``, ``--transition-fast``,
#:    ``--transition-normal``, ``--accent-system-tint-04``: Task 1.1
#:    genuinely missed these four theme-independent layout/motion values
#:    and one color tint — surfaced only by running the real build and
#:    diffing its output against the real, hand-authored CSS files (this
#:    task's whole purpose). All five are live dependencies of
#:    scaffold.css/operator.css/terminal.css/files.css/settings.css/
#:    drawer.css/sessions.css/md-rendered.css/session.html (60+ call
#:    sites) and were added to ``tokens/interfaces/web_terminal.json``
#:    as ``wt-*`` extension tokens rather than left bare, to avoid
#:    reopening the flat-namespace collision risk the wt-/art-/ariel-/
#:    lat-/cf- convention exists to close (namespace-collision is a
#:    validated invariant, see ``validate.check_namespace_collisions``).
#:    Every migration task that still references one of these five old
#:    names by the time it deletes variables.css/theme-light.css must
#:    rewrite it to the ``wt-`` name instead.
_DOCUMENTED_RENAMES: dict[str, str] = {
    "crt-scanline-opacity": "wt-crt-scanline-opacity",
    "crt-vignette-opacity": "wt-crt-vignette-opacity",
    "crt-glow-opacity": "wt-crt-glow-opacity",
    "crt-bezel-shadow": "wt-crt-bezel-shadow",
    "crt-scanline-bg": "wt-crt-scanline-bg",
    "welcome-bg": "wt-welcome-bg",
    "welcome-text": "wt-welcome-text",
    "welcome-text-glow": "wt-welcome-text-glow",
    "welcome-link": "wt-welcome-link",
    "welcome-link-hover": "wt-welcome-link-hover",
    "welcome-prompt": "wt-welcome-prompt",
    "welcome-cursor": "wt-welcome-cursor",
    "header-height": "wt-header-height",
    "status-bar-height": "wt-status-bar-height",
    "transition-fast": "wt-transition-fast",
    "transition-normal": "wt-transition-normal",
    "accent-system-tint-04": "wt-accent-system-tint-04",
}

#: CSS custom property names ``design_system/static/css/base.css``
#: (a different task's hand-written file) references directly. A hard
#: constraint: whatever token sources/emitter changes happen, these five
#: names must keep existing in the emitted CSS.
_BASE_CSS_HARD_DEPENDENCIES: tuple[str, ...] = (
    "bg-primary",
    "text-primary",
    "font-display",
    "neutral-tint-20",
    "neutral-tint-35",
)

_VAR_DECLARATION_PATTERN = re.compile(r"--([a-zA-Z0-9-]+)\s*:")


def _declared_var_names(css_text: str) -> set[str]:
    """Every ``--name`` a CSS custom-property declaration defines in ``css_text``."""
    return set(_VAR_DECLARATION_PATTERN.findall(css_text))


# --- The freshness gate itself ----------------------------------------------------


def test_regeneration_is_byte_identical_to_checked_in_artifacts() -> None:
    """The actual drift gate: real tokens/ must regenerate the real static/ files exactly."""
    artifacts = build_artifacts(DEFAULT_TOKENS_DIR)

    diffs = check_artifacts(artifacts, DEFAULT_STATIC_DIR)

    assert diffs == [], "\n\n".join(
        f"{diff.relative_path} has drifted:\n{diff.unified_diff}" for diff in diffs
    )


def test_check_cli_reports_up_to_date_against_the_real_tree(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The exact `build --check` invocation the validation gate runs, in-process."""
    from osprey.interfaces.design_system.generator.build import main

    exit_code = main(["--check"])

    assert exit_code == 0
    assert "up to date" in capsys.readouterr().out


def test_check_cli_subprocess_matches_validation_gate_command() -> None:
    """Smoke test the literal `python -m ...generator.build --check` the gate runs."""
    result = subprocess.run(
        [sys.executable, "-m", "osprey.interfaces.design_system.generator.build", "--check"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "up to date" in result.stdout


def test_generated_artifacts_are_hook_clean_on_disk() -> None:
    """The checked-in files themselves, not just the in-memory render, are hook-clean."""
    for relative_path in (
        DEFAULT_STATIC_DIR / "css" / "tokens.css",
        DEFAULT_STATIC_DIR / "js" / "tokens.js",
        DEFAULT_STATIC_DIR / "js" / "theme-boot.js",
    ):
        content = relative_path.read_text(encoding="utf-8")
        assert "\r" not in content, f"{relative_path}: must use \\n line endings"
        assert content.endswith("\n") and not content.endswith("\n\n"), (
            f"{relative_path}: must have exactly one trailing newline"
        )
        for line in content.split("\n"):
            assert line == line.rstrip(), f"{relative_path}: trailing whitespace on {line!r}"


# --- Hard constraint: base.css's five CSS variable names ---------------------------


def test_base_css_hard_dependencies_are_emitted() -> None:
    tree = load_token_tree(DEFAULT_TOKENS_DIR)
    css = emit_css(tree)
    names = _declared_var_names(css)

    missing = [name for name in _BASE_CSS_HARD_DEPENDENCIES if name not in names]
    assert missing == [], f"base.css depends on --{{{', '.join(missing)}}}, not emitted"


# --- The rename-mapping table: every old variables.css name is accounted for -------


@pytest.mark.skipif(
    not (_VARIABLES_CSS.is_file() and _THEME_LIGHT_CSS.is_file()),
    reason="web_terminal variables.css/theme-light.css already deleted (post-migration)",
)
def test_every_variables_css_name_is_emitted_verbatim_or_documented_renamed() -> None:
    old_names = _declared_var_names(
        _VARIABLES_CSS.read_text(encoding="utf-8")
    ) | _declared_var_names(_THEME_LIGHT_CSS.read_text(encoding="utf-8"))

    tree = load_token_tree(DEFAULT_TOKENS_DIR)
    new_names = _declared_var_names(emit_css(tree))

    unaccounted = sorted(
        name
        for name in old_names
        if name not in new_names and _DOCUMENTED_RENAMES.get(name) not in new_names
    )
    assert unaccounted == [], (
        f"old variables.css/theme-light.css name(s) {unaccounted} are neither emitted "
        "verbatim nor covered by _DOCUMENTED_RENAMES"
    )


def test_documented_renames_all_resolve_to_real_emitted_names() -> None:
    # Guards the mapping table itself against bit-rot: every declared
    # rename target must actually exist in what the generator emits today.
    tree = load_token_tree(DEFAULT_TOKENS_DIR)
    new_names = _declared_var_names(emit_css(tree))

    dangling = sorted(new for new in _DOCUMENTED_RENAMES.values() if new not in new_names)
    assert dangling == [], f"_DOCUMENTED_RENAMES target(s) {dangling} are not actually emitted"
