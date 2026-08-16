"""Pins for the identity-stable, in-place re-themed CLI consoles.

`styles.py` owns exactly two Console instances - one for stdout, one for stderr.
Both must keep their identity for the process lifetime so that module-level
``from .styles import console`` aliases (taken at import time, before any theme
is applied) stay correct after `set_theme()`.
"""

import sys
from collections.abc import Iterator

import pytest
from rich.style import Style
from rich.text import Text

from osprey.cli import styles
from osprey.cli.styles import ColorTheme, console, err_console, get_active_theme, set_theme


@pytest.fixture(autouse=True)
def restore_theme() -> Iterator[None]:
    """Restore the theme active before the test, so globals stay clean."""
    original = get_active_theme()
    yield
    set_theme(original)


def _theme_stack_depth(target: object) -> int:
    """Return the number of entries on a console's theme stack."""
    return len(target._theme_stack._entries)  # type: ignore[attr-defined]


def _rendered_style(target: object, markup: str) -> Style:
    """Return the resolved Style of the first non-blank segment of `markup`."""
    segments = [seg for seg in target.render(Text.from_markup(markup)) if seg.text.strip()]  # type: ignore[attr-defined]
    assert segments, f"no visible segments rendered for {markup!r}"
    return segments[0].style


def test_console_identity_stable_across_theme_switches() -> None:
    """Both consoles keep their object identity across repeated set_theme calls."""
    stdout_id, stderr_id = id(console), id(err_console)

    for color in ("#00ff00", "#0000ff", "#ff00ff"):
        set_theme(ColorTheme(success=color))
        assert styles.console is console
        assert styles.err_console is err_console
        assert id(styles.console) == stdout_id
        assert id(styles.err_console) == stderr_id


def test_import_time_alias_sees_new_theme() -> None:
    """An alias captured before theming (as command modules do) stays correct."""
    aliased = console  # what `from .styles import console` yields at import time

    set_theme(ColorTheme(success="#00ff00"))

    assert aliased is styles.console
    assert aliased.get_style("success").color.name == "#00ff00"


def test_theme_stack_depth_constant_across_repeated_switches() -> None:
    """Pop-before-push keeps the theme stack flat, however often the theme changes."""
    set_theme(ColorTheme(success="#00ff00"))
    stdout_depth, stderr_depth = _theme_stack_depth(console), _theme_stack_depth(err_console)

    for color in ("#0000ff", "#ff00ff", "#00ffff", "#ffff00"):
        set_theme(ColorTheme(success=color))
        assert _theme_stack_depth(console) == stdout_depth
        assert _theme_stack_depth(err_console) == stderr_depth

    # Base theme plus exactly one pushed theme.
    assert stdout_depth == 2
    assert stderr_depth == 2


def test_err_console_writes_to_stderr_and_is_themed() -> None:
    """The stderr console is a real stderr console carrying the same theme."""
    assert err_console.stderr is True
    assert err_console.file is sys.stderr
    assert console.stderr is False
    assert console.file is sys.stdout

    set_theme(ColorTheme(success="#00ff00", path="#123456"))

    assert err_console.get_style("success") == console.get_style("success")
    assert err_console.get_style("path").color.name == "#123456"


def test_rendered_styles_change_after_set_theme() -> None:
    """Re-theming actually changes rendered output, on both consoles."""
    set_theme(ColorTheme(success="#00ff00"))
    before_out = _rendered_style(console, "[success]ok[/success]")
    before_err = _rendered_style(err_console, "[success]ok[/success]")
    assert before_out.color.triplet == (0, 255, 0)
    assert before_err.color.triplet == (0, 255, 0)

    set_theme(ColorTheme(success="#0000ff"))
    after_out = _rendered_style(console, "[success]ok[/success]")
    after_err = _rendered_style(err_console, "[success]ok[/success]")

    assert after_out.color.triplet == (0, 0, 255)
    assert after_err.color.triplet == (0, 0, 255)
    assert after_out != before_out
    assert after_err != before_err


def test_default_theme_restored_by_switching_back() -> None:
    """Switching back to the default theme restores the default styles."""
    default_success = console.get_style("success")

    set_theme(ColorTheme(success="#00ff00"))
    assert console.get_style("success") != default_success

    set_theme(styles.OSPREY_THEME)
    assert console.get_style("success") == default_success
    assert err_console.get_style("success") == default_success
