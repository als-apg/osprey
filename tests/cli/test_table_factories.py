"""The table and panel factories are the CLI's single table and panel look.

These tests pin the three things a call site is allowed to rely on: the defaults
each factory bakes in, that an override wins over a default, and that the look
is expressed in *theme tokens* rather than colors — so a table built before a
theme switch renders in the new theme afterwards.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from rich import box
from rich.panel import Panel
from rich.segment import Segment
from rich.table import Table
from rich.text import Text

from osprey.cli import styles
from osprey.cli.styles import (
    BORDER_STYLE,
    DATA_TABLE_BOX,
    DATA_TABLE_PADDING,
    LAYOUT_TABLE_GUTTER,
    LAYOUT_TABLE_PADDING,
    PANEL_BOX,
    PANEL_PADDING,
    TABLE_HEADER_STYLE,
    ColorTheme,
    Styles,
    data_table,
    layout_table,
    panel,
)


@pytest.fixture
def restore_theme() -> Iterator[None]:
    """Put the process-wide theme back after a test switches it."""
    original = styles.get_active_theme()
    try:
        yield
    finally:
        styles.set_theme(original)


def _segments(renderable: object) -> list[Segment]:
    """Render on the module console and keep the segments that carry text.

    Under pytest neither console is a terminal, so a captured string carries no
    ANSI. Styles are only visible on the segments themselves.
    """
    return [seg for seg in styles.console.render(renderable) if seg.text.strip()]


def _style_of(renderable: object, text: str) -> object:
    """The resolved style of the first rendered segment whose text is ``text``."""
    for seg in _segments(renderable):
        if seg.text.strip() == text:
            return seg.style
    raise AssertionError(f"no rendered segment with text {text!r}")


# ---------------------------------------------------------------------------
# data_table
# ---------------------------------------------------------------------------


def test_data_table_defaults_are_the_bordered_themed_look():
    table = data_table()

    assert isinstance(table, Table)
    assert table.box is DATA_TABLE_BOX
    assert table.box is box.HEAVY_HEAD
    assert table.show_header is True
    assert table.header_style == TABLE_HEADER_STYLE
    assert table.border_style == BORDER_STYLE
    # Rich normalizes padding to (top, right, bottom, left).
    assert DATA_TABLE_PADDING == (0, 1)
    assert table.padding == (0, 1, 0, 1)


def test_data_table_style_defaults_are_semantic_tokens_not_colors():
    table = data_table()

    assert TABLE_HEADER_STYLE == Styles.HEADER
    assert BORDER_STYLE == Styles.BORDER_DIM
    for value in (table.header_style, table.border_style):
        assert isinstance(value, str)
        assert not str(value).startswith("#")


def test_data_table_returns_a_fresh_table_each_call():
    first, second = data_table(), data_table()

    assert first is not second
    first.add_column("Name")
    assert second.columns == []


def test_data_table_overrides_pass_through():
    table = data_table(title="Services", show_lines=True, expand=False)

    assert table.title == "Services"
    assert table.show_lines is True
    assert table.expand is False
    # Untouched defaults survive an override of something else.
    assert table.box is DATA_TABLE_BOX
    assert table.header_style == TABLE_HEADER_STYLE


def test_data_table_overrides_win_over_the_defaults():
    table = data_table(box=None, header_style=Styles.ACCENT, border_style=Styles.ERROR, padding=0)

    assert table.box is None
    assert table.header_style == Styles.ACCENT
    assert table.border_style == Styles.ERROR
    assert table.padding == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# layout_table
# ---------------------------------------------------------------------------


def test_layout_table_is_boxless_and_headerless():
    table = layout_table()

    assert isinstance(table, Table)
    assert table.box is None
    assert table.show_header is False
    assert table.pad_edge is False
    assert table.padding == LAYOUT_TABLE_PADDING
    assert LAYOUT_TABLE_PADDING == (0, 0, 0, LAYOUT_TABLE_GUTTER)


def test_layout_table_draws_no_rules():
    table = layout_table()
    table.add_column()
    table.add_column()
    table.add_row(Text("alpha"), Text("beta"))

    rendered = "".join(seg.text for seg in styles.console.render(table))
    assert "alpha" in rendered and "beta" in rendered
    # No box means no rule characters anywhere in the output.
    assert not any(char in rendered for char in "─│┌┐└┘━┃┏┓┗┛")


def test_layout_table_overrides_pass_through():
    table = layout_table(padding=(0, 2), pad_edge=True, show_header=True)

    assert table.padding == (0, 2, 0, 2)
    assert table.pad_edge is True
    assert table.show_header is True
    assert table.box is None


def test_live_region_uses_the_layout_table_gutter():
    """The live region's gutter is the factory's, not a second spelling of 3."""
    from osprey.cli import live_render

    assert live_render._GUTTER is LAYOUT_TABLE_GUTTER


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------


def test_panel_defaults_are_the_single_panel_look():
    built = panel("contents")

    assert isinstance(built, Panel)
    assert built.box is PANEL_BOX
    assert built.box is box.ROUNDED
    assert built.border_style == BORDER_STYLE
    assert built.padding == PANEL_PADDING
    assert built.title is None


def test_panel_wraps_the_renderable_it_is_given():
    body = Text("Registry Contents", style=Styles.HEADER)
    built = panel(body)

    assert built.renderable is body


def test_panel_overrides_pass_through():
    built = panel("contents", title="Audit Summary", expand=False, border_style=Styles.ERROR)

    assert built.title == "Audit Summary"
    assert built.expand is False
    assert built.border_style == Styles.ERROR
    assert built.box is PANEL_BOX


def test_panel_renders_its_contents_inside_a_rounded_frame():
    rendered = "".join(seg.text for seg in styles.console.render(panel("contents")))

    assert "contents" in rendered
    assert "╭" in rendered and "╯" in rendered


# ---------------------------------------------------------------------------
# The factories follow the active theme
# ---------------------------------------------------------------------------


def test_data_table_header_follows_set_theme(restore_theme):
    table = data_table()
    table.add_column("Name")
    table.add_row("registry")

    before = _style_of(table, "Name")

    styles.set_theme(ColorTheme(primary="#00ff00"))
    after = _style_of(table, "Name")

    assert before != after
    assert after.color.triplet.hex == "#00ff00"
    assert after.bold is True


def test_panel_border_follows_set_theme(restore_theme):
    built = panel("contents")

    def border_style():
        return next(seg.style for seg in _segments(built) if "╭" in seg.text)

    before = border_style()

    styles.set_theme(ColorTheme(border_dim="#0000ff"))
    after = border_style()

    assert before != after
    assert after.color.triplet.hex == "#0000ff"


def test_a_table_built_before_a_theme_switch_renders_in_the_new_theme(restore_theme):
    """The factory bakes tokens, not colors, so instances outlive a retheme."""
    table = data_table()
    table.add_column("Name")
    table.add_row("registry")

    styles.set_theme(ColorTheme(primary="#123456"))
    switched = _style_of(table, "Name")

    styles.set_theme(ColorTheme(primary="#654321"))
    switched_again = _style_of(table, "Name")

    assert switched.color.triplet.hex == "#123456"
    assert switched_again.color.triplet.hex == "#654321"
