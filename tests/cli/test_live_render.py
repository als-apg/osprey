"""The live build region renders a snapshot, whatever shape the rows are in.

Every assertion goes through a recording :class:`~rich.console.Console` built on
the CLI's own theme, so a style token that does not resolve fails here rather
than at the first cold build.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from rich.console import Console

from osprey.cli import live_render
from osprey.cli.live_render import format_duration, render_live_region
from osprey.cli.styles import ColorTheme, _build_rich_theme, osprey_theme
from osprey.deployment.build_progress import EXPORT_STEP, BuildModel, BuildRow
from tests.deployment.test_build_progress import BOOKKEEPING, PINNED_CAPTIONS

FIXTURES = Path(__file__).resolve().parents[1] / "deployment" / "fixtures"

#: The compose build the parser's own tests replay, rendered here as a table.
COLD_BUILD = FIXTURES / "buildkit_cold_build.log"


def make_row(
    service: str = "virtual-accelerator",
    step: str | None = "6/8",
    caption: str = "RUN pip install -e .",
    started_at: float = 0.0,
    finished: str | None = None,
) -> BuildRow:
    """A row as ``BuildModel.snapshot`` would hand one over."""
    return BuildRow(
        service=service,
        step=step,
        caption=caption,
        started_at=started_at,
        last_line_at=started_at,
        finished=finished,
    )


def render(rows, *, phase_open=False, elapsed=0.0, now=0.0, width=100, styles=False) -> str:
    """Render the region at ``width`` and return what a terminal would show."""
    console = Console(
        theme=osprey_theme,
        width=width,
        record=True,
        force_terminal=True,
        color_system="standard",
        no_color=False,
    )
    console.print(
        render_live_region(rows, phase_open=phase_open, elapsed=elapsed, now=now, width=width)
    )
    return console.export_text(styles=styles)


# -- rows -------------------------------------------------------------------


@pytest.mark.parametrize("step", ["6/8", "internal", "auth", EXPORT_STEP])
def test_row_shows_every_step_shape(step: str) -> None:
    """The step column is not always ``i/T``; each shape prints as itself."""
    row = make_row(step=step, caption="exporting layers")
    line = render([row], now=242.0).strip()
    assert line.split() == ["virtual-accelerator", step, "exporting", "layers", "4m02s"]


def test_row_without_a_step_keeps_its_other_columns() -> None:
    """A row seen before its first header still renders — it does not blank."""
    line = render([make_row(step=None)], now=78.0).strip()
    assert line.split() == ["virtual-accelerator", "-", "RUN", "pip", "install", "-e", ".", "1m18s"]


def test_rows_keep_snapshot_order_and_are_indented_under_the_phase() -> None:
    """Insertion order is the model's; the table nests below the phase line."""
    rows = [
        make_row(service="virtual-accelerator", started_at=0.0),
        make_row(service="bluesky-bridge", started_at=100.0),
    ]
    lines = render(rows, phase_open=True, elapsed=252.0, now=252.0).splitlines()
    assert lines[0].endswith("4m12s")
    assert lines[1] == ""
    assert [line[:4] for line in lines[2:4]] == ["    ", "    "]
    assert [line.split()[0] for line in lines[2:4]] == ["virtual-accelerator", "bluesky-bridge"]


def test_row_age_is_measured_from_the_caller_s_clock() -> None:
    """``now`` is injected, so a fifteen-minute build renders in a millisecond."""
    row = make_row(started_at=1_000.0)
    assert render([row], now=1_000.0).strip().endswith("0m00s")
    assert render([row], now=1_925.0).strip().endswith("15m25s")


def test_step_column_does_not_reflow_as_the_step_changes_shape() -> None:
    """A real build walks None → internal → auth → 1/13 → 10/13 → exporting.

    Those are 1, 8, 4, 4, 5 and 9 columns wide. A step column sized from
    whatever has arrived would widen twice mid-build and drag every caption
    sideways with it, so it is pinned at the widest step there is.
    """
    steps = [None, "internal", "auth", "1/13", "10/13", EXPORT_STEP]
    starts = {render([make_row(step=step)]).index("RUN pip") for step in steps}
    assert len(starts) == 1


def test_caption_is_truncated_to_the_given_width() -> None:
    """Only the caption gives way to a narrow terminal, and it says so with …."""
    row = make_row(caption="Downloading torch (2.1 GB) from an unusually chatty mirror")
    wide = render([row], width=120).strip()
    narrow = render([row], width=60).strip()

    assert wide.endswith("chatty mirror   0m00s")
    assert "…" in narrow
    assert "chatty mirror" not in narrow
    # The columns that identify the row survive intact; the width is respected.
    assert narrow.startswith("virtual-accelerator   6/8         ")
    assert narrow.endswith("0m00s")
    assert len(narrow) + 4 <= 60


def test_ticker_carries_the_spinner_and_elapsed_but_never_the_title() -> None:
    """The phase's title is printed once, permanently, by the reporter — so the
    region that repaints under it carries only what keeps moving.

    A ticker that repeated the title would show it twice on screen and read as a
    rendering glitch, and the operator ruled that out.
    """
    line = render([], phase_open=True, elapsed=252.0).rstrip()
    # The spinner frame itself is whichever one Rich is on; only that a frame
    # was drawn ahead of the duration is asserted.
    assert re.fullmatch(r" {2}\S {1}4m12s", line)


def test_rows_render_without_an_open_phase() -> None:
    """Watchers can outlive the phase that started them; the table stays."""
    out = render([make_row()], phase_open=False)
    assert out.splitlines()[0].lstrip().startswith("virtual-accelerator")


def test_caption_is_never_read_as_markup() -> None:
    """BuildKit captions contain brackets; none of them is a style tag."""
    out = render([make_row(caption="load metadata for [internal] python:3.12")])
    assert "[internal]" in out


# -- what the fixture actually puts in the caption column --------------------

#: One rendered row, split back into its columns. The caption is the only one
#: that may hold spaces, and the duration anchors the split from the right.
_RENDERED_ROW = re.compile(
    r" {4}(?P<service>\S+)\s{2,}(?P<step>\S+)\s{2,}(?P<caption>.*?)\s{2,}\d+m\d{2}s"
)


def replay_rendered_captions() -> list[str]:
    """Every caption the table showed while replaying the cold build.

    The region is re-rendered only when a row actually changed, which is what
    keeps a 860-line replay to a few hundred frames; the states themselves are
    all still visited.
    """
    model = BuildModel()
    captions: list[str] = []
    previous: object = None
    for tick, line in enumerate(COLD_BUILD.read_bytes().decode().split("\n")):
        model.feed(line, float(tick))
        rows = model.snapshot()
        state = [(row.service, row.step, row.caption, row.finished) for row in rows]
        if state == previous:
            continue
        previous = state
        for rendered in render(rows, now=float(tick), width=200).splitlines():
            if not rendered.strip():
                continue
            parsed = _RENDERED_ROW.fullmatch(rendered.rstrip())
            assert parsed is not None, rendered
            captions.append(parsed["caption"])
    return captions


def test_the_table_never_shows_buildkit_bookkeeping() -> None:
    """The symptom that started this: a table whose every caption read
    `DONE 0.0s`, or a step column doubled by BuildKit's own stopwatch
    (`6/8  21.51 Collecting scikit-learn`)."""
    captions = replay_rendered_captions()

    assert captions
    assert [caption for caption in captions if BOOKKEEPING.match(caption)] == []


def test_the_table_shows_the_work_the_stream_reported() -> None:
    """The same five captions the model and the heartbeat lines pin, at the
    surface an operator on a terminal actually reads."""
    captions = set(replay_rendered_captions())

    assert [caption for caption in PINNED_CAPTIONS if caption not in captions] == []


# -- finished services ------------------------------------------------------


def test_finished_services_leave_the_table() -> None:
    """A built image is already permanent in the scrollback; the table is for
    what is still outstanding."""
    rows = [
        make_row(service="virtual-accelerator", finished="my-project/va:local"),
        make_row(service="bluesky-bridge"),
        make_row(service="event-dispatcher", finished="my-project/dispatch:local"),
    ]
    out = render(rows)
    assert [line.split()[0] for line in out.splitlines()] == ["bluesky-bridge"]


def test_all_finished_collapses_the_region() -> None:
    """The last image lands on every successful build; nothing may be left
    behind between then and the phase closing."""
    console = Console(theme=osprey_theme, width=100)
    rows = [make_row(finished="my-project/va:local")]
    region = render_live_region(rows, phase_open=False, elapsed=0.0, now=0.0, width=100)
    assert console.render_lines(region, pad=False) == []


def test_all_finished_under_an_open_phase_leaves_no_blank_gap() -> None:
    """The phase is still running, so its ticker stays; what goes is the table
    and the blank line that separated it."""
    rows = [make_row(finished="my-project/va:local")]
    lines = render(rows, phase_open=True, elapsed=252.0).splitlines()
    assert len(lines) == 1
    assert lines[0].endswith("4m12s")


# -- the empty region -------------------------------------------------------


def test_empty_region_has_zero_height() -> None:
    """No phase and no rows must cost zero lines, not one blank one."""
    console = Console(theme=osprey_theme, width=100)
    region = render_live_region([], phase_open=False, elapsed=0.0, now=0.0, width=100)
    assert console.render_lines(region, pad=False) == []


# -- durations --------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "0m00s"), (0.4, "0m00s"), (54.9, "0m54s"), (252.0, "4m12s"), (-1.0, "0m00s")],
)
def test_format_duration(seconds: float, expected: str) -> None:
    """Fixed ``MmSSs`` width, so a repainting counter does not jitter."""
    assert format_duration(seconds) == expected


# -- theming ----------------------------------------------------------------

#: A literal color where a semantic token belongs: a hex triplet, or one of
#: Rich's standard color names, quoted (optionally behind ``bold``/``dim``).
_COLOR_LITERAL = re.compile(
    r"""["'](?:(?:bold|dim|italic|bright)\s+)*"""
    r"""(?:\#[0-9a-fA-F]{3,6}"""
    r"""|black|red|green|yellow|blue|magenta|cyan|white"""
    r"""|bright_\w+|grey\d*|gray\d*|purple|orange\w*|pink\w*)["']""",
)


def test_module_hardcodes_no_color() -> None:
    """Every style is a token from ``styles.py``, so a retheme reaches here.

    Grepped rather than trusted: a hardcoded ``"cyan"`` still renders, and would
    only be noticed once ``osprey theme-lab`` showed one region ignoring the
    theme.
    """
    source = Path(live_render.__file__).read_text()
    assert _COLOR_LITERAL.search(source) is None


def test_region_follows_the_active_theme() -> None:
    """Swap the theme's colors and the rendered escape codes change with them."""
    rows = [make_row()]
    loud = ColorTheme(primary="#112233", accent="#445566", success="#778899", info="#aabbcc")

    def styled(theme: ColorTheme) -> str:
        console = Console(
            theme=_build_rich_theme(theme),
            width=100,
            record=True,
            force_terminal=True,
            color_system="standard",
            no_color=False,
        )
        console.print(render_live_region(rows, phase_open=True, elapsed=1.0, now=1.0, width=100))
        return console.export_text(styles=True)

    assert styled(ColorTheme()) != styled(loud)
