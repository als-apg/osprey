"""Human and JSON rendering for a completed health-check report.

The runner produces a :class:`~osprey.health.models.CheckReport`; this module
turns it into operator-facing output. Two surfaces are provided:

* :func:`render_report` — the human report, printed entirely through the CLI
  renderer (:mod:`osprey.cli.output`). Category headers and per-check rows are
  report-class lines on stdout, because the grouped report is the document an
  operator ran the verb to read; the verbose recap of every degraded check uses
  the renderer's warning and failure shapes, which land on stderr like all other
  trouble.
* :func:`render_json` — the machine-clean path: ``json.dumps(report.to_dict())``
  and nothing else on the target stream (stdout by default). This is the one
  documented emission seam in the module: it writes to the stream directly
  rather than through the renderer, so nothing can slip a human sentence into
  the document. The caller wraps the whole run in
  :func:`osprey.cli.output.machine_mode`, which sends every renderer line to
  stderr, leaving stdout a single JSON document.

:func:`run_progress` supplies the during-the-run indicator (a transient Rich
``Live`` spinner) on the reporter's own console, so it can never repaint over a
line the renderer is printing — or on the stderr console in machine mode, where
stdout is carrying the document.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO

from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from osprey.cli import output, styles
from osprey.cli.phase_reporter import current_reporter
from osprey.cli.styles import Styles, ThemeConfig
from osprey.health.models import CheckReport, CheckResult, Status

_DEFAULT_PROGRESS_TEXT = "Running health checks…"

#: The renderer's one indentation step, spelled here because
#: :func:`~osprey.cli.output.report` prints its line verbatim: a check row is
#: subordinate to its category header, and structure is the half of that which
#: survives a pipe.
_INDENT = "  "

#: The mark each status opens its row with. Content rather than decoration —
#: these are the marks the renderer and :data:`~osprey.health.models.STATUS_ICONS`
#: already use, so a health row reads like every other line the CLI prints and
#: still says which way a check went once the color is gone.
_GLYPHS = {
    Status.OK: "✓",
    Status.WARNING: "⚠",
    Status.ERROR: "✗",
    Status.SKIP: "-",
}

#: The style token each status' row carries. Skips are absent: they render as
#: notes, which are dim by construction.
_ROW_STYLES = {
    Status.OK: Styles.SUCCESS,
    Status.WARNING: Styles.WARNING,
    Status.ERROR: Styles.ERROR,
}


def _humanize(category: str) -> str:
    """Render a snake_case category name as a display header.

    Kept intentionally generic (``file_system`` -> ``File System``) so YAML- and
    plugin-declared categories render consistently with the core ones, with no
    hand-curated title table to maintain.
    """
    return category.replace("_", " ").title()


def _row_text(result: CheckResult) -> str:
    """The prose half of one row: its message, with its value in parentheses."""
    if result.value:
        return f"{result.message} ({result.value})"
    return result.message


def _render_row(result: CheckResult) -> None:
    """Print one check as a row under its category header."""
    text = _row_text(result)
    if result.status is Status.SKIP:
        # A note is already indented and dim, which is what a skipped check is.
        output.note(f"{_GLYPHS[Status.SKIP]} {text}")
        return
    output.report(f"{_INDENT}{_GLYPHS[result.status]} {text}", style=_ROW_STYLES[result.status])


def _group_by_category(results: list[CheckResult]) -> dict[str, list[CheckResult]]:
    """Group results by category, preserving first-seen category order."""
    grouped: dict[str, list[CheckResult]] = {}
    for result in results:
        grouped.setdefault(result.category, []).append(result)
    return grouped


def _render_details(report: CheckReport) -> None:
    """Print every warning and error again in the renderer's trouble shape.

    The grouped report says what each category found; this recap says what an
    operator has to act on, in the same shape every other warning and failure in
    the CLI wears, and on stderr like all of them. Only reached under
    ``--verbose``.
    """
    for result in report.results:
        summary = f"{result.name}: {result.message}"
        if result.status is Status.WARNING:
            output.warn(summary, result.details)
        elif result.status is Status.ERROR:
            output.fail(summary, result.details)


def render_report(report: CheckReport, *, verbose: bool = False) -> None:
    """Render a completed report as grouped per-category output.

    Prints a bold header per category followed by one glyphed row per check,
    then the summary line. When ``verbose`` is set, every warning and error is
    printed again through the renderer's warning and failure shapes.

    Args:
        report: The completed report to render.
        verbose: Whether to add the per-check details recap.
    """
    for category, results in _group_by_category(report.results).items():
        output.report("")
        output.report(_humanize(category), style=Styles.BOLD)
        for result in results:
            _render_row(result)

    output.report("")
    output.report(f"Summary: {report.summary_line()}", style=Styles.BOLD)

    if verbose and (report.warning_count or report.error_count):
        _render_details(report)


def render_json(report: CheckReport, *, out: IO[str] | None = None) -> None:
    """Write the report as a single JSON document and nothing else.

    Emits ``json.dumps(report.to_dict())`` followed by a newline to ``out``
    (stdout by default). This is the module's machine-output seam: it writes at
    the stream rather than through the renderer, so no human line can reach the
    document. Everything human the run produces goes to stderr instead, because
    the caller holds :func:`osprey.cli.output.machine_mode` open around it.

    Args:
        report: The completed report to serialize.
        out: Target text stream; defaults to :data:`sys.stdout`.
    """
    stream = out if out is not None else sys.stdout
    stream.write(json.dumps(report.to_dict()))
    stream.write("\n")
    stream.flush()


@contextmanager
def run_progress(description: str = _DEFAULT_PROGRESS_TEXT) -> Iterator[None]:
    """Show a transient spinner while the suite runs.

    The spinner is rendered via a Rich :class:`~rich.live.Live` in transient
    mode, so it leaves no residue once the run completes. It mounts on the
    installed reporter's console — the one that owns the cursor — so a line the
    renderer prints during the run lands above the region rather than inside it.

    A live region paints at its console's stream, which is the one thing a
    ``--json`` run cannot afford on stdout, so in machine mode the region mounts
    on the stderr console instead: the operator still sees that the suite is
    running, and the document stays a document. This is the case the primitives
    handle for themselves and a mounted region cannot, which is what
    :func:`~osprey.cli.output.machine_mode_active` is for.

    Args:
        description: Text shown next to the spinner.
    """
    console = styles.err_console if output.machine_mode_active() else current_reporter().out()
    spinner = Spinner(
        "dots", text=Text(description, style=Styles.DIM), style=ThemeConfig.get_spinner_style()
    )
    with Live(spinner, console=console, transient=True):
        yield


__all__ = ["render_report", "render_json", "run_progress"]
