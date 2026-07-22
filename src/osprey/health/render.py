"""Rich and JSON rendering for a completed health-check report.

The runner produces a :class:`~osprey.health.models.CheckReport`; this module
turns it into operator-facing output. Two surfaces are provided:

* :func:`render_report` — grouped, per-category Rich output following the
  ``cli/styles.py`` conventions (``✓``/``!``/``✗`` glyphs, a dim ``-`` for
  skips, ``[bold]`` category headers) followed by a summary panel. In verbose
  mode the panel gains a details section listing every warning and error.
* :func:`render_json` — the machine-clean path: ``json.dumps(report.to_dict())``
  and nothing else on the target stream (stdout by default). All human output
  (the progress spinner, deprecation warnings) is expected to be routed to
  stderr by the caller, keeping stdout a pure JSON document.

:func:`run_progress` supplies the during-the-run indicator (a transient Rich
``Live`` spinner). Every entry point takes an optional ``console`` so the CLI
can direct human output to a stderr console when ``--json`` is in effect while
the report JSON goes to stdout.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner

from osprey.cli.styles import Messages, ThemeConfig
from osprey.cli.styles import console as _default_console
from osprey.health.models import CheckReport, CheckResult, Status

_PANEL_TITLE = "Osprey Health Check Results"
_DEFAULT_PROGRESS_TEXT = "Running health checks…"


def _humanize(category: str) -> str:
    """Render a snake_case category name as a display header.

    Kept intentionally generic (``file_system`` -> ``File System``) so YAML- and
    plugin-declared categories render consistently with the core ones, with no
    hand-curated title table to maintain.
    """
    return category.replace("_", " ").title()


def _format_row(result: CheckResult) -> str:
    """Format one result as a Rich-markup line: glyph, message, optional value."""
    text = result.message
    if result.value:
        text = f"{text} [dim]({result.value})[/dim]"

    if result.status is Status.OK:
        return f"  {Messages.success(text)}"
    if result.status is Status.WARNING:
        return f"  {Messages.warning(text)}"
    if result.status is Status.ERROR:
        return f"  {Messages.error(text)}"
    # Status.SKIP — dim dash, consistent with STATUS_ICONS in models.
    return f"  [dim]- {text}[/dim]"


def _group_by_category(results: list[CheckResult]) -> dict[str, list[CheckResult]]:
    """Group results by category, preserving first-seen category order."""
    grouped: dict[str, list[CheckResult]] = {}
    for result in results:
        grouped.setdefault(result.category, []).append(result)
    return grouped


def _build_panel(report: CheckReport, *, verbose: bool) -> Panel:
    """Build the summary panel; include a details section when verbose."""
    lines = [f"Summary: {report.summary_line()}"]

    if verbose and (report.warning_count or report.error_count):
        lines.append("")
        lines.append("Details:")
        for result in report.results:
            if result.status in (Status.WARNING, Status.ERROR):
                symbol = "!" if result.status is Status.WARNING else "✗"
                lines.append(f"  {symbol} {result.name}: {result.message}")
                if result.details:
                    lines.append(f"     {result.details}")

    return Panel(
        "\n".join(lines),
        title=_PANEL_TITLE,
        border_style="dim",
        expand=False,
        padding=(1, 2),
    )


def render_report(
    report: CheckReport,
    *,
    verbose: bool = False,
    console: Console | None = None,
) -> None:
    """Render a completed report as grouped per-category Rich output.

    Prints a ``[bold]`` header per category followed by one glyphed line per
    check, then a summary :class:`~rich.panel.Panel`. When ``verbose`` is set the
    panel also lists every warning and error with its details.

    Args:
        report: The completed report to render.
        verbose: Whether to include the panel's per-row details section.
        console: Target console; defaults to the shared themed CLI console. Pass
            a stderr-bound console to keep stdout clean alongside ``--json``.
    """
    out = console or _default_console

    for category, results in _group_by_category(report.results).items():
        out.print(f"\n[bold]{_humanize(category)}[/bold]")
        for result in results:
            out.print(_format_row(result))

    out.print()
    out.print(_build_panel(report, verbose=verbose))


def render_json(report: CheckReport, *, out: IO[str] | None = None) -> None:
    """Write the report as a single JSON document and nothing else.

    Emits ``json.dumps(report.to_dict())`` followed by a newline to ``out``
    (stdout by default). No human-facing text is written here, so a caller that
    routes progress and warnings to stderr leaves stdout a pure JSON document.

    Args:
        report: The completed report to serialize.
        out: Target text stream; defaults to :data:`sys.stdout`.
    """
    stream = out if out is not None else sys.stdout
    stream.write(json.dumps(report.to_dict()))
    stream.write("\n")
    stream.flush()


@contextmanager
def run_progress(
    description: str = _DEFAULT_PROGRESS_TEXT,
    *,
    console: Console | None = None,
) -> Iterator[None]:
    """Show a transient spinner while the suite runs.

    The spinner is rendered via a Rich :class:`~rich.live.Live` in transient
    mode, so it leaves no residue once the run completes. In ``--json`` mode the
    caller passes a stderr-bound console so the indicator never touches stdout.

    Args:
        description: Text shown next to the spinner.
        console: Target console; defaults to the shared themed CLI console.
    """
    out = console or _default_console
    spinner = Spinner(
        "dots", text=f"[dim]{description}[/dim]", style=ThemeConfig.get_spinner_style()
    )
    with Live(spinner, console=out, transient=True):
        yield


__all__ = ["render_report", "render_json", "run_progress"]
