"""Plain-line phase reporting for the OSPREY lifecycle verbs.

Lifecycle verbs (``init``, ``build``, ``up``, ``restart``, ``down``, ``reset``)
report progress as a short sequence of phases rendered as plain sequential
lines on stdout: ``→ title`` when a phase starts, ``  ✓ title (elapsed)`` when
it ends, ``  · name`` for sub-steps. No Rich Live, no spinners, no in-place
repainting -- TTY and non-TTY share one code path and only color differs.

The active reporter is a module-level singleton (the same pattern as
``styles.console``): verbs call :func:`install_reporter` at entry, helpers deep
in the call stack reach it through :func:`current_reporter`. The default is a
quiet :class:`NullReporter`, so helper calls are safe before any verb installs.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import TracebackType

from .styles import Styles, console


def format_elapsed(seconds: float) -> str:
    """Format an elapsed duration for a phase line."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"


class Phase:
    """One reported phase. Usable as a context manager or driven directly."""

    def __init__(self, reporter: PhaseReporter, title: str) -> None:
        self.title = title
        self.spool: Path | None = None
        self._reporter = reporter
        self._start = time.monotonic()
        self._lap = self._start
        self._closed = False

    @property
    def elapsed(self) -> float:
        """Seconds since the phase started."""
        return time.monotonic() - self._start

    def set_spool(self, path: Path | None) -> None:
        """Record the spool file the phase is currently writing to."""
        self.spool = path

    def step(self, name: str) -> None:
        """Print a sub-line under this phase.

        The duration shown is the lap since the previous step (or since the
        phase started), so a step reported after a unit of work names that
        unit's own cost. Laps under 50ms print without a duration.
        """
        now = time.monotonic()
        lap, self._lap = now - self._lap, now
        suffix = f" ({format_elapsed(lap)})" if lap >= 0.05 else ""
        self._reporter.emit(f"  · {name}{suffix}")

    def done(self, note: str = "") -> None:
        """Print the success line for this phase."""
        if self._closed:
            return
        self._closed = True
        suffix = f" — {note}" if note else ""
        self._reporter.emit(
            f"  ✓ {self.title} ({format_elapsed(self.elapsed)}){suffix}", style=Styles.SUCCESS
        )

    def fail(self, replay: Path | None = None) -> None:
        """Print the failure line, replaying ``replay``'s content in full."""
        if self._closed:
            return
        self._closed = True
        self._reporter.emit(
            f"  ✗ {self.title} ({format_elapsed(self.elapsed)})", style=Styles.ERROR
        )
        self._reporter.replay(replay)

    def interrupted(self) -> None:
        """Print one line naming the partial spool instead of replaying it."""
        if self._closed:
            return
        self._closed = True
        spool = f" — partial output: {self.spool}" if self.spool else ""
        self._reporter.emit(
            f"  ⚠ {self.title} interrupted ({format_elapsed(self.elapsed)}){spool}",
            style=Styles.WARNING,
        )

    def __enter__(self) -> Phase:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Always close the phase, whatever left the block.

        A clean ``SystemExit`` closes the phase as a success, not a failure. A
        verb whose last act is to exit on a child's status (``down`` propagates
        ``compose``'s exit code verbatim) leaves the block by raising
        ``SystemExit(0)``, and treating that like any other exception would
        print ``✗`` and replay the whole spool for a run that did exactly what
        it was asked to. A non-zero (or non-numeric) code is a real failure and
        still takes the failure path.
        """
        self._reporter.clear_phase(self)
        if exc_type is None or (isinstance(exc, SystemExit) and not exc.code):
            self.done()
        elif issubclass(exc_type, KeyboardInterrupt):
            self.interrupted()
        else:
            self.fail(self.spool)


class PhaseReporter:
    """Prints phase lines to stdout through the themed console."""

    verbose = False

    def __init__(self, *, color: bool | None = None) -> None:
        # The console is force_terminal on win32, so ask stdout directly.
        self.color = sys.stdout.isatty() if color is None else color
        self._phase: Phase | None = None

    def emit(self, text: str, style: str | None = None) -> None:
        """Print one plain line -- styled only when stdout is a terminal."""
        console.print(
            text,
            style=style if self.color else None,
            markup=False,
            highlight=False,
            soft_wrap=True,
        )

    def phase(self, title: str) -> Phase:
        """Start a phase, printing its opening line."""
        self.emit(f"→ {title}", style=Styles.BOLD)
        self._phase = Phase(self, title)
        return self._phase

    @property
    def current_phase(self) -> Phase | None:
        """The most recently started phase, if one is still open."""
        return self._phase

    def clear_phase(self, phase: Phase) -> None:
        """Drop ``phase`` as the current one (no-op if it is not)."""
        if self._phase is phase:
            self._phase = None

    def note_spool(self, path: Path | None) -> None:
        """Record a spool path on the open phase, for interrupt reporting."""
        if self._phase is not None:
            self._phase.set_spool(path)

    def replay(self, path: Path | None) -> None:
        """Dump a spool file's content to stdout in full."""
        if path is None:
            return
        self.emit(f"--- {path} ---")
        try:
            self.emit(Path(path).read_text(errors="replace").rstrip("\n"))
        except OSError as exc:
            self.emit(f"(could not read spool: {exc})")
        self.emit("--- end ---")


class NullReporter(PhaseReporter):
    """No-op reporter installed under the global ``--verbose`` flag."""

    def __init__(self, *, verbose: bool = False) -> None:
        super().__init__(color=False)
        self.verbose = verbose

    def emit(self, text: str, style: str | None = None) -> None:
        """Swallow the line -- verbose mode streams the real output instead."""

    def replay(self, path: Path | None) -> None:
        """Nothing to replay: the output already went to the terminal."""


_reporter: PhaseReporter = NullReporter()


def current_reporter() -> PhaseReporter:
    """Return the installed reporter (a quiet :class:`NullReporter` default)."""
    return _reporter


def install_reporter(reporter: PhaseReporter) -> PhaseReporter:
    """Install ``reporter`` as the active one and return the previous one."""
    global _reporter
    previous, _reporter = _reporter, reporter
    return previous


def is_verbose() -> bool:
    """True when the verbs installed the reporter for global ``--verbose``."""
    return _reporter.verbose


def report_step(name: str) -> None:
    """Report ``name`` as a sub-step of the open phase, if there is one.

    The one helper every deploy site uses to hang a ``  · name`` line off the
    verb's phase. Those sites run both under a lifecycle verb (which installs a
    reporter and opens a phase) and from tests or library callers that never do,
    so the line is conditional on there being a phase to hang it under.

    Call it AFTER the work it names: :meth:`Phase.step` prints the lap since the
    previous step, so a step reported afterwards carries that unit's own cost.
    """
    phase = current_reporter().current_phase
    if phase is not None:
        phase.step(name)
