"""Phase reporting for the OSPREY lifecycle verbs.

Lifecycle verbs (``init``, ``build``, ``up``, ``restart``, ``down``, ``reset``)
report progress as a short sequence of phases rendered as plain sequential
lines on stdout: ``→ title`` when a phase starts, ``  ✓ title (elapsed)`` when
it ends, ``  · name`` for sub-steps. Those lines are the permanent record on
every path, terminal or not, and nothing rewrites them once printed.

On a terminal :class:`LiveReporter` adds a live region *underneath* that record:
the open phase repainted with a spinner and a ticking elapsed, and a row per
service while a build runs (see :mod:`osprey.cli.live_render`). The region is
decoration over the same lines -- it is transient, it never replaces a line, and
switching it off leaves exactly the plain-line output. Off a terminal the same
parsed build state drives append-only heartbeat lines instead, so a quarter-hour
build is never silent either way.

The active reporter is a module-level singleton (the same pattern as
``styles.console``): verbs call :func:`install_reporter` at entry, helpers deep
in the call stack reach it through :func:`current_reporter`. The default is a
quiet :class:`NullReporter`, so helper calls are safe before any verb installs.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING

from rich.console import Group
from rich.live import Live
from rich.logging import RichHandler
from rich.text import Text

from . import styles
from .live_render import render_live_region
from .styles import Styles, console

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rich.console import Console

    from osprey.deployment.build_progress import BuildModel, BuildRow

#: How long a service may go unmentioned on stdout before a heartbeat line
#: proves it is still working. A cold ``compose build`` runs a quarter of an
#: hour with one step in front of it; off a TTY there is no live region to
#: repaint, so the only way to tell that apart from a hang is to keep
#: appending. Deliberately a constant: an operator should never have to
#: configure "tell me you are alive".
_HEARTBEAT_INTERVAL = 30.0

#: How much of a BuildKit caption a heartbeat line carries. A step's opening
#: caption is its entire ``RUN`` command -- hundreds of characters of shell
#: that would swamp the log the heartbeat exists to keep readable.
_CAPTION_LIMIT = 100

#: How often the monitor thread wakes. Fast enough that a live region repaints
#: as motion rather than as a slideshow, slow enough to cost nothing; the
#: heartbeat's own interval is two orders of magnitude longer, so off a TTY
#: almost every tick decides "not yet" and goes back to sleep.
_MONITOR_INTERVAL = 0.25

#: How long a stop waits for the monitor to notice. The loop only ever sleeps
#: on its stop event, so this is reached only if a tick is wedged in terminal
#: I/O -- and the thread is a daemon, so a wedged one must not hold up the exit.
_MONITOR_JOIN_TIMEOUT = 2.0

#: The monitor thread's name. Named rather than anonymous so that a leaked
#: thread is identifiable in ``threading.enumerate()`` -- which is exactly how
#: the tests assert that every teardown path really joined it.
_MONITOR_THREAD_NAME = "osprey-phase-monitor"


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
        return self.elapsed_at(time.monotonic())

    def elapsed_at(self, now: float) -> float:
        """Seconds this phase had run as of ``now``.

        The clock is injected so a live repaint can take ONE reading and spend
        it on both the phase's ticker and the ages of the build rows under it:
        two readings a microsecond apart would render a frame whose parts
        disagree about what time it is.
        """
        return now - self._start

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
        """Print the failure line, replaying ``replay``'s content in full.

        Everything that moves stops first: the last thing an operator sees
        after a failure must be the failure, not a heartbeat for a build that
        died a moment ago, and the replayed spool must land as plain scrollback.
        """
        if self._closed:
            return
        self._closed = True
        self._reporter.stop_rendering()
        self._reporter.emit(
            f"  ✗ {self.title} ({format_elapsed(self.elapsed)})", style=Styles.ERROR
        )
        self._reporter.replay(replay)

    def interrupted(self) -> None:
        """Print one line naming the partial spool instead of replaying it.

        Stops the moving parts first, for the same reason :meth:`fail` does:
        Ctrl-C's answer is one line, and nothing may print after it.
        """
        if self._closed:
            return
        self._closed = True
        self._reporter.stop_rendering()
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


@dataclass(eq=False)
class _Watcher:
    """One registered build, plus when each of its services last beat.

    The beat state lives here rather than on the model so that it dies with
    the registration: a model registered again starts from silence instead of
    inheriting a stale schedule from its previous run.

    Compared by identity, so that deregistering one of two registrations over
    the same model removes that registration and not its twin.
    """

    model: BuildModel
    last_beat: dict[str, float] = field(default_factory=dict)


class PhaseReporter:
    """Prints phase lines to stdout through the themed console."""

    verbose = False

    def __init__(self, *, color: bool | None = None) -> None:
        # The console is force_terminal on win32, so ask stdout directly.
        self.color = sys.stdout.isatty() if color is None else color
        self._phase: Phase | None = None
        self._watchers: list[_Watcher] = []
        self._watch_lock = threading.Lock()
        self._monitor: threading.Thread | None = None
        self._monitor_stop = threading.Event()
        self._monitor_pinned = False
        self._monitor_lock = threading.Lock()
        self._suspend_depth = 0
        self._handed_off = False

    def out(self) -> Console:
        """The console this reporter prints through.

        Resolved per call rather than captured in ``__init__``: the module-level
        console is the one a caller swaps to capture output. The live renderer
        overrides this with the console its ``Live`` was built on -- the region
        and the lines above it MUST be the same console, or each would repaint
        over the other's cursor.
        """
        return console

    def emit(self, text: str, style: str | None = None) -> None:
        """Print one plain line -- styled only when stdout is a terminal."""
        self.out().print(
            text,
            style=style if self.color else None,
            markup=False,
            highlight=False,
            soft_wrap=True,
        )

    def echo(self, text: str) -> None:
        """Print one line of a verb's OWN output, rather than of the phase record.

        Not :meth:`emit`: that is the phase record, and :class:`NullReporter`
        swallows it under ``--verbose``. This is for the lines a verb owes its
        operator whatever reporter is installed -- ``init``'s report of what it
        created, ``reset``'s destruction plan -- so it is never silenced.

        Routed through the console rather than written straight at stdout,
        because on a terminal the console is what owns the cursor: a raw write
        lands INSIDE a mounted live region instead of above it. Off a terminal
        these are ``click.echo``'s bytes exactly, which is the parity the verbs'
        tests pin -- no wrapping (both callers have lines well past 80 columns),
        no markup and no highlighting (paths and brackets here are text, not
        style tags).
        """
        self.out().print(text, markup=False, highlight=False, soft_wrap=True)

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

    # -- live builds --------------------------------------------------------

    @contextmanager
    def watch_build(self, model: BuildModel) -> Iterator[BuildModel]:
        """Watch ``model``'s build for as long as the block runs.

        Scoped to one ``run_captured`` call -- the production shape is::

            with (report := compose_build_step_reporter()):
                run_captured(..., on_line=report)

        Deregistration happens on the way out however the block ends, which is
        the whole point of the scope: a build that has returned (or raised)
        must go quiet immediately. A model that is no longer registered never
        heartbeats again and leaves the live table.

        Yields the model itself, so a caller that has no other handle on it can
        bind one.
        """
        watcher = _Watcher(model)
        self._register(watcher)
        try:
            yield model
        finally:
            self._deregister(watcher)

    def _register(self, watcher: _Watcher) -> None:
        """Add ``watcher`` to the set the heartbeat pass reads.

        The first registration is what puts a clock behind the heartbeat off a
        TTY: nothing else in the process wakes up on its own, and a build that
        has gone quiet is precisely the case where no line will arrive to drive
        a pass. On a TTY the monitor is already running (the live region starts
        it), and starting it again is a no-op.
        """
        with self._watch_lock:
            self._watchers.append(watcher)
            first = len(self._watchers) == 1
        if first:
            self._start_monitor()

    def _deregister(self, watcher: _Watcher) -> None:
        """Drop ``watcher`` and the beat state that went with it.

        With the last build gone there is nothing left to beat for, so an
        unpinned monitor stops here rather than idling for the rest of the verb.
        A pinned one -- the live region's -- keeps running: it still has an open
        phase's elapsed to tick.
        """
        with self._watch_lock:
            if watcher in self._watchers:
                self._watchers.remove(watcher)
            idle = not self._watchers
        if idle and not self._monitor_pinned:
            self._stop_monitor()

    def _heartbeat_pass(self, now: float) -> list[str]:
        """The heartbeat lines due at ``now``, marking their services as beaten.

        One line per registered service that has gone
        :data:`_HEARTBEAT_INTERVAL` seconds without one, whether or not the
        build has printed anything in the meantime -- a service that stops
        producing output is exactly the case the line exists for.

        ``now`` is injected, and nothing here reads a clock: that is what lets
        a quarter-hour build's heartbeats be tested in milliseconds. Calling
        this *is* what advances the beat state, so a caller that drops the
        returned lines also silences the pass that would have followed them.

        Runs under the watch lock, so a build that has already deregistered
        cannot have a line produced for it after the fact.
        """
        lines = []
        with self._watch_lock:
            for watcher in self._watchers:
                for row in watcher.model.snapshot():
                    if row.finished is not None:
                        continue  # Already named its image: no longer running.
                    since = watcher.last_beat.get(row.service, row.started_at)
                    if now - since <= _HEARTBEAT_INTERVAL:
                        continue
                    watcher.last_beat[row.service] = now
                    lines.append(_heartbeat_line(row, now))
        return lines

    def _watched_rows(self) -> list[BuildRow]:
        """Every registered build's rows, as one list for the live region.

        The heartbeat pass's counterpart on the terminal side, and the only
        place the live region touches the models: taken under the watch lock and
        handed on as frozen rows, so the render itself holds no lock and cannot
        see a build change shape halfway down a frame.

        Concatenated rather than merged: two registrations at once means two
        genuinely separate builds (a compose run and a single image), and the
        service names that identify their rows do not collide.
        """
        with self._watch_lock:
            return [row for watcher in self._watchers for row in watcher.model.snapshot()]

    # -- monitor thread -----------------------------------------------------

    def _start_monitor(self, *, pinned: bool = False) -> None:
        """Start the monitor thread if it is not already running.

        One daemon thread serves both renderers: on a TTY it repaints the live
        region, off one it runs the heartbeat pass. It is a daemon because it
        must never be the reason a process lingers -- everything it does is
        decoration over work that has already finished by the time exit runs.

        ``pinned`` says the monitor outlives the builds it watches, and is how
        the live region starts it: a Live has an open phase's elapsed to tick
        long after the last build deregistered, so it -- not the watcher set --
        decides when the thread stops.

        Idempotent, and safe to call from any thread: the first caller wins and
        a later one only raises the pin.
        """
        with self._monitor_lock:
            self._monitor_pinned = self._monitor_pinned or pinned
            if self._monitor is not None:
                return
            self._monitor_stop = threading.Event()
            self._monitor = threading.Thread(
                target=self._monitor_loop,
                args=(self._monitor_stop,),
                name=_MONITOR_THREAD_NAME,
                daemon=True,
            )
            self._monitor.start()

    def _stop_monitor(self) -> None:
        """Stop the monitor and JOIN it, so nothing of it can print afterwards.

        Joining is the whole point rather than a nicety: a thread merely asked
        to stop can still be mid-``emit``, and a heartbeat that lands after a
        ``✗`` line reads as a build that failed and then kept building. Callers
        are entitled to assume that when this returns, the monitor is silent.

        Idempotent. Called from the monitor thread itself it still signals the
        stop and drops the handle; only the join is skipped, because a thread
        cannot join itself.
        """
        with self._monitor_lock:
            monitor, self._monitor = self._monitor, None
            self._monitor_pinned = False
            self._monitor_stop.set()
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=_MONITOR_JOIN_TIMEOUT)

    def _monitor_loop(self, stop: threading.Event) -> None:
        """Tick until stopped, sleeping on ``stop`` so a stop is felt at once.

        The event is passed in rather than read off ``self``, so that a monitor
        restarted while this one is on its way out cannot be stopped by its
        predecessor's event or vice versa.

        The interval is read per tick, which is what lets a test compress a
        quarter-hour build into milliseconds.
        """
        while not stop.wait(_MONITOR_INTERVAL):
            self._monitor_tick()

    def _monitor_tick(self) -> None:
        """One pass: repaint the live region, or emit the heartbeats due now.

        Exactly one of the two -- a TTY that already shows a build table moving
        does not also need lines saying it moved. Note that
        :meth:`_heartbeat_pass` is what advances the beat state, so its lines
        are emitted here and never dropped.
        """
        if self.refresh_live():
            return
        for line in self._heartbeat_pass(time.monotonic()):
            self.emit(line)

    def refresh_live(self) -> bool:
        """Repaint the live region, reporting whether there was one to repaint.

        False on this class: the base reporter prints plain lines and owns no
        live region, so the monitor falls through to heartbeats. The TTY
        renderer mounts a Rich ``Live`` and returns True from here, which is
        what silences the heartbeat lines on a terminal.
        """
        return False

    def start_rendering(self) -> None:
        """Take over the terminal, if this reporter has anything to take it for.

        Nothing on this class: a plain-line reporter owns no region and starts
        its monitor at the first watcher instead. Defined here so that the
        install site can call it on whatever it just installed -- the same
        reason :meth:`stop_rendering` is defined here -- and so that the pair
        reads as one seam in the subclass that has one: everything
        :meth:`LiveReporter.stop_rendering` tears down is exactly what
        :meth:`LiveReporter.start_rendering` puts back.

        Idempotent.
        """

    def stop_rendering(self) -> None:
        """Quiesce everything that moves, before a final plain line is printed.

        The teardown seam, called from :meth:`Phase.fail`, :meth:`Phase.
        interrupted` and :func:`install_reporter`. Order matters for whoever
        extends it: the monitor stops first, then the live region -- stopping
        the region while a tick could still repaint it would put the cursor
        back where the tick expects it and garble the line that follows.
        """
        self._stop_monitor()

    def is_rendering(self) -> bool:
        """Whether this reporter is currently drawing something transient.

        False on this class: plain lines are written and forgotten, and there is
        nothing an interactive prompt could need paused. This is what makes
        :meth:`suspended` a true no-op off a terminal -- it puts back exactly
        what it took away, and so can never start something that was not running
        when the block opened.

        Public because it answers a question outside this module too:
        :mod:`osprey.cli.output` routes a trouble line onto the region's console
        for as long as one is up, and asks here rather than keeping a second
        answer of its own.
        """
        return False

    @contextmanager
    def suspended(self) -> Iterator[None]:
        """Give the terminal back for as long as the block runs.

        For interactive prompts -- the ``.env`` seed confirmation, reset's typed
        destruction confirmation. A prompt asks its question and waits on the
        same terminal the region repaints, so a region left mounted redraws over
        the question while the operator is reading it.

        Restores on the way out however the block ends. A prompt the operator
        Ctrl-Cs is exactly the one that must not leave the rest of the verb
        rendering nothing.

        REENTRANT: nested blocks are counted and only the outermost one acts. An
        inner block restarting on its own exit would remount the region on top
        of the prompt the outer block is still waiting on.

        Off a terminal this is a genuine no-op: nothing is rendering, so nothing
        is stopped -- in particular the monitor thread that heartbeats a build
        keeps running, since off a TTY it belongs to the watcher set and not to
        this seam. Same after a repaint has degraded the region: a run that has
        fallen back to plain lines stays fallen back, rather than being handed a
        fresh region by the next prompt.

        After :meth:`hand_off` the exit remounts nothing either -- the restart
        goes through :meth:`start_rendering`, permanently a no-op by then.
        """
        self._suspend_depth += 1
        # Whether the OUTERMOST block found something to pause. Held as a local
        # so the restart is conditional on what this block actually stopped.
        resume = self._suspend_depth == 1 and self.is_rendering()
        if resume:
            self.stop_rendering()
        try:
            yield
        finally:
            self._suspend_depth -= 1
            if resume and self._suspend_depth == 0:
                self.start_rendering()

    def hand_off(self) -> None:
        """Give the terminal up for good -- someone else is about to own it.

        A non-detached ``up`` (and ``restart``) finishes by ``os.execvpe``\\
        -replacing this process with compose: a moment later the process that
        would have taken the region down does not exist, and a Live still
        mounted at that point leaves the cursor hidden and its last frame
        half-drawn under compose's own output.

        So everything that moves stops and joins here, and the open phase's
        transient state is committed as a permanent line -- the region showing
        it is about to vanish with nothing to replace it.

        The reporter then degrades to plain lines PERMANENTLY. That last part is
        not tidiness: ``os.execvpe`` can raise (a missing compose binary is a
        ``FileNotFoundError``), and the :meth:`Phase.fail` that follows runs in a
        process whose terminal has already been given away. It must print its
        ``✗`` and nothing else -- never mount a second region over output that
        is no longer this process's to draw on.

        Idempotent, and a no-op where there is nothing to hand over (the plain
        reporter off a terminal, :class:`NullReporter`): the call sites invoke it
        on whichever reporter the verb installed, without asking which it was.
        """
        if self._handed_off:
            return
        self._handed_off = True
        self.stop_rendering()
        self._commit_open_phase()

    def _commit_open_phase(self) -> None:
        """Write out anything about the open phase that was only ever drawn.

        Nothing on this class: a plain-line reporter has already printed
        everything it knows, so a hand-off here loses nothing to commit.
        """

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


def _heartbeat_line(row: BuildRow, now: float) -> str:
    """One service's heartbeat, in the sub-step shape the phases already use.

    ``  · va 6/8 Downloading torch... (running 4m00s)``. The step is absent
    before BuildKit has named one, and is a header word (``internal``,
    ``exporting``) as often as a Dockerfile index, so the parts are joined
    rather than laid out in fixed columns.

    The state and the duration ride together in the trailing parenthetical the
    surface already speaks -- ``  · step (lap)``, ``  ✓ title (elapsed)``. A
    reader who has seen one line of this output has already learned where to
    look for a time, and this line says only that the time is still running.
    """
    parts = (row.service, row.step, _clip(row.caption))
    head = " ".join(part for part in parts if part)
    return f"  · {head} (running {format_elapsed(now - row.started_at)})"


def _clip(caption: str) -> str:
    """Shorten a caption to something that reads as one line.

    Progress captions (``Downloading torch...``) are already short and pass
    through untouched; a step's opening ``RUN`` command is not.
    """
    if len(caption) <= _CAPTION_LIMIT:
        return caption
    return caption[: _CAPTION_LIMIT - 1].rstrip() + "…"


class LiveReporter(PhaseReporter):
    """The terminal reporter: the same plain lines, over a live region.

    Installed in place of :class:`PhaseReporter` when stdout is a terminal. The
    lines do not change -- ``phase``, ``step``, ``done`` and ``fail`` still go
    through :meth:`emit` and still land in the scrollback permanently, in the
    same words. What this adds is one Rich :class:`~rich.live.Live` for the
    whole verb, repainted by the monitor thread with the open phase's spinner
    and ticker and a row per service while a build runs.

    The region is transient and holds nothing of its own: everything worth
    keeping was printed as a line before it was ever drawn. That is what makes
    the teardown paths safe -- a failure, a Ctrl-C or simply the end of the verb
    takes the region down and leaves output identical to a run that never had
    one.
    """

    def __init__(self, *, console: Console | None = None) -> None:
        # Only ever installed on a terminal, so the lines are always styled.
        super().__init__(color=True)
        # Read off the module at construction rather than from-imported at
        # module load, so a caller that swapped ``styles.console`` (tests do) is
        # what the Live is built on. A theme change needs no such care:
        # ``set_theme`` re-themes the module consoles in place and they keep
        # their identity for the process lifetime.
        self._console = console if console is not None else styles.console
        self._live: Live | None = None
        # The log handlers whose console this reporter has borrowed, each with
        # the console object it is owed back. Empty whenever no region is up.
        self._borrowed_log_consoles: list[tuple[RichHandler, Console]] = []

    def out(self) -> Console:
        """The console the ``Live`` was built on -- lines and region alike."""
        return self._console

    def emit(self, text: str, style: str | None = None) -> None:
        """Print one line, with the style riding on the TEXT.

        A ``style=`` argument applies to everything the print call renders --
        and while a region is mounted Rich appends it to that same call, so a
        green ``✓`` line would repaint the whole table green underneath it. A
        pre-styled :class:`~rich.text.Text` colors its own words and nothing
        else.
        """
        self.out().print(Text(text, style=(style or "") if self.color else ""), soft_wrap=True)

    def clear_phase(self, phase: Phase) -> None:
        """Drop the phase, and take its ticker off the screen with it.

        Repainted here rather than left to the next tick because this is the
        one state change an operator would see lag: the closing ``✓`` line
        prints immediately below a region still ticking away for the phase it
        just reported finished.
        """
        super().clear_phase(phase)
        self.refresh_live()

    def start_rendering(self) -> None:
        """Mount the region and start the thread that repaints it.

        ``auto_refresh=False`` is load-bearing: the monitor thread is the SOLE
        refresher, and Rich's own refresh thread would race it for the cursor.
        ``redirect_stdout`` proxies raw ``sys.stdout`` writes through the
        console, so a helper that has never heard of the reporter still prints
        above the region instead of into it.

        The monitor is pinned here: the region outlives the builds it shows, so
        the last build deregistering ends the table and not the ticker over it.

        The logging handler is lent this console too, and given its own back when
        the region comes down -- see :meth:`_borrow_log_console`.

        Idempotent. The verb that installs the reporter owns the region, and a
        verb chained inside it (``init --up``) must not mount a second one.

        Permanently a no-op after :meth:`hand_off`: the terminal belongs to
        another process from then on, and every later call -- the exit of a
        ``suspended()`` block that spanned the hand-off, an install site
        starting the reporter it just swapped in -- must leave it alone.
        """
        if self._handed_off or self._live is not None:
            return
        live = Live(
            # An explicitly empty region, not Rich's default of ``""`` -- which
            # renders as one blank line the first repaint would then have to
            # take back.
            Group(),
            console=self._console,
            auto_refresh=False,
            redirect_stdout=True,
            transient=True,
        )
        live.start()
        self._live = live
        self._borrow_log_console()
        self._start_monitor(pinned=True)

    def stop_rendering(self) -> None:
        """Stop the monitor, THEN the region -- in that order, always.

        Reversing it leaves a tick free to repaint a region that is no longer
        there, restoring the cursor onto the row the next plain line is about to
        be written on. Call sites get the ordering by calling one method rather
        than by remembering two: the base class stops the monitor, this override
        adds the region, and inheritance keeps the sequence in one place.
        """
        super().stop_rendering()
        self._close_live()

    def is_rendering(self) -> bool:
        """True exactly while a region is mounted.

        False again once a repaint has raised and taken the region down, which
        is what keeps a run that degraded to plain lines degraded: the next
        prompt pauses nothing and therefore restarts nothing.
        """
        return self._live is not None

    def _commit_open_phase(self) -> None:
        """Leave the phase's elapsed in the scrollback as the region vanishes.

        An attached ``up`` is replaced by compose while its start phase is still
        open, so that phase never prints a ``✓`` -- the only word on it was the
        ticker in the region, which the hand-off erases mid-frame. This puts
        that reading back as a permanent line, in the same shape a heartbeat
        uses off a terminal, and for the same reason: the phase has not
        finished, it has been taken over.

        Printed after the region is down, so it lands as plain scrollback with
        nothing repainting beneath it.
        """
        phase = self._phase
        if phase is None:
            return
        self.emit(f"  · {phase.title} (running {format_elapsed(phase.elapsed)})")

    def refresh_live(self) -> bool:
        """Repaint the region from one clock reading; True while it is mounted.

        True is also what silences the heartbeat lines here: the monitor does a
        repaint or a heartbeat pass per tick, never both, and a table visibly
        moving does not also need lines saying that it moved.

        A repaint that raises takes the region down instead of the monitor with
        it. The region is decoration over a deploy that is still running, and an
        operator would rather lose the decoration than lose the phase lines and
        heartbeats behind it -- so the next tick finds no Live, answers False,
        and the run finishes as a plain-line run.
        """
        live = self._live
        if live is None:
            return False
        now = time.monotonic()
        phase = self._phase
        try:
            live.update(
                render_live_region(
                    self._watched_rows(),
                    phase_open=phase is not None,
                    elapsed=phase.elapsed_at(now) if phase is not None else 0.0,
                    now=now,
                    width=self._console.width,
                ),
                refresh=True,
            )
        except Exception:
            self._close_live()
            return False
        return True

    def _close_live(self) -> None:
        """Take the region down once, whatever state it is in.

        Blanked before it is stopped because stopping repaints one last time:
        without this the frame the failure interrupted would be drawn again and
        then erased, a flicker at exactly the moment the failure line is due.

        Idempotent, and deliberately forgiving. This runs on the failure and
        interrupt paths, and on a terminal that has stopped accepting writes
        (``osprey up | head``) the region must not turn a failed build into a
        traceback about a cursor.
        """
        live, self._live = self._live, None
        if live is None:
            return
        try:
            live.update(Group(), refresh=False)
            live.stop()
        except Exception:
            pass
        finally:
            # Last, so that a record arriving while the region is coming down
            # still goes through the console that knows a region is there.
            self._return_log_console()

    def _borrow_log_console(self) -> None:
        """Print log records through the region's console, for as long as it is up.

        A record written straight to stderr lands in the middle of a region that
        is repainting itself and the two interleave into one garbled line. Rich
        already knows how to put a line above a mounted ``Live`` -- but only for
        the console that ``Live`` was built on, so the handler is lent that
        console rather than having its stream redirected. (``Live(redirect_stderr
        =True)`` cannot do this job: a stderr-bound Console unwraps the proxy
        back to the real stream, by design.)

        Only ever on a terminal, because only this class mounts a region: off one
        the handler keeps the stderr it documents, which is what piped and CI
        consumers read.

        A root logger with no ``RichHandler`` is normal and not an error -- a
        library caller or a test may never have configured logging, and then
        there is nothing writing to stderr to garble anything.
        """
        self._borrowed_log_consoles = [
            (handler, handler.console)
            for handler in logging.getLogger().handlers
            if isinstance(handler, RichHandler)
        ]
        for handler, _ in self._borrowed_log_consoles:
            handler.console = self._console

    def _return_log_console(self) -> None:
        """Give each handler back the console OBJECT it started with.

        Identity, not an equivalent rebuilt from the same arguments: the handler's
        console is configured by whoever installed it (stderr, ``force_terminal``,
        a fixed width), and a verb that drew a region must leave logging exactly
        as it found it -- including for the process that keeps running afterwards.

        Idempotent, and empty whenever nothing was borrowed.
        """
        for handler, original in self._borrowed_log_consoles:
            handler.console = original
        self._borrowed_log_consoles = []


class NullReporter(PhaseReporter):
    """No-op reporter installed under the global ``--verbose`` flag."""

    def __init__(self, *, verbose: bool = False) -> None:
        super().__init__(color=False)
        self.verbose = verbose

    def emit(self, text: str, style: str | None = None) -> None:
        """Swallow the line -- verbose mode streams the real output instead."""

    def replay(self, path: Path | None) -> None:
        """Nothing to replay: the output already went to the terminal."""

    def _start_monitor(self, *, pinned: bool = False) -> None:
        """Run no thread at all: every line it could produce is swallowed.

        Under ``--verbose`` the child's own output streams to the terminal, so
        there is nothing for a heartbeat to prove and no live region to repaint
        -- a monitor here would wake four times a second to discard its work.
        """


_reporter: PhaseReporter = NullReporter()


def current_reporter() -> PhaseReporter:
    """Return the installed reporter (a quiet :class:`NullReporter` default)."""
    return _reporter


def install_reporter(reporter: PhaseReporter) -> PhaseReporter:
    """Install ``reporter`` as the active one and return the previous one.

    The swap itself quiesces the outgoing reporter. Only one reporter may own
    the terminal at a time, and the handover is the one moment every path
    through the code has in common: the verbs' ``finally`` in ``cli/main.py``,
    but equally a test fixture or a library caller that swaps a reporter in and
    back out by hand. Leaving that to the call sites would mean a fixture
    forgetting it leaks a live thread into whatever runs next -- where it shows
    up much later as an unrelated flaky test.
    """
    global _reporter
    previous, _reporter = _reporter, reporter
    if previous is not reporter:
        previous.stop_rendering()
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
