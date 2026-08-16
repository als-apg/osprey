"""Shared fixtures for the ``osprey`` CLI test suite.

CLI tests drive Click commands in-process through ``CliRunner``, so whatever a
command does to process-global state happens inside the pytest process itself.
The guards here keep that blast radius contained.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

# The gold-standard three-zone deployment repo, re-exported so every lifecycle
# test under tests/cli/ can ask for it by name. Defined in tests/fixtures/ with
# the exemplar content it materializes.
from tests.fixtures.lifecycle_repo import (  # noqa: F401
    lifecycle_repo,
    lifecycle_repo_factory,
)


@pytest.fixture(autouse=True)
def _guard_os_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn an in-process ``os._exit`` into an ordinary test failure.

    Leak guarded: ``osprey health`` falls back to ``os._exit`` when a sync check
    was abandoned after a timeout (``src/osprey/cli/health_cmd.py:320-326``),
    because a normal ``sys.exit`` can wedge on interpreter teardown with a
    daemon thread still running. Under an in-process ``CliRunner`` invocation
    that call terminates pytest itself; under xdist it kills the whole worker
    and every test still queued on it, surfacing as a crashed node rather than
    a failing test. Raising instead keeps the failure attributable.

    The real daemon-thread/``os._exit`` guarantee is pinned by the
    real-subprocess no-hang test in ``test_health_cmd.py``, which is unaffected:
    this patch lives in the parent process, not the child's.
    """

    def _raise(code: int) -> None:
        raise AssertionError(f"os._exit({code}) called in-process")

    monkeypatch.setattr(os, "_exit", _raise)


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` and ``$HOME`` to a tmp directory.

    Leak guarded: CLI code that writes under the home directory (skills install,
    template scaffolding, hook registration) would otherwise touch the
    developer's real ``~``. Patching the method on ``Path`` is required because
    callers use ``Path.home()`` directly; ``$HOME`` is patched alongside it for
    subprocess-style code paths that read the environment instead.

    Returns:
        The tmp directory standing in for ``$HOME``.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    return home


# ===================================================================
# Terminal probe — what the operator's terminal actually shows
# ===================================================================


class _ProbeRecordSink(logging.Handler):
    """Every record the loggers emitted, captured before any handler filter."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TerminalProbe:
    """Both halves of a rendering question: what was painted, what was emitted.

    ``rendered_text`` is what the root ``RichHandler`` actually painted — the
    operator's terminal, after the altitude gate
    (``osprey.cli.altitude``) has had its say. ``records`` is the raw record
    stream reaching the root logger, which the gate never touches: it is a
    handler filter, so a record it drops is still emitted and still observable.
    Holding the two side by side is the only way to state the honest form of an
    altitude claim — *this fact no longer clutters the terminal, and nothing was
    lost.*

    Attributes:
        console: The Rich console the handler paints through, writing into an
            in-memory buffer.
    """

    def __init__(self, console: Console, buffer: StringIO, sink: _ProbeRecordSink) -> None:
        self.console = console
        self._buffer = buffer
        self._sink = sink

    @property
    def styled_text(self) -> str:
        """Painted output with its ANSI escape sequences intact.

        Use for style assertions (``"\\x1b[1m" in probe.styled_text``); use
        :attr:`rendered_text` for anything about the words themselves.
        """
        return self._buffer.getvalue()

    @property
    def rendered_text(self) -> str:
        """Painted output as plain text, ANSI escape sequences removed.

        The needle surface: ``assert "the fact" in probe.rendered_text``. Lines
        are padded out to the console width, so prefer substring checks or
        :attr:`lines` over comparing a whole line for equality.
        """
        return Text.from_ansi(self.styled_text).plain

    @property
    def lines(self) -> list[str]:
        """:attr:`rendered_text` split into lines, each right-stripped."""
        return [line.rstrip() for line in self.rendered_text.splitlines()]

    @property
    def records(self) -> list[logging.LogRecord]:
        """Every record that reached the root logger, in emission order."""
        return list(self._sink.records)

    @property
    def messages(self) -> list[str]:
        """The formatted message of every captured record."""
        return [record.getMessage() for record in self._sink.records]

    def arm(self) -> RichHandler:
        """Point the current root ``RichHandler`` at the probe console again.

        Only needed by a test that installs logging itself in a way the fixture
        cannot intercept — the fixture already covers the ordinary paths
        (a handler that existed beforehand, ``configure_logging()`` called from
        the test body, and the CLI's own ``set_handler_console()``).

        Returns:
            The handler now painting into the probe buffer.
        """
        handler = _first_rich_handler()
        if handler is None:
            handler = _build_probe_handler(self.console)
            logging.getLogger().addHandler(handler)
        else:
            handler.console = self.console
        return handler

    def clear(self) -> None:
        """Drop everything captured so far, painted and emitted alike."""
        self._buffer.seek(0)
        self._buffer.truncate(0)
        self._sink.records.clear()


def _first_rich_handler() -> RichHandler | None:
    """The root handler the CLI configures and gates, or None if there is none.

    Same predicate ``configure_logging``, ``set_handler_console`` and
    ``osprey.cli.altitude`` use, so all four agree on which handler is *the*
    handler.
    """
    for handler in logging.getLogger().handlers:
        if isinstance(handler, RichHandler):
            return handler
    return None


def _build_probe_handler(console: Console) -> RichHandler:
    """A stand-in for the handler ``configure_logging()`` would have installed.

    Mirrors ``osprey_connectors.logger._build_rich_handler``'s secure defaults
    without its config lookup, which would pull the config machinery — and its
    ``.env`` load — into a fixture.
    """
    return RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=True,
        show_path=False,
        show_time=True,
        show_level=True,
        tracebacks_show_locals=False,
    )


@pytest.fixture
def terminal_probe(
    monkeypatch: pytest.MonkeyPatch,
    restore_root_logging: None,
) -> Iterator[TerminalProbe]:
    """Capture what the terminal shows *and* what the loggers emitted.

    The instrument for altitude and promotion claims. The gate the CLI installs
    is a filter on the root ``RichHandler``, so a record it drops still reaches
    every record-level observer — ``caplog``, a record sink, ``debug_probe``.
    Those instruments structurally cannot see a handler gate; this one can,
    because it owns the console the handler paints through.

    Which instrument to reach for:

    * **This fixture** for anything about the *rendered* terminal: a promoted
      fact appearing in the default view, its old log form no longer appearing,
      or a WARNING still getting through. Both halves of that claim come from
      one object, so the "and nothing was lost" half cannot be forgotten.
    * **``debug_probe``** (``tests/cli/test_lifecycle_phases.py``) for anything
      about *emission* alone: whether a demoted call site logs at all under the
      level a flag chose. It emits from inside a running verb and reads the
      level the group callback applied, which is a different link in the chain
      from what the handler paints.
    * **``caplog``** stays fine for ordinary "was this logged, with what
      message" assertions that make no claim about the terminal.

    Console settings, and what they let you assert:

    * ``width=100`` — a fixed width, so wrapping is reproducible across
      machines. Long facts do wrap; assert on a distinctive fragment.
    * ``force_terminal=True`` with the CLI's own theme — the handler paints
      exactly the styled output a real terminal gets, theme style names
      (``[success]``, ``[warning]``…) resolve instead of raising, and
      :attr:`~TerminalProbe.styled_text` exposes the escape sequences. Plain
      assertions stay safe because :attr:`~TerminalProbe.rendered_text` strips
      them back off.

    Interplay with a CLI invocation:

        def test_the_fact_is_in_the_default_view(terminal_probe, monkeypatch):
            # The group callback loads an ancestor .env straight into os.environ.
            monkeypatch.setattr(
                "osprey.utils.config.load_project_dotenv", lambda *a, **k: None
            )
            result = CliRunner().invoke(cli, ["up"])

            assert "deployment is browse-only" in result.output       # printed
            assert "browse-only" not in terminal_probe.rendered_text  # not logged
            assert any("browse-only" in m for m in terminal_probe.messages)

    The CLI points the handler at its own stderr console; the fixture wraps
    ``set_handler_console`` so the probe console wins instead, for the whole
    run rather than only afterwards. That splits the two streams cleanly:
    printed output (echo-class ``report()``/``section()`` lines) lands in
    ``CliRunner``'s ``result.output``, while anything the *logging* path painted
    lands in ``rendered_text``. Invoking the CLI before or after asking for the
    probe both work.

    Note:
        A mounted live region borrows the handler's console and restores it by
        identity (``PhaseReporter._borrow_log_console()``), so records logged
        while a progress display is up are painted by the reporter, not here.
        That is the production behaviour, not a fixture artefact.

    Note:
        The root logger's level is left alone. A DEBUG record under a default
        (INFO) run is therefore absent from ``records`` too — which is the
        truth about that run, and what makes a real demotion detectable.

    Yields:
        The :class:`TerminalProbe` for the test body.
    """
    # Imported here, not at module scope: a conftest import runs for every test
    # under tests/cli/, and the theme module pulls in the CLI package.
    from osprey.cli.styles import osprey_theme

    buffer = StringIO()
    console = Console(
        file=buffer,
        width=100,
        force_terminal=True,
        color_system="truecolor",
        legacy_windows=False,
        theme=osprey_theme,
    )

    root = logging.getLogger()
    sink = _ProbeRecordSink()
    root.addHandler(sink)

    existing = _first_rich_handler()
    previous_console = existing.console if existing is not None else None
    if existing is not None:
        existing.console = console
        probe_handler = existing
        probe_owns_handler = False
    else:
        # Installed up front so a later ``configure_logging()`` — which only
        # adds a handler when the root logger has no RichHandler — keeps this
        # one instead of building a stderr console the probe cannot see.
        probe_handler = _build_probe_handler(console)
        root.addHandler(probe_handler)
        probe_owns_handler = True

    import osprey_connectors.logger as logger_mod

    real_set_handler_console = logger_mod.set_handler_console

    def _set_probe_console(_console: Console) -> RichHandler | None:
        """Redirect the caller's console to the probe's, then behave normally."""
        handler: RichHandler | None = real_set_handler_console(console)
        return handler

    monkeypatch.setattr(logger_mod, "set_handler_console", _set_probe_console)

    try:
        yield TerminalProbe(console, buffer, sink)
    finally:
        root.removeHandler(sink)
        if probe_owns_handler:
            root.removeHandler(probe_handler)
        elif previous_console is not None:
            probe_handler.console = previous_console
