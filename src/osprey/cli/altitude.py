"""Altitude policy for the CLI: what a normal run renders to the terminal.

A normal ``osprey`` run should read as a report, not as a transcript. The
policy that makes that true is a single :class:`logging.Filter` installed on the
``RichHandler`` **instance** that renders log records to stderr — never inside
:func:`~osprey.utils.logger.configure_logging`, so the entry points that are not
the CLI (MCP servers, bridges, services, workers) and every external consumer of
the ``osprey-connectors`` distribution keep today's logging behavior exactly.

Gating at the handler means gating at *rendering*, not at emission: a dropped
record is still emitted, still reaches ``caplog``, and still reaches any other
handler or sink on the root logger. Nothing is lost — it is only not painted.

``-v``/``--verbose`` lifts the gate and restores the full transcript. Levels are
never remapped; the gate reads a record's level, plus one bit of CLI state —
whether a lifecycle reporter currently owns the terminal, which is what tells a
WARNING record it must arrive promoted (:func:`osprey.cli.output.warn_fact`)
rather than as a raw transcript block through a phase column.
"""

import logging

from rich.logging import RichHandler

__all__ = ["gate_installed", "install_gate", "lift_gate"]


class _AltitudeGate(logging.Filter):
    """Drop INFO-and-below records at the handler; pass WARNING and above —
    except while a lifecycle reporter owns the terminal, where WARNING is
    dropped too and only ERROR and above still paint.

    The policy is deliberately level-based. Records emitted through
    :class:`~osprey.utils.logger.ComponentLogger` also carry an
    ``osprey_intent`` attribute naming the method that emitted them, but the
    shipped policy does not read it — INFO-family intents are separated by
    rewriting their call sites as report lines, not by filtering on intent.

    The lifecycle exception is the renderer's single-voice rule. While a phase
    record is being drawn, a raw WARNING painted by the handler arrives as a
    timestamped, hard-wrapped transcript block in the middle of a column of
    phase lines — the one shape the output hierarchy exists to prevent. So a
    warning an operator must read during a lifecycle verb is *promoted* at its
    call site (:func:`osprey.cli.output.warn_fact`), exactly as INFO-family
    facts already are, and the un-promoted rest stays where it always was: in
    the emitted record, for ``caplog``, file sinks, and ``-v``. ERROR keeps
    painting unconditionally — a failure outranks the column's tidiness.

    "A lifecycle reporter owns the terminal" is read off the installed reporter
    per record, through the same predicate ``lifecycle_reporter`` uses to
    decide ownership: the quiet ``NullReporter`` default means no lifecycle
    verb is running, and every other reporter means one is.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True when this record should be rendered."""
        if record.levelno >= logging.ERROR:
            return True
        if record.levelno < logging.WARNING:
            return False
        from osprey.cli.phase_reporter import NullReporter, current_reporter

        # Either NullReporter shape means no phase column is being drawn: the
        # quiet default because no lifecycle verb is running, the verbose one
        # because ``-v`` streams the transcript instead of drawing one.
        return isinstance(current_reporter(), NullReporter)


def _root_rich_handler() -> RichHandler | None:
    """Return the first ``RichHandler`` on the root logger, if there is one.

    This is the same predicate :func:`~osprey.utils.logger.configure_logging`
    and :func:`~osprey.utils.logger.set_handler_console` use, so all three
    resolve to the same handler instance.
    """
    for handler in logging.getLogger().handlers:
        if isinstance(handler, RichHandler):
            return handler
    return None


def lift_gate(handler: logging.Handler | None = None) -> None:
    """Remove every altitude gate from ``handler``, restoring the transcript.

    Args:
        handler: Handler to lift the gate from. Defaults to the root
            ``RichHandler`` — the seam a subcommand's own ``--verbose`` uses to
            lift the gate for its own run without knowing where the gate lives.

    Note:
        This restores what the *handler* renders. The root logger's level is a
        separate floor: a normal run configures it to INFO, so lifting the gate
        alone surfaces the INFO transcript, not DEBUG records.
    """
    if handler is None:
        handler = _root_rich_handler()
    if handler is None:
        return
    for existing in list(handler.filters):
        if isinstance(existing, _AltitudeGate):
            handler.removeFilter(existing)


def install_gate(handler: logging.Handler | None = None, *, verbose: bool = False) -> None:
    """Install the altitude gate on ``handler`` for this CLI invocation.

    Self-correcting within a process: any gate left by an earlier invocation is
    removed first, then exactly one fresh gate is added — or none, under
    ``verbose``. The ``RichHandler`` outlives a single CLI invocation (the test
    suite runs the CLI many times in one process), so installing without
    removing first would stack filters.

    Args:
        handler: Handler to gate. Defaults to the root ``RichHandler``.
        verbose: When True the gate is lifted rather than installed, and the
            run renders its full transcript.
    """
    if handler is None:
        handler = _root_rich_handler()
    if handler is None:
        return
    lift_gate(handler)
    if not verbose:
        handler.addFilter(_AltitudeGate())


def gate_installed(handler: logging.Handler | None = None) -> bool:
    """Return True when an altitude gate is currently gating ``handler``."""
    if handler is None:
        handler = _root_rich_handler()
    if handler is None:
        return False
    return any(isinstance(existing, _AltitudeGate) for existing in handler.filters)
