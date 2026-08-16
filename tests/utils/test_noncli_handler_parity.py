"""Regression instrument: non-CLI consumers of ``osprey_connectors`` see no change.

The CLI-output work touched two things in the shared logging module: every
record now carries an ``osprey_intent`` attribute, and ``_build_rich_handler()``
accepts an injected console. Everything else that calls ``configure_logging()``
— service startups, the dispatch worker, MCP servers, and any external consumer
of the published package — passes no console and must keep getting byte-for-byte
the terminal output it got before.

Two instruments pin that, both on a frozen clock so timestamps are deterministic
and both rendering into a ``StringIO`` console:

(a) **Render-inertness of the intent attribute** — one handler fed the same
    records with and without ``osprey_intent`` produces identical bytes, both
    for hand-built records and for the records a real ``ComponentLogger``
    emits.
(b) **Builder parity** — ``_build_rich_handler()`` with no injected console is
    structurally identical to, and renders identically to, an inline frozen
    replica of the pre-feature builder embedded in this file.

The replica below is the baseline. It is a deliberate copy, not a call into the
current module: if the shipped builder drifts, only the copy still describes the
behavior external consumers were promised, and these tests fail.
"""

from __future__ import annotations

import io
import logging
import re
import sys
from datetime import datetime
from typing import Literal, cast

from rich.console import Console
from rich.logging import RichHandler

from osprey.utils.logger import ComponentLogger, _build_rich_handler
from osprey_connectors.config import get_config_value

#: Fixed wall-clock seed for every record. ``RichHandler.render`` derives the
#: displayed timestamp from ``record.created`` alone, so pinning that field is
#: the whole clock freeze — no patching of ``time`` is needed.
FROZEN_EPOCH: float = datetime(2026, 1, 2, 3, 4, 5).timestamp()

#: Level, intent name, and message for the shared record set. Several levels are
#: covered, and one message carries Rich markup plus a bracketed literal that is
#: *not* a tag — the handler is built with ``markup=True``, so markup handling is
#: part of what has to stay identical.
RECORD_SPECS: tuple[tuple[int, str, str], ...] = (
    (logging.DEBUG, "debug", "Resolving handler chain"),
    (logging.INFO, "info", "Connected to 3 channels"),
    (logging.INFO, "key_info", "Profile: als-apg"),
    (logging.INFO, "status", "[bold cyan]Deploying[/bold cyan] stack [1/3]"),
    (logging.WARNING, "warning", "Retrying in 5s"),
    (logging.ERROR, "error", "Write refused for SR:BPM1"),
)


def _prefeature_build_rich_handler() -> RichHandler:
    """Inline frozen replica of ``_build_rich_handler`` as it stood pre-feature.

    Copied from the shipped builder before the console parameter existed. It is
    the baseline these tests compare against, so it must never be rewritten to
    delegate to the current implementation — that would make the comparison
    vacuous.

    Returns:
        A ``RichHandler`` configured exactly as non-CLI entry points used to get.
    """
    try:
        # Security-conscious defaults: hide locals to prevent sensitive data exposure
        rich_tracebacks = get_config_value("logging.rich_tracebacks", True)
        show_traceback_locals = get_config_value("logging.show_traceback_locals", False)
        show_full_paths = get_config_value("logging.show_full_paths", False)
    except Exception:
        rich_tracebacks = True
        show_traceback_locals = False
        show_full_paths = False

    # force_terminal keeps colors in containers and CI, where stderr is a pipe.
    console = Console(
        stderr=True,
        force_terminal=True,
        width=120,
        color_system="truecolor",
    )

    return RichHandler(
        console=console,
        rich_tracebacks=rich_tracebacks,
        markup=True,
        show_path=show_full_paths,
        show_time=True,
        show_level=True,
        tracebacks_show_locals=show_traceback_locals,
    )


def _make_record(level: int, message: str, index: int, intent: str | None) -> logging.LogRecord:
    """Build one fully deterministic record.

    Args:
        level: Standard library level for the record.
        message: Already-formatted message text.
        index: Position in the set; each record is one second after the previous
            so no two render the same timestamp (``omit_repeated_times`` blanks
            a repeat, which would make the comparison depend on ordering).
        intent: Value for ``osprey_intent``, or ``None`` to leave the attribute
            off entirely — the pre-feature shape.

    Returns:
        A ``LogRecord`` whose every render-visible field is fixed.
    """
    record = logging.LogRecord(
        name="osprey.parity",
        level=level,
        pathname="/osprey/parity.py",
        lineno=42,
        msg=message,
        args=(),
        exc_info=None,
        func="emit",
    )
    _freeze(record, index)
    if intent is not None:
        record.osprey_intent = intent
    return record


def _freeze(record: logging.LogRecord, index: int) -> logging.LogRecord:
    """Pin every render-visible field of ``record`` that varies run to run."""
    record.created = FROZEN_EPOCH + index
    record.pathname = "/osprey/parity.py"
    record.lineno = 42
    record.name = "osprey.parity"
    record.funcName = "emit"
    return record


def _record_set(*, with_intent: bool) -> list[logging.LogRecord]:
    """The shared record set, with or without the ``osprey_intent`` attribute."""
    return [
        _make_record(level, message, index, intent if with_intent else None)
        for index, (level, intent, message) in enumerate(RECORD_SPECS)
    ]


def _render(handler: RichHandler, records: list[logging.LogRecord]) -> str:
    """Render ``records`` through ``handler`` into a string.

    The handler's real console is swapped for a ``StringIO`` twin carrying the
    same width, terminal flag, and color system, so a drift in any of those
    still shows up in the captured bytes rather than being normalized away.

    Args:
        handler: Handler under test; its console is restored before returning.
        records: Records to emit, in order.

    Returns:
        Everything the handler wrote, ANSI escapes included.
    """
    buffer = io.StringIO()
    live_console = handler.console
    handler.console = Console(
        file=buffer,
        force_terminal=live_console.is_terminal,
        width=live_console.width,
        # ``Console.color_system`` hands back the plain name it was built from.
        color_system=cast(
            "Literal['auto', 'standard', '256', 'truecolor', 'windows'] | None",
            live_console.color_system,
        ),
    )
    # LogRender remembers the last timestamp it printed; clear it so a pass never
    # depends on what an earlier pass through the same handler rendered.
    handler._log_render._last_time = None
    try:
        for record in records:
            handler.emit(record)
    finally:
        handler.console = live_console
    return buffer.getvalue()


#: ANSI CSI sequences, for the readability guards below only.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _visible(rendered: str) -> str:
    """Strip ANSI escapes so a guard can look for the text an operator reads.

    Only the "did this really render something" guards use this. The parity
    assertions themselves always compare the raw bytes, escapes included —
    a color change is exactly the kind of drift they exist to catch. Rich's
    highlighter styles numbers and brackets mid-token, so a raw substring
    search for ``[1/3]`` or ``3 channels`` would miss.
    """
    return _ANSI.sub("", rendered)


def _console_shape(console: Console) -> dict[str, object]:
    """Every console setting the pre-feature builder pinned."""
    return {
        "file": console.file,
        "stderr": console.stderr,
        "is_terminal": console.is_terminal,
        "width": console.width,
        "color_system": console.color_system,
        "no_color": console.no_color,
    }


def _handler_shape(handler: RichHandler) -> dict[str, object]:
    """Every ``RichHandler`` constructor argument observable on the instance.

    Includes the ``LogRender`` fields, which are where ``show_time``,
    ``show_level``, and ``show_path`` end up.
    """
    render = handler._log_render
    return {
        "level": handler.level,
        "markup": handler.markup,
        "rich_tracebacks": handler.rich_tracebacks,
        "enable_link_path": handler.enable_link_path,
        "highlighter": type(handler.highlighter),
        "keywords": handler.keywords,
        "locals_max_length": handler.locals_max_length,
        "locals_max_string": handler.locals_max_string,
        "tracebacks_width": handler.tracebacks_width,
        "tracebacks_code_width": handler.tracebacks_code_width,
        "tracebacks_extra_lines": handler.tracebacks_extra_lines,
        "tracebacks_theme": handler.tracebacks_theme,
        "tracebacks_word_wrap": handler.tracebacks_word_wrap,
        "tracebacks_show_locals": handler.tracebacks_show_locals,
        "tracebacks_suppress": tuple(handler.tracebacks_suppress),
        "tracebacks_max_frames": handler.tracebacks_max_frames,
        "show_time": render.show_time,
        "show_level": render.show_level,
        "show_path": render.show_path,
        "time_format": render.time_format,
        "omit_repeated_times": render.omit_repeated_times,
        "level_width": render.level_width,
    }


class TestIntentAttributeIsRenderInert:
    """(a) ``osprey_intent`` must change nothing a terminal ever sees."""

    def test_same_handler_renders_identical_bytes_with_and_without_intent(self) -> None:
        handler = _build_rich_handler()
        tagged = _record_set(with_intent=True)
        untagged = _record_set(with_intent=False)

        assert all(hasattr(record, "osprey_intent") for record in tagged)
        assert not any(hasattr(record, "osprey_intent") for record in untagged)

        rendered_tagged = _render(handler, tagged)
        rendered_untagged = _render(handler, untagged)

        # Guard against a vacuous pass on two empty strings.
        assert "Connected to 3 channels" in _visible(rendered_untagged)
        assert rendered_tagged == rendered_untagged

    def test_rendered_output_never_mentions_the_attribute(self) -> None:
        rendered = _visible(_render(_build_rich_handler(), _record_set(with_intent=True)))

        assert "osprey_intent" not in rendered
        for _level, intent, _message in RECORD_SPECS:
            # Intent names double as message-free markers; none may leak. The
            # names that also appear in level text ("debug", "warning",
            # "error") are excluded — those are the level column, not the marker.
            if intent in ("debug", "warning", "error"):
                continue
            assert intent not in rendered

    def test_markup_message_still_renders_as_markup(self) -> None:
        rendered = _visible(_render(_build_rich_handler(), _record_set(with_intent=True)))

        # markup=True: the tags are consumed, the styled words and the
        # bracketed non-tag survive verbatim.
        assert "[bold cyan]" not in rendered
        assert "Deploying" in rendered
        assert "[1/3]" in rendered

    def test_component_logger_records_render_like_plain_stdlib_records(self) -> None:
        """The production path, not just hand-built records.

        A real ``ComponentLogger`` call and a bare ``logging.Logger`` call with
        the same level and text must reach the handler as records that render
        identically — the merged ``extra`` being the only difference between
        them.
        """
        component_records = _emit_and_collect(via_component_logger=True)
        plain_records = _emit_and_collect(via_component_logger=False)

        assert all(hasattr(record, "osprey_intent") for record in component_records)
        assert not any(hasattr(record, "osprey_intent") for record in plain_records)
        assert [record.levelno for record in component_records] == [
            record.levelno for record in plain_records
        ]

        handler = _build_rich_handler()
        rendered_component = _render(handler, component_records)
        rendered_plain = _render(handler, plain_records)

        assert "Connected to 3 channels" in _visible(rendered_plain)
        assert rendered_component == rendered_plain


class TestBuilderParityWithFrozenReplica:
    """(b) The shipped builder, given no console, is the pre-feature builder."""

    def test_console_matches_the_frozen_replica(self) -> None:
        shipped = _build_rich_handler()
        replica = _prefeature_build_rich_handler()

        assert _console_shape(shipped.console) == _console_shape(replica.console)

    def test_console_still_carries_the_pre_feature_literals(self) -> None:
        console = _build_rich_handler().console

        assert console.file is sys.stderr
        assert console.stderr is True
        assert console.is_terminal is True
        assert console.width == 120
        assert console.color_system == "truecolor"

    def test_every_handler_constructor_argument_matches_the_replica(self) -> None:
        shipped = _build_rich_handler()
        replica = _prefeature_build_rich_handler()

        assert _handler_shape(shipped) == _handler_shape(replica)

    def test_handler_still_carries_the_pre_feature_literals(self) -> None:
        shape = _handler_shape(_build_rich_handler())

        assert shape["markup"] is True
        assert shape["show_time"] is True
        assert shape["show_level"] is True
        assert shape["show_path"] is False
        assert shape["tracebacks_show_locals"] is False
        assert shape["rich_tracebacks"] is True

    def test_both_handlers_render_identical_bytes(self) -> None:
        shipped = _build_rich_handler()
        replica = _prefeature_build_rich_handler()
        records = _record_set(with_intent=True)

        rendered_shipped = _render(shipped, records)
        rendered_replica = _render(replica, _record_set(with_intent=False))

        assert "Connected to 3 channels" in _visible(rendered_replica)
        assert rendered_shipped == rendered_replica


def _emit_and_collect(*, via_component_logger: bool) -> list[logging.LogRecord]:
    """Emit ``RECORD_SPECS`` through the real logging path and freeze the records.

    Args:
        via_component_logger: ``True`` drives ``ComponentLogger``'s public intent
            methods (which stamp ``osprey_intent``); ``False`` drives the bare
            ``logging.Logger`` underneath it.

    Returns:
        The captured records, with every run-varying field pinned so two sets
        differ only in the attribute under test.
    """
    which = "component" if via_component_logger else "plain"
    base = logging.getLogger(f"osprey_parity_test.{which}")
    base.setLevel(logging.DEBUG)
    base.propagate = False  # keep records off root handlers other tests own
    collected: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            collected.append(record)

    base.handlers[:] = [_Collector()]
    try:
        component = ComponentLogger(base, "parity")
        for level, intent, message in RECORD_SPECS:
            if via_component_logger:
                getattr(component, intent)(message)
            else:
                base.log(level, message)
    finally:
        base.handlers[:] = []

    for index, record in enumerate(collected):
        _freeze(record, index)
    return collected
