"""Tests for the CLI altitude gate.

The gate is a filter on the stderr ``RichHandler``: in a normal ``osprey`` run
INFO-and-below records are not rendered, WARNING and above are, and ``-v`` lifts
the gate entirely. Gating happens at rendering, so a dropped record is still
emitted and still reaches ``caplog`` — that honesty half is pinned here too.
"""

import logging
from io import StringIO

import pytest
from click.testing import CliRunner
from rich.console import Console
from rich.logging import RichHandler

from osprey.cli.altitude import _AltitudeGate, gate_installed, install_gate, lift_gate
from osprey.cli.main import cli

INFO_MARKER = "INFOMARKER"
WARNING_MARKER = "WARNMARKER"


def _root_handler() -> RichHandler:
    """The stderr RichHandler the CLI gates — the same instance it installs."""
    handlers = [h for h in logging.getLogger().handlers if isinstance(h, RichHandler)]
    assert handlers, "the CLI run should have left a RichHandler on the root logger"
    return handlers[0]


def _gate_count(handler: logging.Handler) -> int:
    return sum(1 for f in handler.filters if isinstance(f, _AltitudeGate))


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch):
    """Invoke the CLI group callback the way a real run does.

    The bare command runs the whole callback (logging, handler console, gate)
    and then prints help. ``load_project_dotenv`` is stubbed out: it writes an
    ancestor ``.env`` straight into ``os.environ``, which outlives the test.
    """
    monkeypatch.setattr("osprey.utils.config.load_project_dotenv", lambda *a, **k: None)
    runner = CliRunner()

    def _run(*args: str) -> None:
        result = runner.invoke(cli, list(args))
        assert result.exit_code == 0, result.output

    return _run


@pytest.fixture
def rendered() -> StringIO:
    """Capture what the handler actually paints, after the CLI has configured it."""
    return StringIO()


def _capture_renders(buffer: StringIO) -> RichHandler:
    """Point the CLI's handler at ``buffer`` without touching its filters."""
    handler = _root_handler()
    handler.console = Console(file=buffer, width=200, force_terminal=False)
    return handler


class TestGateFilter:
    """The filter itself: the record's level, plus one bit of reporter state.

    These direct tests run under the quiet ``NullReporter`` default, which is
    the no-lifecycle-verb state: WARNING and above pass. The lifecycle state is
    ``TestSingleVoiceDuringLifecycleVerbs``' subject.
    """

    @pytest.mark.parametrize(
        ("level", "passes"),
        [
            (logging.DEBUG, False),
            (logging.INFO, False),
            (logging.WARNING, True),
            (logging.ERROR, True),
            (logging.CRITICAL, True),
        ],
    )
    def test_levels(self, level: int, passes: bool):
        record = logging.LogRecord("osprey.test", level, __file__, 1, "msg", None, None)
        assert _AltitudeGate().filter(record) is passes

    def test_record_without_intent_attribute_transits(self):
        """Bare logging and third-party records carry no ``osprey_intent``."""
        record = logging.LogRecord("httpx", logging.WARNING, __file__, 1, "msg", None, None)
        assert not hasattr(record, "osprey_intent")
        assert _AltitudeGate().filter(record) is True

    def test_intent_attribute_does_not_change_the_verdict(self):
        """The shipped policy reads the level, never the intent marker."""
        gated = logging.LogRecord("osprey.test", logging.INFO, __file__, 1, "msg", None, None)
        gated.osprey_intent = "key_info"  # type: ignore[attr-defined]
        assert _AltitudeGate().filter(gated) is False


class TestSingleVoiceDuringLifecycleVerbs:
    """While a reporter owns the terminal, a WARNING arrives promoted or not at all.

    A raw WARNING painted through a phase column is the timestamped transcript
    block the output hierarchy exists to prevent, so the gate drops it there —
    the operator-facing copy comes from ``osprey.cli.output.warn_fact`` at the
    call sites that owe one. ERROR outranks the column and always paints.
    """

    @pytest.fixture
    def lifecycle_reporter_installed(self):
        from osprey.cli.phase_reporter import PhaseReporter, current_reporter, install_reporter

        previous = current_reporter()
        install_reporter(PhaseReporter(color=False))
        yield
        install_reporter(previous)

    def _record(self, level: int) -> logging.LogRecord:
        return logging.LogRecord("osprey.test", level, __file__, 1, "msg", None, None)

    def test_warning_is_dropped_while_a_reporter_is_installed(self, lifecycle_reporter_installed):
        assert _AltitudeGate().filter(self._record(logging.WARNING)) is False

    def test_error_still_paints_while_a_reporter_is_installed(self, lifecycle_reporter_installed):
        assert _AltitudeGate().filter(self._record(logging.ERROR)) is True

    def test_warning_paints_again_once_the_reporter_is_gone(self):
        """The uninstall on the verb's way out is what restores the old policy."""
        assert _AltitudeGate().filter(self._record(logging.WARNING)) is True

    def test_the_verbose_null_reporter_counts_as_no_lifecycle_rendering(self):
        """Under ``-v`` nothing draws a phase column, so nothing needs protecting.

        The gate is normally not even installed under ``-v``; this pins the
        harmless answer for the state where one is.
        """
        from osprey.cli.phase_reporter import NullReporter, current_reporter, install_reporter

        previous = current_reporter()
        install_reporter(NullReporter(verbose=True))
        try:
            assert _AltitudeGate().filter(self._record(logging.WARNING)) is True
        finally:
            install_reporter(previous)


class TestNormalRun:
    """A plain ``osprey`` invocation renders a report, not a transcript."""

    def test_gate_is_installed(self, run_cli):
        run_cli()
        assert gate_installed(_root_handler()) is True

    def test_info_is_not_rendered(self, run_cli, rendered: StringIO):
        run_cli()
        _capture_renders(rendered)

        logging.getLogger("osprey.altitude_probe").info(INFO_MARKER)

        assert INFO_MARKER not in rendered.getvalue()

    def test_warning_and_above_are_rendered(self, run_cli, rendered: StringIO):
        run_cli()
        _capture_renders(rendered)

        logger = logging.getLogger("osprey.altitude_probe")
        logger.warning(WARNING_MARKER)
        logger.error("ERRORMARKER")

        painted = rendered.getvalue()
        assert WARNING_MARKER in painted
        assert "ERRORMARKER" in painted

    def test_bare_logger_records_transit_the_gate(self, run_cli, rendered: StringIO):
        """Records with no ``osprey_intent`` are gated by level, not dropped."""
        run_cli()
        _capture_renders(rendered)

        bare = logging.getLogger("third_party_probe")
        bare.info(INFO_MARKER)
        bare.warning(WARNING_MARKER)

        painted = rendered.getvalue()
        assert INFO_MARKER not in painted
        assert WARNING_MARKER in painted

    def test_gated_records_still_reach_caplog(self, run_cli, rendered: StringIO, caplog):
        """Nothing is lost: the gate stops rendering, not emission."""
        run_cli()
        _capture_renders(rendered)

        with caplog.at_level(logging.INFO, logger="osprey.altitude_probe"):
            logging.getLogger("osprey.altitude_probe").info(INFO_MARKER)

        assert gate_installed(_root_handler()) is True
        assert INFO_MARKER in caplog.text
        assert INFO_MARKER not in rendered.getvalue()


class TestVerboseRun:
    """``-v`` lifts the gate and restores today's full transcript."""

    def test_no_gate_installed(self, run_cli):
        run_cli("-v")
        assert gate_installed(_root_handler()) is False
        assert _root_handler().filters == []

    def test_info_is_rendered(self, run_cli, rendered: StringIO):
        run_cli("--verbose")
        _capture_renders(rendered)

        logging.getLogger("osprey.altitude_probe").info(INFO_MARKER)

        assert INFO_MARKER in rendered.getvalue()


class TestSameProcessReinstall:
    """The handler outlives an invocation; the gate is decided per invocation."""

    def test_verbose_then_plain(self, run_cli):
        run_cli("-v")
        assert gate_installed(_root_handler()) is False

        run_cli()
        handler = _root_handler()
        assert gate_installed(handler) is True
        assert _gate_count(handler) == 1

    def test_plain_then_verbose(self, run_cli):
        run_cli()
        assert gate_installed(_root_handler()) is True

        run_cli("-v")
        handler = _root_handler()
        assert gate_installed(handler) is False
        assert _gate_count(handler) == 0

    def test_repeated_plain_runs_never_stack_filters(self, run_cli):
        for _ in range(3):
            run_cli()
            handler = _root_handler()
            assert _gate_count(handler) == 1
            assert len(handler.filters) == 1

    def test_the_handler_is_the_same_instance_across_invocations(self, run_cli):
        run_cli()
        first = _root_handler()
        run_cli("-v")
        assert _root_handler() is first


class TestPublicSeam:
    """The API a subcommand's own ``--verbose`` uses (Task 5.6)."""

    def test_lift_gate_defaults_to_the_root_handler(self, run_cli):
        run_cli()
        assert gate_installed() is True

        lift_gate()

        assert gate_installed() is False

    def test_install_gate_defaults_to_the_root_handler(self, run_cli):
        run_cli("-v")
        assert gate_installed() is False

        install_gate()

        assert gate_installed() is True
        assert _gate_count(_root_handler()) == 1

    def test_install_gate_is_idempotent(self):
        handler = logging.NullHandler()
        for _ in range(3):
            install_gate(handler)
        assert _gate_count(handler) == 1

        install_gate(handler, verbose=True)
        assert handler.filters == []

    def test_lift_gate_leaves_foreign_filters_alone(self):
        handler = logging.NullHandler()
        foreign = logging.Filter("some.logger")
        handler.addFilter(foreign)
        install_gate(handler)

        lift_gate(handler)

        assert handler.filters == [foreign]

    def test_no_rich_handler_is_a_no_op(self, monkeypatch: pytest.MonkeyPatch):
        """Non-CLI processes have no RichHandler to gate; nothing raises."""
        root = logging.getLogger()
        monkeypatch.setattr(
            root, "handlers", [h for h in root.handlers if not isinstance(h, RichHandler)]
        )

        install_gate()
        lift_gate()

        assert gate_installed() is False
