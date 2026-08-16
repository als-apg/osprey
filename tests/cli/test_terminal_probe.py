"""Tests for the ``terminal_probe`` fixture — the instrument, not a subject.

Every altitude and promotion needle downstream reads its verdict off this
fixture, so the fixture's own claims have to be pinned first: that
``rendered_text`` really is what the gated handler painted, that ``records``
really is the ungated stream, that a CLI run cannot steal the console back, and
that nothing survives the test that installed it.
"""

import logging
from io import StringIO

import pytest
from click.testing import CliRunner
from rich.console import Console
from rich.logging import RichHandler

from osprey.cli.altitude import gate_installed
from osprey.cli.main import cli
from osprey_connectors.logger import configure_logging
from tests.cli.conftest import TerminalProbe, _ProbeRecordSink

INFO_MARKER = "PROBEINFOMARKER"
WARNING_MARKER = "PROBEWARNMARKER"

PROBE_LOGGER = "osprey.terminal_probe_subject"


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


def _root_rich_handlers() -> list[RichHandler]:
    return [h for h in logging.getLogger().handlers if isinstance(h, RichHandler)]


class TestGatedRun:
    """A default run: the gate decides the terminal, not what was emitted."""

    def test_info_is_absent_from_the_terminal_but_present_in_the_records(
        self, run_cli, terminal_probe: TerminalProbe
    ):
        run_cli()

        logger = logging.getLogger(PROBE_LOGGER)
        logger.info(INFO_MARKER)
        # A live-console witness in the same test: an absence assertion against
        # a console nothing can reach passes for the wrong reason.
        logger.warning(WARNING_MARKER)

        assert gate_installed() is True
        assert WARNING_MARKER in terminal_probe.rendered_text
        assert INFO_MARKER not in terminal_probe.rendered_text
        assert INFO_MARKER in terminal_probe.messages

    def test_warning_is_present_in_both(self, run_cli, terminal_probe: TerminalProbe):
        run_cli()

        logging.getLogger(PROBE_LOGGER).warning(WARNING_MARKER)

        assert WARNING_MARKER in terminal_probe.rendered_text
        assert WARNING_MARKER in terminal_probe.messages

    def test_records_are_the_records_themselves(self, run_cli, terminal_probe: TerminalProbe):
        """``records`` yields ``LogRecord``s, so needles can read level and name."""
        run_cli()

        logging.getLogger(PROBE_LOGGER).info(INFO_MARKER)

        gated = [r for r in terminal_probe.records if r.getMessage() == INFO_MARKER]
        assert [(r.levelno, r.name) for r in gated] == [(logging.INFO, PROBE_LOGGER)]

    def test_the_probe_does_not_disturb_the_gate(self, run_cli, terminal_probe: TerminalProbe):
        """Swapping the console must not cost the handler its filter."""
        run_cli()

        handler = _root_rich_handlers()[0]
        assert handler.console is terminal_probe.console
        assert gate_installed(handler) is True


class TestVerboseRun:
    """``-v`` lifts the gate; the probe sees the full transcript."""

    def test_info_reaches_the_terminal(self, run_cli, terminal_probe: TerminalProbe):
        run_cli("-v")

        logging.getLogger(PROBE_LOGGER).info(INFO_MARKER)

        assert gate_installed() is False
        assert INFO_MARKER in terminal_probe.rendered_text
        assert INFO_MARKER in terminal_probe.messages


class TestInvocationOrder:
    """The CLI repoints the handler mid-run; the probe has to win anyway."""

    def test_probe_first_then_cli(self, terminal_probe: TerminalProbe, run_cli):
        """``set_handler_console(err_console)`` must not steal the console."""
        from osprey.cli.styles import err_console

        run_cli()

        handler = _root_rich_handlers()[0]
        assert handler.console is terminal_probe.console
        assert handler.console is not err_console

    def test_cli_first_then_probe(self, run_cli, request: pytest.FixtureRequest):
        """A run in an earlier fixture is fine: the probe repoints on setup."""
        run_cli()
        probe: TerminalProbe = request.getfixturevalue("terminal_probe")

        logging.getLogger(PROBE_LOGGER).warning(WARNING_MARKER)

        assert _root_rich_handlers()[0].console is probe.console
        assert WARNING_MARKER in probe.rendered_text

    def test_repeated_invocations_keep_the_probe_console(
        self, terminal_probe: TerminalProbe, run_cli
    ):
        for args in ((), ("-v",), ()):
            run_cli(*args)
            assert _root_rich_handlers()[0].console is terminal_probe.console


class TestWithoutTheCli:
    """The instrument works for a bare entry point too — no CLI, no gate."""

    def test_bare_configure_logging_paints_into_the_probe(self, terminal_probe: TerminalProbe):
        configure_logging()

        logging.getLogger(PROBE_LOGGER).info(INFO_MARKER)

        assert len(_root_rich_handlers()) == 1
        assert gate_installed() is False
        assert INFO_MARKER in terminal_probe.rendered_text
        assert INFO_MARKER in terminal_probe.messages

    def test_no_configuration_at_all(self, terminal_probe: TerminalProbe):
        """The fixture installs the handler itself when nobody else has."""
        logging.getLogger(PROBE_LOGGER).warning(WARNING_MARKER)

        assert WARNING_MARKER in terminal_probe.rendered_text
        assert WARNING_MARKER in terminal_probe.messages

    def test_the_root_level_is_left_alone(self, terminal_probe: TerminalProbe):
        """A real demotion stays detectable: DEBUG under INFO never emits."""
        configure_logging()

        logging.getLogger(PROBE_LOGGER).debug("a demoted call site speaking")

        assert terminal_probe.messages == []


class TestRenderedTextShape:
    """Styled and plain assertions are both possible off one render."""

    def test_styled_text_keeps_the_escape_sequences(self, terminal_probe: TerminalProbe):
        logging.getLogger(PROBE_LOGGER).warning("[success]%s[/success] tail", WARNING_MARKER)

        assert "\x1b[" in terminal_probe.styled_text
        assert f"{WARNING_MARKER} tail" in terminal_probe.rendered_text
        assert "\x1b[" not in terminal_probe.rendered_text

    def test_theme_style_names_resolve(self, terminal_probe: TerminalProbe):
        """The CLI's own theme is on the probe console, so its markup renders."""
        logging.getLogger(PROBE_LOGGER).warning("[warning]%s[/warning]", WARNING_MARKER)

        assert WARNING_MARKER in terminal_probe.rendered_text

    def test_lines_are_right_stripped(self, terminal_probe: TerminalProbe):
        logging.getLogger(PROBE_LOGGER).warning(WARNING_MARKER)

        painted = [line for line in terminal_probe.lines if WARNING_MARKER in line]
        assert painted and painted[0].endswith(WARNING_MARKER)

    def test_clear_drops_both_halves(self, terminal_probe: TerminalProbe):
        logging.getLogger(PROBE_LOGGER).warning(WARNING_MARKER)
        assert terminal_probe.rendered_text

        terminal_probe.clear()

        assert terminal_probe.rendered_text == ""
        assert terminal_probe.records == []

        logging.getLogger(PROBE_LOGGER).warning("AFTERCLEAR")
        assert "AFTERCLEAR" in terminal_probe.rendered_text

    def test_arm_repoints_a_handler_installed_behind_the_fixtures_back(
        self, terminal_probe: TerminalProbe
    ):
        """The escape hatch for a test that rebuilds logging itself."""
        root = logging.getLogger()
        for handler in _root_rich_handlers():
            root.removeHandler(handler)
        root.addHandler(RichHandler(console=Console(file=StringIO())))

        terminal_probe.arm()

        logging.getLogger(PROBE_LOGGER).warning(WARNING_MARKER)
        assert WARNING_MARKER in terminal_probe.rendered_text


class TestNoLeak:
    """Nothing the probe installs may outlive the test that asked for it."""

    def test_a_run_under_the_probe(self, run_cli, terminal_probe: TerminalProbe):
        run_cli()
        logging.getLogger(PROBE_LOGGER).warning(WARNING_MARKER)
        assert WARNING_MARKER in terminal_probe.rendered_text

    def test_the_next_test_sees_no_probe_console(self):
        for handler in _root_rich_handlers():
            assert not isinstance(handler.console.file, StringIO)

    def test_the_next_test_sees_no_probe_sink(self):
        assert not any(
            isinstance(handler, _ProbeRecordSink) for handler in logging.getLogger().handlers
        )
