"""What a channel-finder run does to logging, and what it must no longer do.

The group used to run its own altitude policy: ``_initialize_registry`` pinned
the ``osprey`` and ``channel_finder`` loggers to WARNING, so a non-verbose run
never *emitted* an INFO record at all. The CLI's altitude gate
(``osprey.cli.altitude``) now owns that policy for every command, and it is a
handler filter — a record it drops is still emitted and still observable.

The difference is the point of this module: what the terminal shows is
unchanged, but nothing is destroyed on the way there any more.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli.altitude import gate_installed
from osprey.cli.main import cli
from tests.cli.conftest import TerminalProbe

# The two loggers the group used to pin. Both are framework loggers, so a pin on
# either silences call sites far outside this command group.
PINNED_LOGGERS = ("osprey", "channel_finder")

INFO_MARKER = "CHANNELFINDERINFOMARKER"
WARNING_MARKER = "CHANNELFINDERWARNMARKER"


@pytest.fixture
def unpinned_loggers() -> Iterator[None]:
    """Start from NOTSET on both loggers, and hand back whatever was there.

    Levels are process-global and a pin left by some other module would fail
    these tests for a reason they are not about. Resetting first makes the
    assertion the exact claim: *this run* set no level. Under the old code the
    same test fails, because ``_initialize_registry`` pinned both to WARNING.
    """
    loggers = [logging.getLogger(name) for name in PINNED_LOGGERS]
    saved = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.NOTSET)
    try:
        yield
    finally:
        for logger, level in zip(loggers, saved, strict=True):
            logger.setLevel(level)


@pytest.fixture
def stub_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the real ``_initialize_registry`` run without building a registry.

    Patching ``_initialize_registry`` itself — what the other channel-finder
    suites do — would make every claim here vacuous: the deleted pins lived
    inside it. Only the registry build underneath is stubbed.
    """
    monkeypatch.setattr("osprey.registry.initialize_registry", lambda *a, **k: None)


@pytest.fixture
def in_context_database(tmp_path: Path) -> str:
    """A minimal in-context database ``validate`` can actually load."""
    db_file = tmp_path / "channel_db.json"
    db_file.write_text(
        '{"channels": [{"template": false, "channel": "CH1", '
        '"address": "PV:CH1", "description": "Test channel"}]}',
        encoding="utf-8",
    )
    return str(db_file)


@pytest.fixture
def run_validate(monkeypatch: pytest.MonkeyPatch, in_context_database: str):
    """Invoke ``osprey channel-finder validate`` through the real CLI group.

    Through ``cli``, not the ``channel_finder`` group directly: the group
    callback is what installs the altitude gate, so anything asserted about
    rendering has to go the whole way round. ``load_project_dotenv`` is stubbed
    — it writes an ancestor ``.env`` straight into ``os.environ``, outliving the
    test — and ``_setup_config`` is stubbed so the run needs no built project.
    """
    monkeypatch.setattr("osprey.utils.config.load_project_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("osprey.cli.channel_finder_cmd._setup_config", lambda *a, **k: None)
    runner = CliRunner()

    def _run(*args: str):
        return runner.invoke(
            cli,
            [
                "channel-finder",
                "validate",
                "--database",
                in_context_database,
                "--pipeline",
                "in_context",
                *args,
            ],
        )

    return _run


class TestNoPrivateLevelPins:
    """Neither the registry init nor a whole command sets a logger level."""

    def test_registry_init_pins_nothing(self, unpinned_loggers, stub_registry):
        from osprey.cli.channel_finder_cmd import _initialize_registry

        _initialize_registry()

        assert [logging.getLogger(n).level for n in PINNED_LOGGERS] == [
            logging.NOTSET,
            logging.NOTSET,
        ]

    def test_a_command_run_pins_nothing(self, unpinned_loggers, stub_registry, run_validate):
        result = run_validate()

        # The command reached its own work — otherwise it could pin nothing for
        # an uninteresting reason.
        assert "VALID" in result.output
        assert [logging.getLogger(n).level for n in PINNED_LOGGERS] == [
            logging.NOTSET,
            logging.NOTSET,
        ]

    def test_the_loggers_inherit_the_run_level(self, unpinned_loggers, stub_registry, run_validate):
        """NOTSET is not merely "unset": it is the run's level, inherited."""
        run_validate()

        assert logging.getLogger("channel_finder").getEffectiveLevel() == logging.INFO
        assert logging.getLogger("osprey").getEffectiveLevel() == logging.INFO

    def test_the_module_configures_no_logging_of_its_own(self):
        """``basicConfig`` is gone — the CLI entry point owns that call.

        It was dead where it stood: the group callback runs
        ``configure_logging`` before any subcommand body, so the root logger
        always has a handler by then and ``basicConfig`` returns without
        touching level or handlers.
        """
        import osprey.cli.channel_finder_cmd as module

        source = Path(module.__file__).read_text(encoding="utf-8")

        assert "basicConfig" not in source
        assert "setLevel(logging.WARNING)" not in source


class TestGatePolicyStillHolds:
    """What the operator sees is unchanged; what survives underneath is not."""

    @pytest.fixture
    def noisy_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make ``validate``'s own work log one INFO and one WARNING record.

        Emitted from inside the running command, on the group's own logger, so
        the gate is asked the question in situ rather than after the fact.
        """

        def _run_validation(**kwargs) -> int:
            logger = logging.getLogger("channel_finder.validate")
            logger.info(INFO_MARKER)
            logger.warning(WARNING_MARKER)
            return 0

        monkeypatch.setattr(
            "osprey.services.channel_finder.tools.validate_database.run_validation",
            _run_validation,
        )

    def test_warning_still_renders(
        self,
        unpinned_loggers,
        stub_registry,
        noisy_validation,
        run_validate,
        terminal_probe: TerminalProbe,
    ):
        run_validate()

        assert gate_installed() is True
        assert WARNING_MARKER in terminal_probe.rendered_text

    def test_info_does_not_render_but_is_still_emitted(
        self,
        unpinned_loggers,
        stub_registry,
        noisy_validation,
        run_validate,
        terminal_probe: TerminalProbe,
    ):
        """The honest form of the claim: gated from the terminal, not destroyed.

        The WARNING is the armed witness — against a console nothing reaches,
        the absence assertion would pass for the wrong reason. And the record
        half is what the old pins made impossible: at WARNING on the
        ``channel_finder`` logger, this INFO was never emitted at all.
        """
        run_validate()

        assert WARNING_MARKER in terminal_probe.rendered_text
        assert INFO_MARKER not in terminal_probe.rendered_text
        assert INFO_MARKER in terminal_probe.messages

    def test_the_gated_record_keeps_its_level_and_name(
        self,
        unpinned_loggers,
        stub_registry,
        noisy_validation,
        run_validate,
        terminal_probe: TerminalProbe,
    ):
        run_validate()

        gated = [r for r in terminal_probe.records if r.getMessage() == INFO_MARKER]
        assert [(r.levelno, r.name) for r in gated] == [(logging.INFO, "channel_finder.validate")]
