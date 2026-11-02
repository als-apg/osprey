"""``osprey chat``'s one switch: OSPREY's output policy stands down for the REPL.

The verb ends by handing the terminal to another program. Everything OSPREY has
to say about how output should look — the altitude gate on the log handler, the
phase reporter's decoration — applies to OSPREY's own report and not to a
full-screen agent session, so a single call turns both off immediately before the
handoff.

What that call replaced was a root-logger pin to CRITICAL. The pin silenced the
process rather than the painting: records were never emitted, so a warning worth
having was gone rather than merely unpainted, and the companion-server failure
path had to print by hand to get around it. These tests pin the switch's three
observable effects and the absence of that pin.

Every test stops at the handoff boundary. ``subprocess.run`` is replaced by a
recorder that reads the logging and reporter state at the exact moment the agent
would have started, which is the only moment the assertions are about.
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from osprey.cli.altitude import (
    _root_rich_handler,
    gate_installed,
    handoff_gate_installed,
    install_gate,
)
from osprey.cli.chat_cmd import (
    _launch_companion_servers,
    _stand_back_up,
    _stand_down_output_policy,
    chat,
)
from osprey.cli.phase_reporter import LiveReporter, NullReporter, current_reporter, install_reporter
from osprey_connectors.logger import configure_logging
from tests.cli._lifecycle_build import stub_build
from tests.cli._scoped_subprocess import patch_subprocess


@dataclass
class Handoff:
    """The output-policy state read at the moment the agent CLI starts."""

    gate_installed: bool
    root_level: int
    reporter: object
    handoff_gate_installed: bool = False
    session_logs: tuple[Path, ...] = ()
    painted: tuple[str, ...] = ()


@pytest.fixture(autouse=True)
def _no_managed_policy(monkeypatch: pytest.MonkeyPatch):
    """Pin the policy read to an empty policy.

    It reads OS-standard policy files, so on a machine that has one every launch
    here would refuse for a reason none of these tests is about.
    """
    monkeypatch.setattr(
        "osprey.agent_runner.provider_env.read_managed_policy_env",
        lambda paths=None: {},
    )


@pytest.fixture
def gated(
    monkeypatch: pytest.MonkeyPatch,
    restore_installed_reporter: None,  # noqa: ARG001 - snapshots the reporter before this installs one
) -> None:
    """Put the CLI's normal starting state on the root logger.

    ``runner.invoke(chat, ...)`` reaches the subcommand without passing through
    the group callback that configures logging and installs the gate, so a test
    that wants to watch the gate come off has to put it on first. The root
    conftest's ``restore_root_logging`` takes both back off afterwards.

    A live reporter goes with it: the default reporter is already the quiet one,
    so a switch asserted against that default would pass while doing nothing.
    The reporter is a process singleton, so the restore fixture is requested
    here rather than left to the module's ``usefixtures`` mark: a test-level
    mark sets its fixtures up before a module-level one, which would let the
    restore snapshot this live reporter and put it back after the test. Left
    installed, it tells the altitude gate a lifecycle verb owns the terminal,
    and every later CLI run on the worker paints no WARNING at all.
    """
    monkeypatch.setattr("osprey_connectors.config.load_project_dotenv", lambda *a, **k: None)
    configure_logging()
    install_gate()
    assert gate_installed(), "the fixture failed to install the gate it is about to watch"
    install_reporter(LiveReporter())


@pytest.fixture
def handoff(monkeypatch: pytest.MonkeyPatch) -> list[Handoff]:
    """Record the output-policy state instead of starting an agent."""
    recorded: list[Handoff] = []

    def fake_run(argv, *args, **kwargs):  # noqa: ARG001 - the command position subprocess.run is called at
        # What a companion thread would say while the agent owns the screen,
        # and which of it the terminal handler would still paint.
        thread_logger = logging.getLogger("tests.companion-thread")
        rich = _root_rich_handler()
        painted = []
        for level, text in (
            (logging.INFO, "late health check passed"),
            (logging.WARNING, "pool initialization incomplete"),
            (logging.ERROR, "companion thread crashed"),
        ):
            record = thread_logger.makeRecord(
                thread_logger.name, level, __file__, 0, text, (), None
            )
            thread_logger.handle(record)
            if rich is not None and rich.filter(record):
                painted.append(text)
        recorded.append(
            Handoff(
                gate_installed=gate_installed(),
                root_level=logging.getLogger().level,
                reporter=current_reporter(),
                handoff_gate_installed=handoff_gate_installed(),
                session_logs=tuple(
                    Path(h.baseFilename)
                    for h in logging.getLogger().handlers
                    if isinstance(h, logging.FileHandler)
                ),
                painted=tuple(painted),
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("osprey.cli.chat_cmd._launch_companion_servers", lambda project_dir: [])
    with patch_subprocess("osprey.cli.chat_cmd", side_effect=fake_run):
        yield recorded


class TestTheSwitchItself:
    """Called directly, with no verb around it."""

    @pytest.mark.usefixtures("gated")
    def test_it_lifts_the_gate(self) -> None:
        _stand_down_output_policy()
        assert gate_installed() is False

    @pytest.mark.usefixtures("gated")
    def test_it_installs_the_quiet_reporter(self) -> None:
        _stand_down_output_policy()
        reporter = current_reporter()
        assert isinstance(reporter, NullReporter)
        assert reporter.verbose is False

    @pytest.mark.usefixtures("gated")
    def test_it_touches_no_logger_level(self) -> None:
        """Standing down is about painting, not about what is emitted."""
        before = logging.getLogger().level
        _stand_down_output_policy()
        assert logging.getLogger().level == before
        assert logging.getLogger().level != logging.CRITICAL

    def test_it_is_safe_without_a_handler_to_lift_from(self) -> None:
        """A library caller may have configured no Rich handler at all."""
        for handler in list(logging.getLogger().handlers):
            logging.getLogger().removeHandler(handler)
        _stand_down_output_policy()
        assert gate_installed() is False

    @pytest.mark.usefixtures("gated")
    def test_with_a_session_log_the_terminal_paints_error_and_above_only(
        self, tmp_path: Path
    ) -> None:
        """INFO and WARNING go to the file; ERROR still reaches the screen too."""
        session_log = tmp_path / "var" / "logs" / "chat.log"
        sink = _stand_down_output_policy(session_log)
        try:
            assert sink is not None
            assert handoff_gate_installed() is True
            rich = _root_rich_handler()
            assert rich is not None
            thread_logger = logging.getLogger("tests.companion-thread")
            verdicts = {}
            for level in (logging.INFO, logging.WARNING, logging.ERROR):
                record = thread_logger.makeRecord(
                    thread_logger.name, level, __file__, 0, f"at {level}", (), None
                )
                thread_logger.handle(record)
                verdicts[level] = bool(rich.filter(record))
            sink.flush()
        finally:
            _stand_back_up(sink)
        assert verdicts == {logging.INFO: False, logging.WARNING: False, logging.ERROR: True}
        written = session_log.read_text(encoding="utf-8")
        for level in (logging.INFO, logging.WARNING, logging.ERROR):
            assert f"at {level}" in written, "every record reaches the log"

    @pytest.mark.usefixtures("gated")
    def test_standing_back_up_leaves_nothing_behind(self, tmp_path: Path) -> None:
        sink = _stand_down_output_policy(tmp_path / "chat.log")
        _stand_back_up(sink)
        assert handoff_gate_installed() is False
        assert sink not in logging.getLogger().handlers

    @pytest.mark.usefixtures("gated")
    def test_a_log_that_cannot_be_opened_keeps_the_terminal_painting(self, tmp_path: Path) -> None:
        """Painting over the screen loses less than dropping the record."""
        blocker = tmp_path / "var"
        blocker.write_text("not a directory")
        sink = _stand_down_output_policy(blocker / "logs" / "chat.log")
        assert sink is None
        assert handoff_gate_installed() is False


class TestAtTheHandoff:
    """The state the agent process is actually started in."""

    @pytest.mark.usefixtures("gated")
    def test_the_gate_is_lifted_before_the_agent_starts(
        self, runner: CliRunner, handoff: list[Handoff], lifecycle_repo: Path
    ) -> None:
        stub_build(lifecycle_repo)

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert len(handoff) == 1
        assert handoff[0].gate_installed is False

    @pytest.mark.usefixtures("gated")
    def test_the_root_logger_is_not_pinned_to_critical(
        self, runner: CliRunner, handoff: list[Handoff], lifecycle_repo: Path
    ) -> None:
        """The hack this replaced set the root logger to CRITICAL right here."""
        stub_build(lifecycle_repo)
        before = logging.getLogger().level

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert handoff[0].root_level != logging.CRITICAL
        assert handoff[0].root_level == before

    @pytest.mark.usefixtures("gated")
    def test_the_reporter_is_the_quiet_one(
        self, runner: CliRunner, handoff: list[Handoff], lifecycle_repo: Path
    ) -> None:
        stub_build(lifecycle_repo)

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert isinstance(handoff[0].reporter, NullReporter)

    @pytest.mark.usefixtures("gated")
    def test_companion_records_go_to_the_session_log_not_the_screen(
        self, runner: CliRunner, handoff: list[Handoff], lifecycle_repo: Path
    ) -> None:
        """While the agent owns the screen, INFO and WARNING land in var/logs/.

        A health check that passes late or a pool that times out used to be
        painted over the agent's first screen on a slow host. ERROR keeps
        painting: a dead companion thread outranks the screen's tidiness.
        """
        stub_build(lifecycle_repo)

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        state = handoff[0]
        assert state.handoff_gate_installed is True
        assert state.painted == ("companion thread crashed",)
        session_logs = [
            p for p in state.session_logs if p.parent == lifecycle_repo / "var" / "logs"
        ]
        assert len(session_logs) == 1, state.session_logs
        assert session_logs[0].name.startswith("chat-")
        written = session_logs[0].read_text(encoding="utf-8")
        assert "late health check passed" in written
        assert "pool initialization incomplete" in written
        assert "companion thread crashed" in written
        assert str(session_logs[0]) in result.output

    @pytest.mark.usefixtures("gated")
    def test_the_terminal_is_handed_back_when_the_agent_exits(
        self, runner: CliRunner, handoff: list[Handoff], lifecycle_repo: Path
    ) -> None:
        stub_build(lifecycle_repo)

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert handoff[0].handoff_gate_installed is True
        assert handoff_gate_installed() is False
        assert not any(
            isinstance(h, logging.FileHandler)
            and Path(h.baseFilename).parent == lifecycle_repo / "var" / "logs"
            for h in logging.getLogger().handlers
        )


class TestCompanionServers:
    """The function the deleted pin lived in."""

    def test_launching_them_leaves_every_logger_level_alone(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No level is pinned, so a warning from a daemon thread survives.

        Silencing by level was the wrong instrument twice over: it dropped the
        records instead of not painting them, and it swallowed exactly the
        warnings the launch path exists to report.

        Every server is offered its chance and declines to have started, which
        is what keeps this a test of the logging levels and not of the servers.
        The stubs go on ``server_launcher``'s own names rather than on
        ``registry.web.FRAMEWORK_WEB_SERVERS``: that registry is read at IMPORT
        time to build ``_launchers`` and the auto-launch checkers, so emptying
        it around the first import of ``server_launcher`` in a worker builds
        those tables empty and leaves them that way for every test afterwards.
        """
        from osprey.infrastructure import server_launcher

        monkeypatch.setattr(server_launcher, "ensure_web_server", lambda key: None)
        monkeypatch.setattr(
            server_launcher,
            "_launchers",
            defaultdict(lambda: SimpleNamespace(_launched=False)),
        )
        root = logging.getLogger()
        before = root.level

        assert _launch_companion_servers(tmp_path) == []

        assert root.level == before
        assert root.level != logging.CRITICAL


class TestAgentTextPassesThrough:
    """FR1: what the agent writes is not OSPREY's to shape."""

    def test_the_agent_stream_is_never_captured(
        self, runner: CliRunner, lifecycle_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No pipe, no capture: the child inherits the terminal's own streams.

        This is the whole mechanism behind "streams exactly as written" — a
        ``stdout=PIPE`` here would put OSPREY between the agent and the reader,
        and every guarantee about the text would then depend on what OSPREY did
        with it next.

        ``env=`` is the one kwarg the spawn is allowed to carry: the agent is
        launched with a deliberately built environment (the operator secret
        stripped, the panel token re-added) rather than the parent's
        ``os.environ``. It redirects nothing, so it is not a capture — hence
        the assertion names every kwarg that WOULD be one instead of demanding
        an empty kwargs dict, which would fail on any future non-redirecting
        argument for a reason unrelated to what this test protects.
        """
        stub_build(lifecycle_repo)
        monkeypatch.setattr("osprey.cli.chat_cmd._launch_companion_servers", lambda p: [])

        with patch_subprocess("osprey.cli.chat_cmd") as fake:
            fake.run.return_value = SimpleNamespace(returncode=0)
            result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        kwargs = fake.run.call_args.kwargs
        for redirecting in ("stdout", "stderr", "stdin", "capture_output", "input", "text"):
            assert redirecting not in kwargs, (
                f"{redirecting}= puts OSPREY between the agent and the reader"
            )
        assert set(kwargs) <= {"env"}
        assert isinstance(kwargs.get("env", {}), dict)
        assert len(fake.run.call_args.args) == 1

    @pytest.mark.usefixtures("gated")
    def test_agent_text_reaches_the_reader_unchanged(
        self, runner: CliRunner, lifecycle_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Byte-for-byte, including what the renderer would have eaten.

        The line carries a Rich markup tag, a box-drawing character and trailing
        spaces: markup would be interpreted, a themed console would restyle it,
        and soft-wrapping or a `.strip()` anywhere on the path would lose the
        tail. The recorder writes it the way the agent process would.
        """
        agent_line = "[bold]literal[/bold] │ 100%   "
        stub_build(lifecycle_repo)
        monkeypatch.setattr("osprey.cli.chat_cmd._launch_companion_servers", lambda p: [])

        def fake_run(argv, *args, **kwargs):  # noqa: ARG001 - the command position subprocess.run is called at
            sys.stdout.write(agent_line + "\n")
            return SimpleNamespace(returncode=0)

        with patch_subprocess("osprey.cli.chat_cmd", side_effect=fake_run):
            result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert agent_line + "\n" in result.stdout
