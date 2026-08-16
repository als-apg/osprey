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
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from osprey.cli.altitude import gate_installed, install_gate
from osprey.cli.chat_cmd import (
    _launch_companion_servers,
    _stand_down_output_policy,
    chat,
)
from osprey.cli.phase_reporter import LiveReporter, NullReporter, current_reporter, install_reporter
from osprey.utils.logger import configure_logging
from tests.cli._lifecycle_build import stub_build
from tests.cli._scoped_subprocess import patch_subprocess


@dataclass
class Handoff:
    """The output-policy state read at the moment the agent CLI starts."""

    gate_installed: bool
    root_level: int
    reporter: object


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture(autouse=True)
def _restore_process_state():
    """Undo what an in-process launch does to the interpreter's globals.

    ``chat`` chdirs into ``build/``, overlays the deployment's ``.env`` onto
    ``os.environ``, and installs a reporter — all deliberate, all process-global,
    and all able to reach every test that runs after this one in the same worker.
    """
    cwd = Path.cwd()
    environ = dict(os.environ)
    reporter = current_reporter()
    yield
    install_reporter(reporter)
    os.chdir(cwd)
    os.environ.clear()
    os.environ.update(environ)


@pytest.fixture(autouse=True)
def _no_managed_policy(monkeypatch: pytest.MonkeyPatch):
    """Pin the managed-policy scan to "no conflicts".

    It reads OS-standard policy files, so on a machine that has one every launch
    here would refuse for a reason none of these tests is about.
    """
    monkeypatch.setattr(
        "osprey.build.claude_code_resolver.detect_managed_policy_conflicts",
        lambda paths=None: {},
    )


@pytest.fixture
def gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Put the CLI's normal starting state on the root logger.

    ``runner.invoke(chat, ...)`` reaches the subcommand without passing through
    the group callback that configures logging and installs the gate, so a test
    that wants to watch the gate come off has to put it on first. The root
    conftest's ``restore_root_logging`` takes both back off afterwards.

    A live reporter goes with it: the default reporter is already the quiet one,
    so a switch asserted against that default would pass while doing nothing.
    """
    monkeypatch.setattr("osprey.utils.config.load_project_dotenv", lambda *a, **k: None)
    configure_logging()
    install_gate()
    assert gate_installed(), "the fixture failed to install the gate it is about to watch"
    install_reporter(LiveReporter())


@pytest.fixture
def handoff(monkeypatch: pytest.MonkeyPatch) -> list[Handoff]:
    """Record the output-policy state instead of starting an agent."""
    recorded: list[Handoff] = []

    def fake_run(argv, *args, **kwargs):
        recorded.append(
            Handoff(
                gate_installed=gate_installed(),
                root_level=logging.getLogger().level,
                reporter=current_reporter(),
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("osprey.cli.chat_cmd._launch_companion_servers", lambda project_dir: [])
    with patch_subprocess("osprey.cli.chat_cmd", side_effect=fake_run):
        yield recorded


class TestTheSwitchItself:
    """Called directly, with no verb around it."""

    def test_it_lifts_the_gate(self, gated: None) -> None:
        _stand_down_output_policy()
        assert gate_installed() is False

    def test_it_installs_the_quiet_reporter(self, gated: None) -> None:
        _stand_down_output_policy()
        reporter = current_reporter()
        assert isinstance(reporter, NullReporter)
        assert reporter.verbose is False

    def test_it_touches_no_logger_level(self, gated: None) -> None:
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


class TestAtTheHandoff:
    """The state the agent process is actually started in."""

    def test_the_gate_is_lifted_before_the_agent_starts(
        self, runner: CliRunner, gated: None, handoff: list[Handoff], lifecycle_repo: Path
    ) -> None:
        stub_build(lifecycle_repo)

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert len(handoff) == 1
        assert handoff[0].gate_installed is False

    def test_the_root_logger_is_not_pinned_to_critical(
        self, runner: CliRunner, gated: None, handoff: list[Handoff], lifecycle_repo: Path
    ) -> None:
        """The hack this replaced set the root logger to CRITICAL right here."""
        stub_build(lifecycle_repo)
        before = logging.getLogger().level

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert handoff[0].root_level != logging.CRITICAL
        assert handoff[0].root_level == before

    def test_the_reporter_is_the_quiet_one(
        self, runner: CliRunner, gated: None, handoff: list[Handoff], lifecycle_repo: Path
    ) -> None:
        stub_build(lifecycle_repo)

        result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert isinstance(handoff[0].reporter, NullReporter)


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
        """
        stub_build(lifecycle_repo)
        monkeypatch.setattr("osprey.cli.chat_cmd._launch_companion_servers", lambda p: [])

        with patch_subprocess("osprey.cli.chat_cmd") as fake:
            fake.run.return_value = SimpleNamespace(returncode=0)
            result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert fake.run.call_args.kwargs == {}
        assert len(fake.run.call_args.args) == 1

    def test_agent_text_reaches_the_reader_unchanged(
        self, runner: CliRunner, gated: None, lifecycle_repo: Path, monkeypatch: pytest.MonkeyPatch
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

        def fake_run(argv, *args, **kwargs):
            sys.stdout.write(agent_line + "\n")
            return SimpleNamespace(returncode=0)

        with patch_subprocess("osprey.cli.chat_cmd", side_effect=fake_run):
            result = runner.invoke(chat, ["--repo", str(lifecycle_repo)])

        assert result.exit_code == 0
        assert agent_line + "\n" in result.stdout
