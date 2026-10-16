"""The startup line that names the Claude Code build the two views run.

The terminal launches the CLI named by the argv :mod:`osprey.agent_runner.launcher`
builds; the chat goes through the Agent SDK, which OSPREY points at the
installed ``claude`` of the same version. The server settles that executable
once at startup and names it beside the terminal's argv, and says so in the
same place when the pinned version is not installed.

Nothing here runs a real CLI: the ``PATH`` lookup and the version probe are
replaced.
"""

from __future__ import annotations

import logging

import pytest

from osprey.agent_runner import launcher
from osprey.interfaces.web_terminal import app as web_app

PIN = "2.1.146"
INSTALLED = "/usr/local/bin/claude"

APP_LOGGER = web_app.logger.name

#: These tests are about the real lookup, not the suite-wide stand-in.
pytestmark = pytest.mark.usefixtures("real_agent_cli")


@pytest.fixture
def project(tmp_path):
    (tmp_path / "config.yml").write_text(f"claude_code:\n  cli_version: '{PIN}'\n")
    return tmp_path


@pytest.fixture
def installed(monkeypatch):
    """Put a ``claude`` on PATH that reports the given version."""

    def install(version, path=INSTALLED):
        monkeypatch.setattr("shutil.which", lambda *_a, **_k: path)
        monkeypatch.setattr(launcher, "_probe_cli_version", lambda _exe: version)

    return install


def _settle(project, caplog) -> list[logging.LogRecord]:
    argv = launcher.build_claude_launch_argv(launcher.project_claude_code_config(project))
    with caplog.at_level(logging.INFO, logger=APP_LOGGER):
        web_app._settle_agent_cli(argv, str(project))
    return [r for r in caplog.records if r.getMessage().startswith("Claude Code CLI")]


def test_one_line_names_the_terminals_argv_and_the_chats_executable(project, installed, caplog):
    installed(PIN)

    (line,) = _settle(project, caplog)

    assert line.levelno == logging.INFO
    message = line.getMessage()
    assert f"terminal spawns npx -y @anthropic-ai/claude-code@{PIN}" in message
    assert f"chat runs {INSTALLED} ({PIN})" in message


def test_the_line_names_the_executable_a_chat_run_is_pointed_at(project, installed, caplog):
    from osprey.agent_runner.primitives import build_agent_options

    installed(PIN)

    (line,) = _settle(project, caplog)

    options = build_agent_options(project, disallowed_tools=[], env={})
    assert options.cli_path == INSTALLED
    assert f"chat runs {options.cli_path} " in line.getMessage()


def test_an_unpinned_project_names_the_installed_claude(tmp_path, installed, caplog):
    installed("2.1.999")

    (line,) = _settle(tmp_path, caplog)

    assert line.levelno == logging.INFO
    assert "terminal spawns claude " in line.getMessage()
    assert f"chat runs {INSTALLED} (2.1.999)" in line.getMessage()


def test_nothing_warns_about_two_versions(project, installed, caplog):
    """There is one build; a line that compares two cannot exist."""
    installed(PIN)

    _settle(project, caplog)

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.parametrize(
    ("found", "expected"),
    [("2.1.999", "reports 2.1.999"), (None, "did not report a version")],
    ids=["another-version", "no-version"],
)
def test_a_pinned_version_that_is_not_installed_is_one_error_line(
    project, installed, caplog, found, expected
):
    installed(found)

    (line,) = _settle(project, caplog)

    assert line.levelno == logging.ERROR
    message = line.getMessage()
    assert "chat cannot run" in message
    assert expected in message
    assert f"npm install -g @anthropic-ai/claude-code@{PIN}" in message
    assert line.exc_info is None, "the message is for the operator; no traceback"


def test_a_chat_run_is_refused_with_the_same_message_the_line_gave(project, installed, caplog):
    from osprey.agent_runner.errors import AgentCliUnavailableError
    from osprey.agent_runner.primitives import build_agent_options

    installed("2.1.999")
    (line,) = _settle(project, caplog)

    with pytest.raises(AgentCliUnavailableError) as refused:
        build_agent_options(project, disallowed_tools=[], env={})

    assert str(refused.value) in line.getMessage()


def test_the_executable_is_settled_once_for_the_life_of_the_server(project, installed, caplog):
    installed(PIN)
    _settle(project, caplog)

    installed(PIN, path="/somewhere/else/claude")

    assert launcher.project_agent_cli_path(project) == INSTALLED
