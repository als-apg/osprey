"""The Claude Code a headless run executes.

An SDK run is pointed at an installed ``claude`` and nothing else: with a
``claude_code.cli_version`` pin, the one on ``PATH`` once it has reported
exactly that version; without a pin, the one the interactive launch resolves.
Anything else is a refusal that says what was asked for, what was found and
how to fix it. Nothing here runs a real CLI: the ``PATH`` lookup is replaced,
and the version probe either is replaced or runs a stand-in shell script.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from osprey.agent_runner import launcher
from osprey.agent_runner.errors import AgentCliUnavailableError
from osprey.agent_runner.launcher import CLI_NAME, build_claude_launch_argv
from tests.agent_runner.test_cli_version_pin import BAD_PINS

#: These tests are about the real lookup, not the suite-wide stand-in.
pytestmark = pytest.mark.usefixtures("real_agent_cli")

PIN = "2.1.146"
GLOBAL = "/usr/local/bin/claude"
INSTALL_LINE = f"npm install -g @anthropic-ai/claude-code@{PIN}"


@pytest.fixture
def installed(monkeypatch):
    """Put a ``claude`` on PATH that reports the given version (or raises it)."""

    def install(version, path=GLOBAL):
        monkeypatch.setattr("shutil.which", lambda name, **_k: path if name == CLI_NAME else None)
        probed: list[str] = []

        def probe(executable):
            probed.append(executable)
            if isinstance(version, BaseException):
                raise version
            return version

        monkeypatch.setattr(launcher, "_probe_cli_version", probe)
        return probed

    return install


@pytest.fixture
def nothing_installed(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda *_a, **_k: None)
    monkeypatch.setattr(
        launcher, "_probe_cli_version", lambda exe: pytest.fail(f"nothing to probe, got {exe}")
    )


@pytest.fixture(autouse=True)
def no_download(monkeypatch):
    """A headless resolution never reaches for ``npx`` or ``npm``."""
    real_run = subprocess.run

    def run(argv, *args, **kwargs):
        assert not {"npx", "npm"} & {Path(str(word)).name for word in argv}, argv
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(launcher.subprocess, "run", run)


class TestPinnedProject:
    def test_a_pinned_config_runs_the_global_install_when_it_reports_the_pin(self, installed):
        probed = installed(PIN)

        assert launcher.agent_cli_path({"cli_version": PIN}) == GLOBAL
        assert probed == [GLOBAL]

    def test_the_path_handed_to_the_sdk_is_absolute(self, installed, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        installed(PIN, path="bin/claude")

        assert launcher.agent_cli_path({"cli_version": PIN}) == str(tmp_path / "bin" / "claude")

    def test_a_pinned_config_refuses_when_the_installed_claude_reports_another_version(
        self, installed
    ):
        installed("2.1.999")

        with pytest.raises(AgentCliUnavailableError) as refused:
            launcher.agent_cli_path({"cli_version": PIN})

        message = str(refused.value)
        assert f"pins Claude Code {PIN}" in message
        assert f"({GLOBAL}) reports 2.1.999" in message
        assert INSTALL_LINE in message.splitlines()[1]
        assert "change claude_code.cli_version" in message
        assert "nothing was started" in message

    @pytest.mark.usefixtures("nothing_installed")
    def test_a_pinned_config_refuses_when_no_claude_is_installed(self):
        with pytest.raises(AgentCliUnavailableError) as refused:
            launcher.agent_cli_path({"cli_version": PIN})

        message = str(refused.value)
        assert f"pins Claude Code {PIN}" in message
        assert "no `claude` was found on PATH" in message
        assert INSTALL_LINE in message
        assert "change claude_code.cli_version" in message

    def test_a_pinned_config_refuses_when_the_installed_claude_reports_no_version(self, installed):
        installed(None)

        with pytest.raises(AgentCliUnavailableError) as refused:
            launcher.agent_cli_path({"cli_version": PIN})

        assert f"({GLOBAL}) did not report a version" in str(refused.value)
        assert INSTALL_LINE in str(refused.value)

    def test_a_probe_that_times_out_is_its_own_refusal(self, installed):
        installed(subprocess.TimeoutExpired([GLOBAL, "--version"], 15))

        with pytest.raises(AgentCliUnavailableError) as refused:
            launcher.agent_cli_path({"cli_version": PIN})

        message = str(refused.value)
        assert f"`{GLOBAL} --version` did not answer within" in message
        assert PIN in message
        assert "Nothing was started" in message
        assert "npm install" not in message, "a slow answer is not a missing version"

    @pytest.mark.usefixtures("nothing_installed")
    def test_a_refusal_is_one_message_without_a_cause_to_print(self):
        with pytest.raises(AgentCliUnavailableError) as refused:
            launcher.agent_cli_path({"cli_version": PIN})

        assert isinstance(refused.value, RuntimeError)
        assert refused.value.__cause__ is None

    @pytest.mark.parametrize("pin", ["", "   ", 2])
    def test_an_empty_pin_is_refused_for_sdk_runs_as_for_the_interactive_launch(self, pin):
        with pytest.raises(ValueError) as interactive:
            build_claude_launch_argv({"cli_version": pin})
        with pytest.raises(ValueError) as sdk:
            launcher.agent_cli_path({"cli_version": pin})

        assert str(sdk.value) == str(interactive.value)

    @pytest.mark.parametrize("pin", BAD_PINS, ids=repr)
    def test_the_headless_path_refuses_a_bad_pin_before_looking_for_anything(
        self, monkeypatch, pin
    ):
        monkeypatch.setattr("shutil.which", lambda *_a, **_k: pytest.fail("looked"))

        with pytest.raises(ValueError, match="claude_code.cli_version") as headless:
            launcher.agent_cli_path({"cli_version": pin})
        with pytest.raises(ValueError) as interactive:
            build_claude_launch_argv({"cli_version": pin})

        assert str(headless.value) == str(interactive.value)


class TestUnpinnedProject:
    def test_an_unpinned_config_runs_the_installed_claude_whatever_its_version(self, installed):
        probed = installed("2.1.999")

        assert launcher.agent_cli_path({}) == GLOBAL
        assert probed == [], "without a pin there is no version to check"

    def test_claude_is_found_by_the_lookup_the_interactive_launch_uses(self, monkeypatch):
        from osprey.agent_runner.launcher import resolve_cli_name

        asked: list[str] = []

        def resolve(name):
            asked.append(name)
            return "/home/operator/.local/bin/claude"

        monkeypatch.setattr("osprey.utils.shell_resolver.resolve_shell_command", resolve)

        headless = launcher.agent_cli_path({})
        interactive = resolve_cli_name(build_claude_launch_argv({}))

        assert headless == interactive[0] == "/home/operator/.local/bin/claude"
        assert asked == [CLI_NAME, CLI_NAME]

    @pytest.mark.usefixtures("nothing_installed")
    def test_an_unpinned_config_without_claude_on_path_never_falls_back_to_the_sdk_bundle(self):
        with pytest.raises(AgentCliUnavailableError) as refused:
            launcher.agent_cli_path({})

        message = str(refused.value)
        assert "No `claude` was found on PATH" in message
        assert "npm install -g @anthropic-ai/claude-code" in message
        assert "claude_code.cli_version" in message


class TestVersionProbe:
    """The probe runs a stand-in executable and tells slow from absent."""

    @staticmethod
    def _script(tmp_path, body):
        path = tmp_path / "claude"
        path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        path.chmod(0o755)
        return str(path)

    def test_the_version_is_read_from_the_executable(self, tmp_path):
        exe = self._script(tmp_path, 'echo "2.1.7 (Claude Code)"')

        assert launcher._probe_cli_version(exe) == "2.1.7"
        assert launcher.cli_version_of(exe) == "2.1.7"

    @pytest.mark.parametrize("body", ["exit 3", 'echo "not a version"'])
    def test_a_failing_or_unreadable_executable_reports_no_version(self, tmp_path, body):
        assert launcher._probe_cli_version(self._script(tmp_path, body)) is None

    def test_a_missing_executable_reports_no_version_quietly(self, tmp_path, caplog):
        with caplog.at_level("DEBUG", logger=launcher.logger.name):
            assert launcher._probe_cli_version(str(tmp_path / "absent")) is None

        assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR")]

    def test_a_slow_executable_is_a_timeout_not_a_missing_version(self, tmp_path, monkeypatch):
        monkeypatch.setattr(launcher, "_VERSION_PROBE_TIMEOUT_S", 0.2)
        exe = self._script(tmp_path, "exec sleep 5")

        with pytest.raises(subprocess.TimeoutExpired):
            launcher._probe_cli_version(exe)
        assert launcher.cli_version_of(exe) is None


class TestResolvedOncePerProcess:
    @pytest.fixture
    def project(self, tmp_path):
        (tmp_path / "config.yml").write_text(f"claude_code:\n  cli_version: '{PIN}'\n")
        return tmp_path

    def test_the_projects_pin_names_the_executable(self, project, installed):
        installed(PIN)

        assert launcher.project_agent_cli_path(project) == GLOBAL

    def test_the_executable_is_located_and_probed_once(self, project, installed):
        probed = installed(PIN)

        for _ in range(3):
            assert launcher.project_agent_cli_path(project) == GLOBAL

        assert probed == [GLOBAL]

    def test_a_later_edit_of_the_config_changes_nothing_in_this_process(self, project, installed):
        installed(PIN)
        first = launcher.project_agent_cli_path(project)

        (project / "config.yml").write_text("claude_code:\n  cli_version: '2.1.999'\n")

        assert launcher.project_agent_cli_path(project) == first == GLOBAL

    def test_a_later_move_of_the_binary_changes_nothing_in_this_process(self, project, installed):
        installed(PIN)
        first = launcher.project_agent_cli_path(project)

        installed(PIN, path="/somewhere/else/claude")

        assert launcher.project_agent_cli_path(project) == first == GLOBAL

    def test_a_relative_and_an_absolute_spelling_are_one_project(
        self, project, installed, monkeypatch
    ):
        probed = installed(PIN)
        monkeypatch.chdir(project.parent)

        launcher.project_agent_cli_path(project)
        launcher.project_agent_cli_path(Path(project.name))

        assert probed == [GLOBAL]

    @pytest.mark.usefixtures("nothing_installed")
    def test_a_refusal_is_not_remembered(self, project, installed):
        with pytest.raises(AgentCliUnavailableError):
            launcher.project_agent_cli_path(project)

        installed(PIN)

        assert launcher.project_agent_cli_path(project) == GLOBAL

    def test_each_project_is_resolved_on_its_own(self, project, installed, tmp_path_factory):
        other = tmp_path_factory.mktemp("other")
        (other / "config.yml").write_text("claude_code:\n  cli_version: '2.1.999'\n")
        installed(PIN)

        assert launcher.project_agent_cli_path(project) == GLOBAL
        with pytest.raises(AgentCliUnavailableError, match="pins Claude Code 2.1.999"):
            launcher.project_agent_cli_path(other)


class TestProjectClaudeCodeConfig:
    def test_the_claude_code_block_of_the_projects_config(self, tmp_path):
        (tmp_path / "config.yml").write_text(
            "claude_code:\n  cli_version: '2.1.146'\n  provider: x\n", encoding="utf-8"
        )

        assert launcher.project_claude_code_config(tmp_path) == {
            "cli_version": "2.1.146",
            "provider": "x",
        }

    @pytest.mark.parametrize("text", [None, "", "other: 1\n", "claude_code:\n", "claude_code: 3\n"])
    def test_no_block_is_an_empty_mapping(self, tmp_path, text):
        if text is not None:
            (tmp_path / "config.yml").write_text(text, encoding="utf-8")

        assert launcher.project_claude_code_config(tmp_path) == {}
