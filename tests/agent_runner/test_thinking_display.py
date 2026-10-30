"""The one resolution of ``claude_code.thinking.display``.

The key takes ``summarized`` (the default) or ``omitted``. The
``--thinking-display`` flag is sent only to a CLI at or above
:data:`~osprey.agent_runner.launcher.THINKING_DISPLAY_MIN_CLI_VERSION`; for an
older, prerelease-of-the-minimum or unknown version it is withheld and one
warning says so. An SDK launch takes its path, version and display from one
cached resolution per project, so the three cannot disagree.

The threshold rests on a discriminating probe. ``--version`` exits 0 whatever
flags precede it, so it proves nothing; an invalid value for the flag does. In
an isolated environment (``$T`` a fresh empty directory)::

    env -i PATH=/usr/bin:/bin HOME=$T CLAUDE_CONFIG_DIR=$T <binary> --thinking-display bogus -p x

The CLI bundled with the pinned SDK 0.2.136 (``2.1.228 (Claude Code)``)::

    error: option '--thinking-display <display>' argument 'bogus' is invalid.
    Allowed choices are summarized, omitted.

The default pin (``2.1.300 (Claude Code)``)::

    error: option '--thinking-display <display>' argument 'bogus' is invalid.
    Allowed choices are summarized, omitted, highlights.

A CLI without the flag answers ``unknown option`` instead. Nothing here runs a
real CLI: the ``PATH`` lookup and the version probe are replaced.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

from osprey.agent_runner import launcher
from osprey.agent_runner.errors import AgentCliUnavailableError
from osprey.agent_runner.launcher import (
    CLI_NAME,
    THINKING_DISPLAY_DEFAULT,
    THINKING_DISPLAY_KEY,
    THINKING_DISPLAY_MIN_CLI_VERSION,
    THINKING_DISPLAY_VALUES,
    ResolvedAgentCli,
    ThinkingDisplay,
    launch_cli_version,
    resolve_thinking_display,
    validate_thinking_display,
    warn_thinking_flag_withheld,
)
from tests._agent_cli import STUB_AGENT_CLI

GLOBAL = "/usr/local/bin/claude"
PIN = "2.1.300"


@pytest.fixture
def installed(monkeypatch):
    """Put a ``claude`` on PATH that reports *version*; return the probe record."""

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


def _project(tmp_path, body: str):
    (tmp_path / "config.yml").write_text(body, encoding="utf-8")
    return tmp_path


def _withheld(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and THINKING_DISPLAY_MIN_CLI_VERSION in r.getMessage()
    ]


class TestConstants:
    def test_the_contract_values(self):
        assert THINKING_DISPLAY_KEY == "claude_code.thinking.display"
        assert THINKING_DISPLAY_VALUES == ("summarized", "omitted")
        assert THINKING_DISPLAY_DEFAULT == "summarized"
        assert THINKING_DISPLAY_MIN_CLI_VERSION == "2.1.228"


class TestValidateThinkingDisplay:
    @pytest.mark.parametrize("value", THINKING_DISPLAY_VALUES)
    def test_an_allowed_value_is_returned_unchanged(self, value):
        assert validate_thinking_display(value) == value

    def test_none_means_the_default(self):
        assert validate_thinking_display(None) == "summarized"

    @pytest.mark.parametrize(
        "value", ["highlights", "Summarized", " summarized", "", "on", True, 1, ["omitted"], {}]
    )
    def test_anything_else_is_refused_naming_key_and_both_values(self, value):
        with pytest.raises(ValueError) as info:
            validate_thinking_display(value)
        message = str(info.value)
        assert "claude_code.thinking.display" in message
        assert "summarized" in message and "omitted" in message
        assert repr(value) in message


class TestThinkingDisplay:
    def test_is_frozen_with_value_and_flag_allowed(self):
        shown = ThinkingDisplay("summarized", True)
        assert (shown.value, shown.flag_allowed) == ("summarized", True)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            shown.value = "omitted"  # type: ignore[misc]


class TestResolveThinkingDisplay:
    def test_no_thinking_block_resolves_to_the_default(self):
        assert resolve_thinking_display({}, PIN) == ThinkingDisplay("summarized", True)

    def test_an_empty_thinking_block_resolves_to_the_default(self):
        assert resolve_thinking_display({"thinking": {}}, PIN).value == "summarized"

    def test_a_null_thinking_block_resolves_to_the_default(self):
        assert resolve_thinking_display({"thinking": None}, PIN).value == "summarized"

    def test_the_configured_value(self):
        cc = {"thinking": {"display": "omitted"}}
        assert resolve_thinking_display(cc, PIN) == ThinkingDisplay("omitted", True)

    def test_a_bad_display_value_raises(self):
        with pytest.raises(ValueError, match=r"claude_code\.thinking\.display"):
            resolve_thinking_display({"thinking": {"display": "loud"}}, PIN)

    @pytest.mark.parametrize("value", ["summarized", ["display"], 3, True])
    def test_a_non_mapping_thinking_value_raises_the_same_error(self, value):
        with pytest.raises(ValueError) as non_mapping:
            resolve_thinking_display({"thinking": value}, PIN)
        with pytest.raises(ValueError) as bad_display:
            validate_thinking_display("loud")
        assert str(non_mapping.value) == str(bad_display.value).replace("'loud'", repr(value))

    @pytest.mark.parametrize(
        "version, allowed",
        [
            ("2.1.228", True),
            ("2.1.229", True),
            ("2.1.300", True),
            ("2.2.0", True),
            ("3.0.0", True),
            ("2.1.227", False),
            ("2.0.999", False),
            ("1.9.0", False),
            ("2.1.228-beta.1", False),
            ("2.1.228-rc.2", False),
            ("2.1.229-beta.1", True),
            (None, False),
            ("", False),
            ("not a version", False),
            ("2.1.x", False),
        ],
    )
    def test_the_flag_is_allowed_only_at_or_above_the_minimum(self, version, allowed):
        assert resolve_thinking_display({}, version).flag_allowed is allowed

    def test_it_is_pure(self, caplog, monkeypatch):
        monkeypatch.setattr(
            launcher.subprocess, "run", lambda *a, **k: pytest.fail("resolve ran a process")
        )
        with caplog.at_level(logging.DEBUG, logger=launcher.logger.name):
            resolve_thinking_display({}, None)
            resolve_thinking_display({}, "1.0.0")
        assert caplog.records == []


@pytest.mark.usefixtures("real_agent_cli")
class TestLaunchCliVersion:
    def test_the_pin_wins_without_probing(self, monkeypatch):
        monkeypatch.setattr(launcher, "cli_version_of", lambda exe: pytest.fail("probed"))
        assert launch_cli_version({"cli_version": PIN}, GLOBAL) == PIN

    def test_the_pin_wins_even_with_no_executable(self):
        assert launch_cli_version({"cli_version": PIN}, None) == PIN

    def test_unpinned_probes_the_executable(self, installed):
        probed = installed("2.1.250")
        assert launch_cli_version({}, GLOBAL) == "2.1.250"
        assert probed == [GLOBAL]

    def test_no_pin_flag_probes_the_executable_despite_a_pin(self, installed):
        probed = installed("2.1.250")
        assert launch_cli_version({"cli_version": PIN}, GLOBAL, no_pin=True) == "2.1.250"
        assert probed == [GLOBAL]

    def test_no_executable_unpinned_is_unknown(self, monkeypatch):
        monkeypatch.setattr(launcher, "cli_version_of", lambda exe: pytest.fail("probed"))
        assert launch_cli_version({}, None) is None
        assert launch_cli_version({"cli_version": PIN}, None, no_pin=True) is None

    def test_a_slow_probe_is_unknown(self, installed):
        import subprocess

        installed(subprocess.TimeoutExpired(["claude", "--version"], 15))
        assert launch_cli_version({}, GLOBAL) is None

    def test_a_bad_pin_raises(self):
        with pytest.raises(ValueError, match="claude_code.cli_version"):
            launch_cli_version({"cli_version": "latest"}, GLOBAL)

    def test_it_is_not_cached(self, installed):
        installed("2.1.250")
        assert launch_cli_version({}, GLOBAL) == "2.1.250"
        installed("2.1.260")
        assert launch_cli_version({}, GLOBAL) == "2.1.260"


class TestWarnThinkingFlagWithheld:
    def test_one_warning_naming_the_minimum_and_the_fix(self, caplog):
        with caplog.at_level(logging.WARNING, logger=launcher.logger.name):
            warn_thinking_flag_withheld("2.1.100")
        assert len(caplog.records) == 1
        message = caplog.records[0].getMessage()
        assert caplog.records[0].levelno == logging.WARNING
        assert THINKING_DISPLAY_MIN_CLI_VERSION in message
        assert "claude_code.cli_version" in message
        assert "2.1.100" in message
        assert "thinking" in message.lower()

    def test_an_unknown_version_is_said_so(self, caplog):
        with caplog.at_level(logging.WARNING, logger=launcher.logger.name):
            warn_thinking_flag_withheld(None)
        assert len(caplog.records) == 1
        assert "unknown" in caplog.records[0].getMessage()


@pytest.mark.usefixtures("real_agent_cli")
class TestProjectResolution:
    def test_pinned_project_resolves_path_version_and_display_from_one_probe(
        self, tmp_path, installed
    ):
        project = _project(
            tmp_path, f"claude_code:\n  cli_version: '{PIN}'\n  thinking:\n    display: omitted\n"
        )
        probed = installed(PIN)

        resolved = launcher._project_agent_cli(str(project))

        assert resolved == ResolvedAgentCli(GLOBAL, PIN, ThinkingDisplay("omitted", True))
        assert launcher.project_agent_cli_path(project) == GLOBAL
        assert launcher.project_thinking_display(project) == ThinkingDisplay("omitted", True)
        assert probed == [GLOBAL]

    def test_unpinned_project_probes_the_path_exactly_once(self, tmp_path, installed):
        project = _project(tmp_path, "claude_code: {}\n")
        probed = installed("2.1.250")

        for _ in range(3):
            assert launcher.project_agent_cli_path(project) == GLOBAL
            assert launcher.project_thinking_display(project) == ThinkingDisplay("summarized", True)

        assert probed == [GLOBAL]
        assert launcher._project_agent_cli(str(project)).version == "2.1.250"

    def test_no_config_at_all_resolves_the_default(self, tmp_path, installed):
        installed("2.1.250")
        assert launcher.project_thinking_display(tmp_path) == ThinkingDisplay("summarized", True)

    def test_an_old_unpinned_cli_withholds_the_flag_and_warns_once(
        self, tmp_path, installed, caplog
    ):
        project = _project(tmp_path, "claude_code: {}\n")
        installed("2.1.100")

        with caplog.at_level(logging.WARNING, logger=launcher.logger.name):
            for _ in range(3):
                assert launcher.project_thinking_display(project) == ThinkingDisplay(
                    "summarized", False
                )
                launcher.project_agent_cli_path(project)

        assert len(_withheld(caplog)) == 1

    def test_an_unreadable_unpinned_version_withholds_the_flag(self, tmp_path, installed, caplog):
        import subprocess

        project = _project(tmp_path, "claude_code: {}\n")
        installed(subprocess.TimeoutExpired(["claude", "--version"], 15))

        with caplog.at_level(logging.WARNING, logger=launcher.logger.name):
            assert launcher.project_agent_cli_path(project) == GLOBAL
            assert launcher.project_thinking_display(project).flag_allowed is False

        assert len(_withheld(caplog)) == 1

    def test_an_allowed_flag_warns_nothing(self, tmp_path, installed, caplog):
        project = _project(tmp_path, f"claude_code:\n  cli_version: '{PIN}'\n")
        installed(PIN)
        with caplog.at_level(logging.WARNING, logger=launcher.logger.name):
            launcher.project_thinking_display(project)
        assert _withheld(caplog) == []

    def test_each_project_warns_on_its_own(self, tmp_path_factory, installed, caplog):
        a = _project(tmp_path_factory.mktemp("a"), "claude_code: {}\n")
        b = _project(tmp_path_factory.mktemp("b"), "claude_code: {}\n")
        installed("2.1.100")
        with caplog.at_level(logging.WARNING, logger=launcher.logger.name):
            launcher.project_thinking_display(a)
            launcher.project_thinking_display(b)
            launcher.project_thinking_display(a)
        assert len(_withheld(caplog)) == 2

    def test_a_config_edit_changes_nothing_until_restart(self, tmp_path, installed):
        project = _project(
            tmp_path, f"claude_code:\n  cli_version: '{PIN}'\n  thinking:\n    display: omitted\n"
        )
        installed(PIN)
        first = launcher.project_thinking_display(project)

        (project / "config.yml").write_text("claude_code:\n  cli_version: '2.1.100'\n")

        assert launcher.project_thinking_display(project) == first
        assert launcher.project_agent_cli_path(project) == GLOBAL

    def test_a_bad_display_raises_and_is_not_remembered(self, tmp_path, installed):
        project = _project(
            tmp_path, f"claude_code:\n  cli_version: '{PIN}'\n  thinking:\n    display: loud\n"
        )
        installed(PIN)

        with pytest.raises(ValueError, match=r"claude_code\.thinking\.display"):
            launcher.project_thinking_display(project)
        with pytest.raises(ValueError, match=r"claude_code\.thinking\.display"):
            launcher.project_agent_cli_path(project)

        (project / "config.yml").write_text(f"claude_code:\n  cli_version: '{PIN}'\n")
        assert launcher.project_thinking_display(project) == ThinkingDisplay("summarized", True)

    def test_a_non_mapping_thinking_value_raises(self, tmp_path, installed):
        project = _project(tmp_path, "claude_code:\n  thinking: summarized\n")
        installed("2.1.250")
        with pytest.raises(ValueError, match=r"claude_code\.thinking\.display"):
            launcher.project_thinking_display(project)

    def test_a_refusal_is_not_remembered(self, tmp_path, installed):
        project = _project(tmp_path, f"claude_code:\n  cli_version: '{PIN}'\n")
        installed("2.1.100")
        with pytest.raises(AgentCliUnavailableError):
            launcher.project_thinking_display(project)

        installed(PIN)
        assert launcher.project_thinking_display(project) == ThinkingDisplay("summarized", True)


class TestSuiteStub:
    """The suite-wide stand-in covers the display, so a stubbed test never probes."""

    def test_the_stub_resolves_what_the_default_pin_resolves(self, tmp_path, monkeypatch):
        monkeypatch.setattr(launcher, "_probe_cli_version", lambda exe: pytest.fail("probed"))

        assert launcher.project_agent_cli_path(tmp_path) == STUB_AGENT_CLI
        assert launcher.project_thinking_display(tmp_path) == ThinkingDisplay("summarized", True)
        assert resolve_thinking_display({"cli_version": PIN}, PIN) == ThinkingDisplay(
            "summarized", True
        )
