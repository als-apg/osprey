"""The build and status each say a model warning once.

The provider resolver runs many times per verb: three times per render pass of a
build, twice per ``osprey status``. Its own record of an alias running the main
model is INFO, for the sinks. The verb that owns the terminal promotes the fact
from the resolved spec, through ``warn_fact`` in the build and as one trouble
line in status, so the operator reads it once however often the verb resolved.

One real exemplar build with ``provider: openai`` serves every test. The
build also pins one agent to an id the provider does not list, so it carries that
warning too.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from rich.console import Console

from osprey.cli import build_cmd
from osprey.cli.main import cli
from osprey.cli.phase_reporter import PhaseReporter, install_reporter
from osprey.deployment import status_display
from tests.cli.conftest import TerminalProbe

SUBSTITUTION = "haiku, sonnet, opus aliases run the main model gpt-6-sol"
UNLISTED = "'openai' does not list gpt-6-preview"

_WITNESS = "MODELWARNINGSWITNESSMARKER"


class _Capture(PhaseReporter):
    """A reporter whose console is a buffer, so printed lines are readable."""

    def __init__(self, console: Console) -> None:
        super().__init__(color=False)
        self._console = console

    def out(self) -> Console:
        return self._console


class _RecordSink(logging.Handler):
    """Every record the build emitted, kept whole so level and text stay paired."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _flowed(text: str) -> str:
    """``text`` with its whitespace collapsed, so console wrapping does not matter."""
    return " ".join(text.split())


@pytest.fixture(scope="module")
def openai_build(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One build of the exemplar repo switched to ``provider: openai``.

    Module-scoped because the build is the expensive part. The build's trouble
    stream is the reporter's console too, so one buffer holds every line the
    operator saw.
    """
    from osprey.cli import styles
    from osprey.cli.styles import osprey_theme
    from tests.fixtures.lifecycle_repo import EXEMPLAR_DIRNAME, build_exemplar_repo

    repo = build_exemplar_repo(tmp_path_factory.mktemp("openai") / EXEMPLAR_DIRNAME)
    profile = repo / "profile.yml"
    text = profile.read_text()
    assert text.count("\nprovider: anthropic\n") == 1
    text = text.replace("\nprovider: anthropic\n", "\nprovider: openai\n")
    example = "  # claude_code.agent_models.logbook-search: claude-sonnet-5\n"
    assert text.count(example) == 1
    profile.write_text(
        text.replace(example, "  claude_code.agent_models.logbook-search: gpt-6-preview\n")
    )

    buffer = StringIO()
    console = Console(
        file=buffer, width=200, force_terminal=False, legacy_windows=False, theme=osprey_theme
    )
    sink = _RecordSink()
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(sink)
    root.setLevel(logging.DEBUG)
    previous_cwd = Path.cwd()
    os.chdir(repo)
    previous_reporter = install_reporter(_Capture(console))
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(styles, "err_console", console)
            result = CliRunner().invoke(build_cmd.build, ["--skip-deps", "--skip-lifecycle"])
    finally:
        install_reporter(previous_reporter)
        os.chdir(previous_cwd)
        root.removeHandler(sink)
        root.setLevel(previous_level)

    assert result.exit_code == 0, result.output
    # Status resolves telemetry with the provider, and the exemplar's ingest
    # token has no default: without it status reads no spec at all.
    (repo / ".env").write_text("ZO_INGEST_SA_TOKEN=probe-token\n")
    return SimpleNamespace(repo=repo, printed=buffer.getvalue(), records=sink.records)


class TestTheBuildSaysItOnce:
    """Every render pass resolves the provider; the build states each fact once."""

    def test_the_substitution_is_one_line_naming_the_aliases_and_the_main_model(self, openai_build):
        flowed = _flowed(openai_build.printed)
        assert flowed.count("run the main model") == 1, flowed
        assert SUBSTITUTION in flowed
        assert "claude_code.aliases.<name>" in flowed

    def test_the_configured_ids_outside_the_list_are_named_on_one_line(self, openai_build):
        flowed = _flowed(openai_build.printed)
        assert flowed.count("does not list") == 1, flowed
        assert UNLISTED in flowed

    def test_each_promoted_warning_keeps_one_record(self, openai_build):
        warnings = [r.getMessage() for r in openai_build.records if r.levelno == logging.WARNING]
        assert len([m for m in warnings if SUBSTITUTION in m]) == 1
        assert len([m for m in warnings if UNLISTED in m]) == 1

    def test_the_resolver_records_it_below_the_gate(self, openai_build):
        records = [
            r
            for r in openai_build.records
            if r.name == "osprey.build.claude_code_resolver" and SUBSTITUTION in r.getMessage()
        ]
        assert records
        assert all(r.levelno == logging.INFO for r in records)


@pytest.fixture
def no_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty deployment: ``ps`` finds nothing, and nothing else may run."""
    monkeypatch.setattr(
        status_display,
        "get_ps_command",
        lambda config, all_containers=False: ["docker", "ps", "-a", "--format", "json"],
    )
    monkeypatch.setattr(status_display, "get_runtime_command", lambda config=None: ["docker"])

    def _run(cmd, **kwargs):
        argv = list(cmd)
        if argv[:2] == ["docker", "ps"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps([]), "")
        if argv[:2] == ["docker", "volume"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(f"a read-only verb ran: {argv}")

    monkeypatch.setattr(status_display.subprocess, "run", _run)


@pytest.mark.usefixtures("no_runtime")
def test_status_says_it_once_and_the_log_handler_paints_nothing(
    openai_build, terminal_probe: TerminalProbe, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("osprey.utils.config.load_project_dotenv", lambda *a, **k: None)
    result = CliRunner().invoke(cli, ["status", "--agents", "--repo", str(openai_build.repo)])
    assert result.exit_code == 0, result.output

    flowed = _flowed(result.output)
    assert flowed.count("run the main model") == 1, flowed
    assert SUBSTITUTION in flowed
    assert "gpt-6-sol (main model)" in flowed

    assert "run the main model" not in terminal_probe.rendered_text
    assert any(SUBSTITUTION in m for m in terminal_probe.messages)

    # Armed witness: an ERROR is above the gate on every path, so its absence
    # would mean the probe console was never reachable.
    logging.getLogger("tests.model_warnings_report").error(_WITNESS)
    assert _WITNESS in terminal_probe.rendered_text
