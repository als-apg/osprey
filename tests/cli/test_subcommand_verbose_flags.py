"""Every subcommand ``-v``/``--verbose`` lifts the altitude gate for its own run.

The CLI's altitude gate (:mod:`osprey.cli.altitude`) is installed by the group
callback and drops INFO-and-below at the rendering handler, so a normal run
reads as a report. A subcommand's own verbose flag is the operator asking for
the transcript of *that* run, and it has to mean the same thing everywhere:
``audit``, ``health``, and the three channel-finder flags.

Each flag is pinned twice, in the shape the ``terminal_probe`` fixture calls for:

* under the flag, an INFO record emitted from inside the running command is
  painted by the handler;
* without it, the same record is not painted — asserted next to an **armed
  witness**, a WARNING from the same logger in the same run. Without that
  witness an absence assertion passes for a console nothing ever reached.

Each flag's own display semantics are unchanged and are pinned by the suites
that own them (``test_audit_display.py``, ``test_health_cmd_output.py``,
``test_channel_finder_logging.py``); what is new here is only the gate lift.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.altitude import gate_installed
from osprey.cli.main import cli
from tests.cli.conftest import TerminalProbe

INFO_MARKER = "VERBOSEFLAGINFOMARKER"
WARNING_MARKER = "VERBOSEFLAGWARNMARKER"


def _log_markers(logger_name: str) -> None:
    """Emit the needle and its armed witness on *logger_name*."""
    logger = logging.getLogger(logger_name)
    logger.info(INFO_MARKER)
    logger.warning(WARNING_MARKER)


@pytest.fixture(autouse=True)
def stub_project_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the group callback from writing an ancestor ``.env`` into the process.

    ``load_project_dotenv`` runs for every ``osprey`` invocation and its writes
    to ``os.environ`` outlive the test that triggered them.
    """
    monkeypatch.setattr("osprey.utils.config.load_project_dotenv", lambda *a, **k: None)


@pytest.fixture
def unpinned_osprey_logger() -> Iterator[None]:
    """Hand back the ``osprey`` logger's level, and start from a known one.

    ``channel-finder benchmark --verbose`` raises that logger to DEBUG as its
    emission floor. Levels are process-global, so the pin has to be undone or it
    leaks into every later test in the session.
    """
    logger = logging.getLogger("osprey")
    saved = logger.level
    logger.setLevel(logging.NOTSET)
    try:
        yield
    finally:
        logger.setLevel(saved)


# --------------------------------------------------------------------------- #
# osprey audit
# --------------------------------------------------------------------------- #

AUDIT_LOGGER = "osprey.audit.verbose_probe"


@pytest.fixture
def run_audit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Invoke ``osprey audit`` with the reviewer agent replaced by a stub.

    The stub logs from inside the command body — after the flag has had its
    chance to lift the gate — and hands back a report document the real parsing
    path accepts, so the run reaches its normal end rather than a failure path.
    """
    monkeypatch.setattr("osprey.cli.audit_cmd._SDK_AVAILABLE", True)

    async def _fake_run_audit(*args: object, **kwargs: object) -> tuple[str, float, int]:
        _log_markers(AUDIT_LOGGER)
        document = json.dumps(
            {"summary": "Nothing alarming", "overall_risk": "low", "findings": []}
        )
        return document, 0.01, 2

    monkeypatch.setattr("osprey.cli.audit_cmd._run_audit", _fake_run_audit)

    target = tmp_path / "audited-project"
    target.mkdir()
    (target / "config.yml").write_text("project_name: audited\n", encoding="utf-8")
    runner = CliRunner()

    def _run(*args: str):
        return runner.invoke(cli, ["audit", str(target), *args])

    return _run


class TestAuditVerbose:
    """``audit -v`` is a display flag today; it lifts the gate as well."""

    def test_verbose_renders_the_transcript(self, run_audit, terminal_probe: TerminalProbe):
        result = run_audit("-v")

        assert result.exit_code == 0, result.output
        assert INFO_MARKER in terminal_probe.rendered_text
        assert gate_installed() is False

    def test_a_default_run_renders_no_transcript(self, run_audit, terminal_probe: TerminalProbe):
        result = run_audit()

        assert result.exit_code == 0, result.output
        assert gate_installed() is True
        assert WARNING_MARKER in terminal_probe.rendered_text
        assert INFO_MARKER not in terminal_probe.rendered_text
        assert INFO_MARKER in terminal_probe.messages


# --------------------------------------------------------------------------- #
# osprey health
# --------------------------------------------------------------------------- #

HEALTH_LOGGER = "osprey.health.verbose_probe"


@pytest.fixture
def run_health(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Invoke ``osprey health`` over a throwaway project, checks stubbed out.

    ``load_config`` is wrapped rather than replaced: the markers are emitted at
    the point of the real call, inside ``_quiet_run_logs``, which is where a
    gate lift has to survive health's own silencing to be worth anything. The
    marker logger is deliberately not one of ``_NOISY_LOADER_LOGGERS``, so what
    the assertions read is the gate and not that silencing.
    """
    import osprey.cli.health_cmd as health_cmd

    real_load_config = health_cmd.load_config

    def _noisy_load_config(*args: object, **kwargs: object):
        _log_markers(HEALTH_LOGGER)
        return real_load_config(*args, **kwargs)

    monkeypatch.setattr(health_cmd, "load_config", _noisy_load_config)

    project = tmp_path / "health-project"
    project.mkdir()
    (project / "config.yml").write_text("project_name: verbose_flags\n", encoding="utf-8")
    runner = CliRunner()

    def _run(*args: str):
        from osprey.health.models import CheckReport

        empty = CheckReport(results=[], elapsed_ms=1.0, deadline_hit=False)
        with patch("osprey.cli.health_cmd._run_suite", AsyncMock(return_value=empty)):
            return runner.invoke(cli, ["health", "--project", str(project), *args])

    return _run


class TestHealthVerbose:
    """``health -v`` keeps its detail meaning and gains the gate lift."""

    def test_verbose_renders_the_transcript(self, run_health, terminal_probe: TerminalProbe):
        run_health("-v")

        assert INFO_MARKER in terminal_probe.rendered_text
        assert gate_installed() is False

    def test_a_default_run_renders_no_transcript(self, run_health, terminal_probe: TerminalProbe):
        run_health()

        assert gate_installed() is True
        assert WARNING_MARKER in terminal_probe.rendered_text
        assert INFO_MARKER not in terminal_probe.rendered_text
        assert INFO_MARKER in terminal_probe.messages

    def test_json_stdout_is_still_one_document_under_verbose(self, run_health):
        """The lift must not put a log line where a machine reads a document.

        The handler renders at stderr and ``--json`` silences every root handler
        for the loading stretch, so a lifted gate cannot reach stdout.
        """
        result = run_health("-v", "--json")

        assert json.loads(result.stdout)["results"] == []


# --------------------------------------------------------------------------- #
# osprey channel-finder (group --verbose, validate --verbose)
# --------------------------------------------------------------------------- #

VALIDATE_LOGGER = "channel_finder.validate.verbose_probe"


@pytest.fixture
def run_validate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Invoke ``osprey channel-finder validate`` through the real CLI group.

    Through ``cli`` so the group callback installs the gate in the first place.
    ``_setup_config`` is stubbed (no built project is needed), the registry
    build underneath ``_initialize_registry`` is stubbed rather than the
    function itself, and ``run_validation`` — the command's own work — is what
    emits the markers, so the gate is asked its question in situ.
    """
    monkeypatch.setattr("osprey.cli.channel_finder_cmd._setup_config", lambda *a, **k: None)
    monkeypatch.setattr("osprey.registry.initialize_registry", lambda *a, **k: None)

    def _run_validation(**kwargs: object) -> int:
        _log_markers(VALIDATE_LOGGER)
        return 0

    monkeypatch.setattr(
        "osprey.services.channel_finder.tools.validate_database.run_validation",
        _run_validation,
    )

    database = tmp_path / "channel_db.json"
    database.write_text(
        '{"channels": [{"template": false, "channel": "CH1", '
        '"address": "PV:CH1", "description": "Test channel"}]}',
        encoding="utf-8",
    )
    runner = CliRunner()

    def _run(*group_args: str, subcommand_args: tuple[str, ...] = ()):
        return runner.invoke(
            cli,
            [
                "channel-finder",
                *group_args,
                "validate",
                "--database",
                str(database),
                "--pipeline",
                "in_context",
                *subcommand_args,
            ],
        )

    return _run


class TestChannelFinderGroupVerbose:
    """``channel-finder -v <anything>`` lifts the gate for the whole subcommand."""

    def test_verbose_renders_the_transcript(self, run_validate, terminal_probe: TerminalProbe):
        result = run_validate("-v")

        assert result.exit_code == 0, result.output
        assert INFO_MARKER in terminal_probe.rendered_text
        assert gate_installed() is False

    def test_a_default_run_renders_no_transcript(self, run_validate, terminal_probe: TerminalProbe):
        result = run_validate()

        assert result.exit_code == 0, result.output
        assert gate_installed() is True
        assert WARNING_MARKER in terminal_probe.rendered_text
        assert INFO_MARKER not in terminal_probe.rendered_text
        assert INFO_MARKER in terminal_probe.messages


class TestChannelFinderValidateVerbose:
    """``validate -v`` carries its own lift, without the group flag."""

    def test_verbose_renders_the_transcript(self, run_validate, terminal_probe: TerminalProbe):
        result = run_validate(subcommand_args=("-v",))

        assert result.exit_code == 0, result.output
        assert INFO_MARKER in terminal_probe.rendered_text
        assert gate_installed() is False

    def test_a_default_run_renders_no_transcript(self, run_validate, terminal_probe: TerminalProbe):
        assert run_validate().exit_code == 0

        assert gate_installed() is True
        assert WARNING_MARKER in terminal_probe.rendered_text
        assert INFO_MARKER not in terminal_probe.rendered_text


# --------------------------------------------------------------------------- #
# osprey channel-finder benchmark
# --------------------------------------------------------------------------- #

BENCHMARK_LOGGER = "osprey.benchmark.verbose_probe"


@pytest.fixture
def run_benchmark(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Invoke ``osprey channel-finder benchmark`` with a stub runner.

    The stub logs while loading queries and then refuses to run them, so the
    command reaches its own work — and the gate decision that precedes it —
    without a provider, a dataset, or a network call. The refusal lands in the
    command's own ``except`` arm, which aborts: exit code 1 is the run getting
    that far, not the flag failing.
    """
    from osprey.services.channel_finder.benchmarks import runner as runner_module

    class _StubRunner:
        provider = "stub"
        model = "stub/model"

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def load_queries(self) -> list[str]:
            _log_markers(BENCHMARK_LOGGER)
            return ["one query"]

        async def run_queries(self, **kwargs: object):
            raise RuntimeError("the stub runner runs nothing")

    monkeypatch.setattr(runner_module, "BenchmarkRunner", _StubRunner)

    project = tmp_path / "benchmark-project"
    project.mkdir()
    (project / "config.yml").write_text("project_name: verbose_flags\n", encoding="utf-8")
    runner = CliRunner()

    def _run(*args: str):
        return runner.invoke(
            cli,
            [
                "channel-finder",
                "--project",
                str(project),
                "benchmark",
                "--model",
                "stub/model",
                *args,
            ],
        )

    return _run


class TestChannelFinderBenchmarkVerbose:
    """``benchmark -v`` composes the gate lift with its DEBUG emission floor."""

    def test_verbose_renders_the_transcript(
        self, run_benchmark, unpinned_osprey_logger, terminal_probe: TerminalProbe
    ):
        result = run_benchmark("-v")

        assert result.exit_code == 1, result.output
        assert INFO_MARKER in terminal_probe.rendered_text
        assert gate_installed() is False

    def test_a_default_run_renders_no_transcript(
        self, run_benchmark, unpinned_osprey_logger, terminal_probe: TerminalProbe
    ):
        result = run_benchmark()

        assert result.exit_code == 1, result.output
        assert gate_installed() is True
        assert WARNING_MARKER in terminal_probe.rendered_text
        assert INFO_MARKER not in terminal_probe.rendered_text
        assert INFO_MARKER in terminal_probe.messages

    def test_verbose_still_lowers_the_emission_floor(
        self, run_benchmark, unpinned_osprey_logger, terminal_probe: TerminalProbe
    ):
        """The lift is the render half; the level floor stays what it was.

        Lifting the gate alone would surface INFO and no more — the framework
        logger's DEBUG floor is a separate line, and both are needed for a
        DEBUG record to be painted.
        """
        run_benchmark("-v")

        assert logging.getLogger("osprey").level == logging.DEBUG
