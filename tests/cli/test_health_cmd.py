"""Contract tests for the ``osprey health`` CLI command.

The command was rebuilt as a thin Click wrapper over the ``osprey.health``
framework (see ``.claude/plans/configurable-health-system-p1-core-cli/PROPOSAL.md``
§CLI rebuild). This module pins the *CLI-level contracts* — the observable
behaviors a caller (human, ``--json`` consumer, or CI) depends on — rather than
the internals of any single check, which are owned by the ``osprey.health.*``
module tests.

Contracts pinned here
---------------------
* Exit-code mapping 0/1/2/3/130 (ok / warnings / errors / unexpected wrapper
  failure / interrupt).
* ``--basic`` hidden alias: no-op with a deprecation notice on **stderr**, never
  on stdout.
* ``--full`` gating: on_demand categories (``model_chat``, ``claude_cli_pinned``)
  run only with ``--full``; ``--category`` selection never elevates cost class.
* ``--category`` validation: unknown name on a valid config → ``UsageError``
  (exit 2); a non-core name under a *broken* config passes through (report still
  renders, no ``UsageError``), and the ``configuration`` category is force-included
  so config-error rows always surface.
* ``claude_cli_pinned`` on an unpinned but valid config → exit 0 skip row.
* Config-load failures degrade into ``configuration`` error rows (never a crash):
  missing file → ``config_file_exists``; bad YAML → ``yaml_valid``; invalid
  ``health:`` → ``health_config``; the full report still renders alongside.
* No container runtime → a single ``skip`` row (exit 0 contribution).
* ``--project`` from a foreign cwd resolves that project's ``config.yml`` and
  ``.env`` (env-var resolution canary).
* ``--json`` is machine-clean: stdout is exactly the report JSON, round-trips
  through ``json.loads``, and carries the locked wire-shape keys — even when the
  config load logs an ERROR (loader chatter must not corrupt stdout).
* Subprocess no-hang: a never-returning plugin check still lets the process exit
  within the suite deadline with the correct code (the daemon-thread / ``os._exit``
  guarantee). This *must* be a real subprocess — in-process it would kill pytest.

Deliberately dropped from the old 749-line suite
------------------------------------------------
The previous file tested a since-deleted ``HealthChecker`` class and its
``HealthCheckResult`` value object. Those were implementation-detail unit tests of
a class that no longer exists; the *behaviors* they approximated are now pinned
either by the contract tests above or by the ``osprey.health.*`` module tests that
own each check. Consciously dropped assertions, and why:

* ``HealthCheckResult`` construction/repr/status — deleted type; replaced by
  ``osprey.health.models.CheckResult`` (owned by the models tests).
* ``HealthChecker`` init/options/``add_result``/``results``/``config`` state —
  deleted class; the CLI no longer holds mutable checker state.
* ``check_configuration`` missing/valid/empty/invalid-YAML unit calls — the
  configuration *rows* are now pinned here via ``--json`` (``config_file_exists``,
  ``yaml_valid``, ``health_config``); row-message internals belong to
  ``core/configuration`` tests.
* ``check_file_system`` / ``check_python_environment`` per-row unit calls
  (``disk_space``, ``env_file``, ``python_version``, ``virtual_environment``) —
  owned by ``core/file_system`` and ``core/python_environment`` tests; here we
  only assert these config-independent categories still render.
* ``check_containers`` docker/podman ``--version`` mocking — owned by
  ``core/containers`` + the container probe tests; here we pin only the
  no-runtime skip contract.
* ``check_api_providers`` registry mocking — owned by ``core/providers`` + the
  provider-canary tests.
* ``_check_timezone`` UTC/configured/env-var unit calls — owned by
  ``core/configuration`` tests (the timezone row); the CLI does not special-case it.
* ``check_claude_cli_version`` pinned-match/mismatch/npx-missing/unpinned subprocess
  mocking — owned by ``core/claude_cli`` tests; here we pin only the CLI-visible
  unpinned-skip and ``--full`` gating contracts.
* ``display_results`` all-passing/warnings/errors/verbose calls — rendering is now
  ``osprey.health.render`` (owned by the render tests); the CLI wires it and we
  assert the stdout/stderr split, not the glyph layout.
* ``check_openobserve`` healthz/retention unit calls — owned by
  ``core/openobserve`` tests.
* The old exit-code tests patched ``HealthChecker`` to inject results; the mapping
  is now a property of ``CheckReport`` and is pinned here by injecting a report at
  the ``_run_suite`` boundary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.health_cmd import health
from osprey.health.models import CheckReport, CheckResult, Status

# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def cli_runner() -> CliRunner:
    """A Click CLI test runner (stdout and stderr are captured separately)."""
    return CliRunner()


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot and restore ``os.environ`` around every test.

    The command loads the project's ``.env`` into ``os.environ`` with override
    semantics; run in-process, that would otherwise leak canary variables into
    sibling tests. This guarantees a pristine environment regardless.
    """
    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture(autouse=True)
def _guard_os_exit():
    """Force the CLI's normal ``sys.exit`` path for in-process invocations.

    When a sync check is abandoned after a timeout, the command falls back to
    ``os._exit`` — which, in an in-process ``CliRunner`` test, would terminate the
    pytest process itself. The daemon-thread / ``os._exit`` guarantee is instead
    pinned by the real-subprocess no-hang test; every in-process test pins
    ``abandoned_count`` at 0 so the caught ``SystemExit`` carries the exit code.
    """
    with patch("osprey.health.offload.abandoned_count", return_value=0):
        yield


def _write_config(project: Path, body: str) -> Path:
    """Write ``config.yml`` into *project* (created if needed) and return its path."""
    project.mkdir(parents=True, exist_ok=True)
    config_path = project / "config.yml"
    config_path.write_text(body)
    return config_path


# A minimal but valid config: parses, has a ``models`` section, references no
# providers or deployed services (so no network or container I/O is attempted).
_VALID_CONFIG = """\
project_name: test_project
models:
  python_code_generator:
    provider: mock
    model_id: mock-model
"""

# A valid config with no models at all — used to exercise ``model_chat``'s
# zero-item short-circuit under ``--full`` without issuing a real completion.
_NO_MODELS_CONFIG = "project_name: no_models\n"


@pytest.fixture
def valid_project(tmp_path: Path) -> Path:
    """A project directory holding a minimal valid ``config.yml``."""
    project = tmp_path / "proj"
    _write_config(project, _VALID_CONFIG)
    return project


def _result(
    name: str, status: Status, *, category: str = "demo", message: str = "msg"
) -> CheckResult:
    return CheckResult(name=name, category=category, status=status, message=message)


def _report(
    *results: CheckResult, elapsed_ms: float = 1.5, deadline_hit: bool = False
) -> CheckReport:
    return CheckReport(results=list(results), elapsed_ms=elapsed_ms, deadline_hit=deadline_hit)


def _patch_suite(report: CheckReport):
    """Patch the CLI's ``_run_suite`` boundary to return *report* without running checks."""
    return patch("osprey.cli.health_cmd._run_suite", AsyncMock(return_value=report))


def _no_container_runtime():
    """Patch the containers category so it degrades to its no-runtime skip row."""
    return patch(
        "osprey.health.core.containers.get_runtime_command",
        side_effect=RuntimeError("no container runtime"),
    )


def _find(results: list[dict], name: str) -> dict | None:
    """Return the first wire-shape result row with ``name``, or ``None``."""
    return next((r for r in results if r.get("name") == name), None)


# --------------------------------------------------------------------------- #
# Exit-code mapping (0 / 1 / 2 / 3 / 130)
# --------------------------------------------------------------------------- #


class TestExitCodes:
    """The CLI's process exit code follows the report / failure class."""

    def test_exit_code_all_ok_is_0(self, cli_runner, valid_project):
        report = _report(_result("a", Status.OK), _result("b", Status.OK))
        with _patch_suite(report):
            result = cli_runner.invoke(health, ["--project", str(valid_project)])
        assert result.exit_code == 0

    def test_exit_code_warnings_only_is_1(self, cli_runner, valid_project):
        report = _report(_result("a", Status.OK), _result("b", Status.WARNING))
        with _patch_suite(report):
            result = cli_runner.invoke(health, ["--project", str(valid_project)])
        assert result.exit_code == 1

    def test_exit_code_errors_is_2(self, cli_runner, valid_project):
        report = _report(_result("a", Status.OK), _result("b", Status.ERROR))
        with _patch_suite(report):
            result = cli_runner.invoke(health, ["--project", str(valid_project)])
        assert result.exit_code == 2

    def test_exit_code_unexpected_wrapper_failure_is_3(self, cli_runner):
        # A failure anywhere in the wrapper (here: project resolution) is exit 3.
        with patch(
            "osprey.cli.project_utils.resolve_project_path",
            side_effect=RuntimeError("boom"),
        ):
            result = cli_runner.invoke(health, [])
        assert result.exit_code == 3

    def test_exit_code_keyboard_interrupt_is_130(self, cli_runner):
        with patch(
            "osprey.cli.project_utils.resolve_project_path",
            side_effect=KeyboardInterrupt,
        ):
            result = cli_runner.invoke(health, [])
        assert result.exit_code == 130


# --------------------------------------------------------------------------- #
# --basic hidden alias
# --------------------------------------------------------------------------- #


class TestBasicAlias:
    """``--basic`` is a hidden, deprecated no-op that warns only on stderr."""

    def test_basic_flag_is_hidden_in_help(self, cli_runner):
        result = cli_runner.invoke(health, ["--help"])
        assert result.exit_code == 0
        # Hidden option: present as a flag but never advertised in --help.
        assert "--basic" not in result.output
        assert "--full" in result.output
        assert "--json" in result.output
        assert "--category" in result.output

    def test_basic_deprecation_on_stderr_not_stdout(self, cli_runner, valid_project):
        report = _report(_result("a", Status.OK))
        with _patch_suite(report):
            result = cli_runner.invoke(health, ["--project", str(valid_project), "--basic"])
        assert result.exit_code == 0
        assert "deprecated" in result.stderr.lower()
        assert "deprecated" not in result.stdout.lower()

    def test_basic_is_noop(self, cli_runner, valid_project):
        report = _report(_result("a", Status.WARNING))
        with _patch_suite(report):
            without = cli_runner.invoke(health, ["--project", str(valid_project)])
        with _patch_suite(report):
            with_basic = cli_runner.invoke(health, ["--project", str(valid_project), "--basic"])
        # Same exit code and same report on stdout — the flag changes nothing.
        assert without.exit_code == with_basic.exit_code == 1
        assert without.stdout == with_basic.stdout


# --------------------------------------------------------------------------- #
# --full gating (on_demand categories) and --category non-elevation
# --------------------------------------------------------------------------- #


class TestFullGating:
    """``--full`` is the sole on_demand gate; ``--category`` never elevates cost."""

    def test_full_gating_model_chat_skipped_without_full(self, cli_runner, tmp_path):
        project = tmp_path / "proj"
        _write_config(project, _NO_MODELS_CONFIG)
        result = cli_runner.invoke(
            health, ["--project", str(project), "--json", "--category", "model_chat"]
        )
        payload = json.loads(result.stdout)
        row = _find(payload["results"], "model_chat")
        assert row is not None
        assert row["status"] == "skip"
        assert "--full" in row["message"]

    def test_full_gating_model_chat_runs_with_full(self, cli_runner, tmp_path):
        # With --full the on_demand category actually executes; a zero-model
        # config short-circuits to its own skip row (no live completion issued).
        project = tmp_path / "proj"
        _write_config(project, _NO_MODELS_CONFIG)
        result = cli_runner.invoke(
            health, ["--project", str(project), "--json", "--full", "--category", "model_chat"]
        )
        payload = json.loads(result.stdout)
        row = _find(payload["results"], "model_chat")
        assert row is not None
        assert row["status"] == "skip"
        assert row["message"] == "no models configured"
        assert "--full" not in row["message"]
        # The zero-model guard short-circuits before the deadline machinery: a
        # skip-only report is exit 0 with no deadline hit.
        assert result.exit_code == 0
        assert payload["deadline_hit"] is False

    def test_full_category_does_not_elevate_cost(self, cli_runner, valid_project):
        # Selecting an on_demand category without --full must not run it.
        result = cli_runner.invoke(
            health, ["--project", str(valid_project), "--json", "--category", "claude_cli_pinned"]
        )
        payload = json.loads(result.stdout)
        row = _find(payload["results"], "claude_cli_pinned")
        assert row is not None
        assert row["status"] == "skip"
        assert "--full" in row["message"]


# --------------------------------------------------------------------------- #
# --category validation and claude_cli_pinned unpinned skip
# --------------------------------------------------------------------------- #


class TestCategorySelection:
    """``--category`` resolution: unknown-name handling and the pinned-skip row."""

    def test_unknown_category_on_valid_config_is_usage_error(self, cli_runner, valid_project):
        result = cli_runner.invoke(
            health, ["--project", str(valid_project), "--category", "does_not_exist"]
        )
        assert result.exit_code == 2
        assert "Unknown health categor" in result.output

    def test_unknown_category_with_missing_config_renders_no_usage_error(
        self, cli_runner, tmp_path
    ):
        # No config.yml: category names cannot be validated, so a non-core name
        # passes through — the report renders (exit 2 from config errors) rather
        # than raising a UsageError.
        project = tmp_path / "empty"
        project.mkdir()
        with _no_container_runtime():
            result = cli_runner.invoke(
                health, ["--project", str(project), "--json", "--category", "does_not_exist"]
            )
        assert result.exit_code == 2
        assert "Unknown health categor" not in result.output
        payload = json.loads(result.stdout)
        # configuration is force-included under a config failure even when filtered.
        config_row = _find(payload["results"], "config_file_exists")
        assert config_row is not None
        assert config_row["status"] == "error"

    def test_category_claude_cli_pinned_unpinned_skip(self, cli_runner, valid_project):
        # A valid config without a claude_code.cli_version pin: the on_demand
        # category runs under --full but returns a skip row (no npx invoked).
        result = cli_runner.invoke(
            health,
            [
                "--project",
                str(valid_project),
                "--json",
                "--full",
                "--category",
                "claude_cli_pinned",
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        row = _find(payload["results"], "claude_cli_pinned")
        assert row is not None
        assert row["status"] == "skip"
        assert "no cli_version pin configured" in row["message"]


# --------------------------------------------------------------------------- #
# Config-load failure degradation (missing / bad YAML / bad health:)
# --------------------------------------------------------------------------- #


class TestConfigFailureDegradation:
    """A config-load failure degrades into report rows; the report still renders."""

    def test_missing_config_yields_config_file_exists_error(self, cli_runner, tmp_path):
        project = tmp_path / "empty"
        project.mkdir()
        with _no_container_runtime():
            result = cli_runner.invoke(health, ["--project", str(project), "--json"])
        assert result.exit_code == 2
        payload = json.loads(result.stdout)
        row = _find(payload["results"], "config_file_exists")
        assert row is not None and row["status"] == "error"
        # Full report still renders: config-independent categories are present.
        categories = {r["category"] for r in payload["results"]}
        assert "file_system" in categories
        assert "python_environment" in categories

    def test_malformed_yaml_yields_yaml_valid_error(self, cli_runner, tmp_path):
        project = tmp_path / "proj"
        _write_config(project, "invalid: yaml: content:\n")
        with _no_container_runtime():
            result = cli_runner.invoke(health, ["--project", str(project), "--json"])
        assert result.exit_code == 2
        payload = json.loads(result.stdout)
        assert _find(payload["results"], "config_file_exists")["status"] == "ok"
        yaml_row = _find(payload["results"], "yaml_valid")
        assert yaml_row is not None and yaml_row["status"] == "error"

    def test_malformed_health_section_yields_health_config_error(self, cli_runner, tmp_path):
        project = tmp_path / "proj"
        _write_config(
            project,
            "project_name: bad_health\nhealth:\n  suite_timeout_s: not-a-number\n",
        )
        with _no_container_runtime():
            result = cli_runner.invoke(health, ["--project", str(project), "--json"])
        assert result.exit_code == 2
        payload = json.loads(result.stdout)
        assert _find(payload["results"], "config_file_exists")["status"] == "ok"
        assert _find(payload["results"], "yaml_valid")["status"] == "ok"
        health_row = _find(payload["results"], "health_config")
        assert health_row is not None and health_row["status"] == "error"


# --------------------------------------------------------------------------- #
# No container runtime → skip
# --------------------------------------------------------------------------- #


class TestContainerRuntime:
    """Absent container runtime contributes a skip row, not an error."""

    def test_no_container_runtime_is_skip(self, cli_runner, valid_project):
        with _no_container_runtime():
            result = cli_runner.invoke(
                health, ["--project", str(valid_project), "--json", "--category", "containers"]
            )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        row = _find(payload["results"], "container_runtime")
        assert row is not None and row["status"] == "skip"
        assert "no container runtime available" in row["message"]


# --------------------------------------------------------------------------- #
# --project from a foreign cwd (config + .env resolution canary)
# --------------------------------------------------------------------------- #


class TestProjectResolution:
    """``--project`` resolves config and ``.env`` from the named directory."""

    def test_project_resolves_config_and_env_from_foreign_cwd(
        self, cli_runner, tmp_path, monkeypatch
    ):
        project = tmp_path / "proj"
        _write_config(
            project,
            "project_name: canary\n"
            "models:\n"
            "  python_code_generator:\n"
            "    provider: mock\n"
            "    model_id: ${OSPREY_HEALTH_CANARY}\n",
        )
        # The canary variable is defined only in the project's .env — never in the
        # ambient environment — so an "ok" env-var row proves the project .env was
        # loaded even though the process runs from an unrelated cwd.
        (project / ".env").write_text("OSPREY_HEALTH_CANARY=resolved-from-project-env\n")
        assert "OSPREY_HEALTH_CANARY" not in os.environ

        foreign = tmp_path / "foreign"
        foreign.mkdir()
        monkeypatch.chdir(foreign)

        with _no_container_runtime():
            result = cli_runner.invoke(
                health, ["--project", str(project), "--json", "--category", "configuration"]
            )
        payload = json.loads(result.stdout)

        env_row = _find(payload["results"], "environment_variables")
        assert env_row is not None and env_row["status"] == "ok"

        # Config itself was read from the --project directory, not the cwd.
        config_row = _find(payload["results"], "config_file_exists")
        assert config_row is not None and config_row["status"] == "ok"
        assert str(project) in config_row["message"]


# --------------------------------------------------------------------------- #
# --json machine-clean wire contract
# --------------------------------------------------------------------------- #


class TestJsonOutput:
    """``--json`` emits exactly one JSON document on stdout, in the locked shape."""

    _WIRE_KEYS = {
        "summary",
        "ok",
        "warnings",
        "errors",
        "skips",
        "total",
        "elapsed_ms",
        "deadline_hit",
        "results",
    }

    def test_json_stdout_roundtrips_to_report(self, cli_runner, valid_project):
        report = _report(
            _result("a", Status.OK),
            _result("b", Status.WARNING),
            _result("c", Status.SKIP),
        )
        with _patch_suite(report):
            result = cli_runner.invoke(health, ["--project", str(valid_project), "--json"])
        assert result.exit_code == 1
        payload = json.loads(result.stdout)
        assert payload == report.to_dict()

    def test_json_wire_shape_keys(self, cli_runner, valid_project):
        report = _report(_result("a", Status.OK))
        with _patch_suite(report):
            result = cli_runner.invoke(health, ["--project", str(valid_project), "--json"])
        payload = json.loads(result.stdout)
        assert set(payload) == self._WIRE_KEYS

    def test_json_stdout_is_single_document(self, cli_runner, valid_project):
        # Exactly one JSON document, nothing else, on stdout (spinner is on stderr).
        report = _report(_result("a", Status.OK))
        with _patch_suite(report):
            result = cli_runner.invoke(health, ["--project", str(valid_project), "--json"])
        stripped = result.stdout.strip()
        assert stripped.startswith("{") and stripped.endswith("}")
        assert "\n" not in stripped  # a single compact line

    def test_json_clean_despite_config_load_error(self, cli_runner, tmp_path):
        # Regression: the loader logs bad-YAML failures at ERROR through a
        # stdout-bound handler. Under --json those logs must be silenced so stdout
        # stays a single parseable JSON document.
        project = tmp_path / "proj"
        _write_config(project, "invalid: yaml: content:\n")
        with _no_container_runtime():
            result = cli_runner.invoke(health, ["--project", str(project), "--json"])
        payload = json.loads(result.stdout)  # must not raise
        assert _find(payload["results"], "yaml_valid")["status"] == "error"


# --------------------------------------------------------------------------- #
# Subprocess no-hang guarantee (must be a real process, not in-process)
# --------------------------------------------------------------------------- #


class TestNoHang:
    """A never-returning check must not block process exit."""

    def test_subprocess_no_hang_hung_plugin_exits(self, tmp_path):
        project = tmp_path / "proj"
        # A short suite deadline plus a plugin whose sync check never returns: the
        # runner off-loads it to a daemon thread, abandons it at the deadline, and
        # the CLI must still exit (via os._exit) with an error code.
        _write_config(
            project,
            "project_name: hung_test\n"
            "health:\n"
            "  suite_timeout_s: 2\n"
            "  plugins:\n"
            "    - hung_health_plugin\n",
        )

        fixtures_dir = Path(__file__).parent / "fixtures"
        osprey_bin = Path(sys.executable).parent / "osprey"
        assert osprey_bin.exists(), f"osprey console script not found at {osprey_bin}"

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(fixtures_dir), env.get("PYTHONPATH", "")]).rstrip(
            os.pathsep
        )

        # Wall-clock bound well above the 2s suite deadline but far below any hang:
        # a TimeoutExpired here is the no-hang failure signal.
        wall_clock_bound_s = 30.0
        started = time.monotonic()
        proc = subprocess.run(
            [str(osprey_bin), "health", "--project", str(project), "--category", "hung"],
            env=env,
            capture_output=True,
            text=True,
            timeout=wall_clock_bound_s,
        )
        elapsed = time.monotonic() - started

        assert elapsed < wall_clock_bound_s
        # Lower bound: the process must have actually reached the 2s suite deadline
        # (abandon-then-exit), not fast-failed before running the hung check.
        assert elapsed >= 1.5, f"exited too fast ({elapsed:.2f}s); deadline path may not have fired"
        # The abandoned category yields a deadline error row → exit 2.
        assert proc.returncode == 2, (
            f"rc={proc.returncode}\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
        )
        # Positively pin that the os._exit deadline path fired: the synthesized
        # budget-exceeded row for the hung category is rendered on stdout.
        assert "exceeded its" in proc.stdout and "budget" in proc.stdout, (
            f"missing deadline row\nstdout={proc.stdout!r}"
        )
