"""Tests for the osprey audit command."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import click
import pytest

from osprey.cli.audit_cmd import _detect_target_type, _extract_json, _list_files
from osprey.cli.audit_prompts import AuditFinding, AuditReport
from tests.cli._unrecorded import (
    a_host_local_record_is_up,
    act_as_a_recording_child,
    files_under,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_report() -> AuditReport:
    return AuditReport(
        summary="Test audit complete",
        overall_risk="low",
        findings=[
            AuditFinding(
                category="permissions",
                severity="warning",
                title="Open permission",
                explanation="A permission is too broad",
                file_path="settings.json",
                recommendation="Restrict the permission scope",
            ),
        ],
    )


@pytest.fixture
def sample_report_json(sample_report) -> str:
    return sample_report.model_dump_json()


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project directory."""
    (tmp_path / "config.yml").write_text("name: test\n")
    (tmp_path / ".claude" / "settings.json").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".claude" / "settings.json").rmdir()
    (tmp_path / ".claude").rmdir()
    subdir = tmp_path / "hooks"
    subdir.mkdir()
    (subdir / "pre_write.sh").write_text("#!/bin/bash\n")
    return tmp_path


#: What the runner's routing hands a launch on a project with telemetry on.
_ROUTED_ENV = {
    "ANTHROPIC_BASE_URL": "https://gateway.example",
    "ANTHROPIC_AUTH_TOKEN": "tok",
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector",
    "OSPREY_SPOOL_DIR": "/spool/alice",
}


@pytest.fixture
def routed_launch(monkeypatch) -> list[dict]:
    """Stand in for the runner's routing of the reviewer's launch.

    The fixture projects name no provider, so the real routing refuses them;
    the tests that fake ``stream_query`` need only the env it would route with.
    Returns the keyword arguments of each routing call.
    """
    from types import SimpleNamespace

    calls: list[dict] = []

    def _route(project_dir, **kwargs):
        calls.append({"project_dir": project_dir, **kwargs})
        return SimpleNamespace(env=dict(_ROUTED_ENV), model=kwargs.get("model"))

    monkeypatch.setattr("osprey.cli.audit_cmd.build_agent_options", _route, raising=True)
    return calls


@pytest.fixture
def stub_provider_check(monkeypatch):
    """Stand in for the reviewer's provider check.

    The command checks the audited project's provider in its own body, and
    these fixture projects name none, so the real check refuses them. The tests
    that use this fixture drive a faked agent loop, which needs no provider.
    """
    monkeypatch.setattr(
        "osprey.cli.audit_cmd._check_reviewer_provider",
        lambda project_dir: None,
        raising=True,
    )


@pytest.fixture
def tmp_profile(tmp_path):
    """Create a minimal profile YAML."""
    profile = tmp_path / "test-profile.yml"
    profile.write_text("name: test\nprovider: mock\nmodel: test\n")
    return profile


# ---------------------------------------------------------------------------
# Input detection tests
# ---------------------------------------------------------------------------


class TestDetectTargetType:
    def test_detect_yaml_profile(self, tmp_profile):
        assert _detect_target_type(str(tmp_profile)) == "profile"

    def test_detect_yaml_extension(self, tmp_path):
        f = tmp_path / "profile.yaml"
        f.write_text("name: test\n")
        assert _detect_target_type(str(f)) == "profile"

    def test_detect_directory(self, tmp_project):
        assert _detect_target_type(str(tmp_project)) == "project"

    def test_invalid_target(self, tmp_path):
        f = tmp_path / "readme.txt"
        f.write_text("hello")
        with pytest.raises(click.BadParameter, match="must be a .yml/.yaml"):
            _detect_target_type(str(f))


# ---------------------------------------------------------------------------
# File listing tests
# ---------------------------------------------------------------------------


class TestListFiles:
    def test_lists_files(self, tmp_project):
        listing = _list_files(tmp_project)
        assert "config.yml" in listing
        assert "hooks/pre_write.sh" in listing

    def test_max_files_limit(self, tmp_path):
        for i in range(10):
            (tmp_path / f"file{i}.txt").write_text("")
        listing = _list_files(tmp_path, max_files=3)
        assert "... and 7 more files" in listing


# ---------------------------------------------------------------------------
# JSON extraction tests
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_extract_raw_json(self, sample_report_json):
        text = f"Here is the report: {sample_report_json}"
        result = _extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["overall_risk"] == "low"

    def test_extract_markdown_fenced(self, sample_report_json):
        text = f"```json\n{sample_report_json}\n```"
        result = _extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["overall_risk"] == "low"

    def test_extract_no_fence_label(self, sample_report_json):
        text = f"```\n{sample_report_json}\n```"
        result = _extract_json(text)
        assert result is not None

    def test_invalid_no_json(self):
        assert _extract_json("No JSON here at all") is None

    def test_partial_json(self):
        # Truncated output — still extracts what's there
        result = _extract_json('{"summary": "test", "overall_risk": "low"}')
        assert result is not None


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------


class TestAuditModels:
    def test_finding_roundtrip(self):
        f = AuditFinding(
            category="safety",
            severity="error",
            title="Missing limits hook",
            explanation="channel_write lacks bounds checking",
            file_path="hooks/pre_write.sh",
            recommendation="Add limits hook",
        )
        data = json.loads(f.model_dump_json())
        f2 = AuditFinding.model_validate(data)
        assert f2.category == "safety"

    def test_report_roundtrip(self, sample_report):
        data = json.loads(sample_report.model_dump_json())
        r2 = AuditReport.model_validate(data)
        assert r2.overall_risk == "low"
        assert len(r2.findings) == 1

    def test_report_empty_findings(self):
        r = AuditReport(summary="Clean", overall_risk="low", findings=[])
        assert r.findings == []


# ---------------------------------------------------------------------------
# CLI invocation tests (agent run faked)
# ---------------------------------------------------------------------------


class TestAuditCLI:
    """Test CLI invocation with the agent run faked."""

    def _get_audit_cmd(self):
        from osprey.cli.audit_cmd import audit

        return audit

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    @pytest.mark.usefixtures("stub_provider_check")
    def test_audit_project_success(self, mock_asyncio, runner, tmp_project, sample_report):
        report_json = sample_report.model_dump_json()
        mock_asyncio.run.return_value = (report_json, 0.01, 5)

        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project)])
        assert result.exit_code == 0

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    @pytest.mark.usefixtures("stub_provider_check")
    def test_audit_json_output(self, mock_asyncio, runner, tmp_project, sample_report):
        report_json = sample_report.model_dump_json()
        mock_asyncio.run.return_value = (report_json, 0.01, 5)

        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project), "--json"])
        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["overall_risk"] == "low"

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    @pytest.mark.usefixtures("stub_provider_check")
    def test_audit_verbose(self, mock_asyncio, runner, tmp_project, sample_report):
        report_json = sample_report.model_dump_json()
        mock_asyncio.run.return_value = (report_json, 0.05, 10)

        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project), "-v"])
        assert result.exit_code == 0

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", False)
    def test_audit_missing_sdk(self, runner, tmp_project):
        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project)])
        assert result.exit_code == 1
        assert "claude-agent-sdk" in result.output

    def test_audit_nonexistent_target(self, runner):
        result = runner.invoke(self._get_audit_cmd(), ["/nonexistent/path"])
        assert result.exit_code == 2

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    @pytest.mark.usefixtures("stub_provider_check")
    def test_audit_invalid_json_output(self, mock_asyncio, runner, tmp_project):
        mock_asyncio.run.return_value = ("Not valid JSON at all", None, None)

        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project)])
        assert result.exit_code == 1

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    @pytest.mark.usefixtures("stub_provider_check")
    def test_audit_markdown_fenced_json(self, mock_asyncio, runner, tmp_project, sample_report):
        report_json = sample_report.model_dump_json()
        fenced = f"```json\n{report_json}\n```"
        mock_asyncio.run.return_value = (fenced, 0.01, 5)

        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project)])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Build flag tests
# ---------------------------------------------------------------------------


class TestBuildFlag:
    def _get_audit_cmd(self):
        from osprey.cli.audit_cmd import audit

        return audit

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    def test_build_flag_requires_profile(self, runner, tmp_project):
        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project), "--build"])
        assert result.exit_code == 1
        # Re-pinned for the renderer's failure shape: summary, cause, remedy.
        assert "--build needs a .yml or .yaml profile" in result.output

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    @patch("osprey.cli.audit_cmd.click.get_current_context")
    @pytest.mark.usefixtures("stub_provider_check")
    def test_build_flag_invokes_build_cmd(
        self, mock_ctx, mock_asyncio, runner, tmp_profile, sample_report
    ):
        report_json = sample_report.model_dump_json()
        mock_asyncio.run.return_value = (report_json, 0.01, 5)

        mock_context = MagicMock()
        mock_ctx.return_value = mock_context

        runner.invoke(self._get_audit_cmd(), [str(tmp_profile), "--build"])
        # The build invoke is called on the context
        assert mock_context.invoke.called
        call_kwargs = mock_context.invoke.call_args
        # project_name should start with "audit-"
        args = call_kwargs[1] if call_kwargs[1] else {}
        if "project_name" in args:
            assert args["project_name"].startswith("audit-")


class TestReviewerProvider:
    """The reviewer runs on the audited deployment's own provider."""

    def _get_audit_cmd(self):
        from osprey.cli.audit_cmd import audit

        return audit

    def test_the_reviewer_runs_through_the_runner_on_the_projects_provider(
        self, tmp_project, monkeypatch, routed_launch
    ):
        """The reviewer runs through the shared runner on the audited project's
        own provider: its launch env is the one the runner routes the project
        with, so nothing overrides the project's endpoint, auth and models —
        less only what would carry the run into a record.
        """
        from osprey.cli import audit_cmd

        captured: dict = {}

        async def fake_stream(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return
            yield  # pragma: no cover - makes this an async generator

        monkeypatch.setattr("osprey.cli.audit_cmd.stream_query", fake_stream, raising=True)

        asyncio.run(
            audit_cmd._run_audit(tmp_project, "p", model="some-model", budget=5.0, verbose=False)
        )

        [route] = routed_launch
        assert route["project_dir"] == tmp_project
        assert route["model"] == "some-model"
        assert route["setting_sources"] == []
        for key in ("env", "provider"):
            assert key not in route

        kwargs = captured["kwargs"]
        assert captured["args"][0] == tmp_project
        assert kwargs["model"] == "some-model"
        assert kwargs["max_budget_usd"] == 5.0
        assert kwargs["max_turns"] == 30
        assert kwargs["permission_mode"] == "bypassPermissions"
        assert kwargs["disallowed_tools"] == []
        # The reviewer reads the target; it must not run as it.
        assert kwargs["setting_sources"] == []
        assert kwargs["await_mcp_servers"] == ()
        for key in ("provider", "mcp_servers"):
            assert key not in kwargs
        env = kwargs["env"]
        assert env["ANTHROPIC_BASE_URL"] == "https://gateway.example"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "tok"
        assert env["OSPREY_AGENT_RECORD_OFF"] == "1"
        assert not [k for k in env if k.startswith("OTEL_")]
        assert "CLAUDE_CODE_ENABLE_TELEMETRY" not in env
        assert "OSPREY_SPOOL_DIR" not in env

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    def test_a_project_that_names_no_provider_is_refused(self, runner, tmp_project):
        """A project whose config.yml sets no provider has nothing to run on.

        The builder already says so, but it was called from inside the event
        loop, so the sentence reached the operator as a traceback.
        """
        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project)])

        flat = " ".join(result.output.split())
        assert result.exit_code == 1
        assert "Traceback" not in result.output
        assert "The reviewer has no provider to run on" in flat
        assert "set `provider:` in profile.yml and run `osprey build`" in flat

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    @pytest.mark.usefixtures("stub_provider_check")
    def test_default_model_is_the_projects_main_model(
        self, mock_asyncio, runner, tmp_project, sample_report, monkeypatch
    ):
        mock_asyncio.run.return_value = (sample_report.model_dump_json(), 0.01, 5)
        seen: dict = {}

        def fake_resolve(project_dir):
            seen["project_dir"] = project_dir
            return "gateway/claude-sonnet"

        monkeypatch.setattr(
            "osprey.agent_runner.primitives.resolve_default_model", fake_resolve, raising=True
        )

        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project)])

        assert result.exit_code == 0
        assert seen["project_dir"] == tmp_project
        assert "gateway/claude-sonnet" in result.output

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    @pytest.mark.usefixtures("stub_provider_check")
    def test_an_explicit_model_still_wins(
        self, mock_asyncio, runner, tmp_project, sample_report, monkeypatch
    ):
        mock_asyncio.run.return_value = (sample_report.model_dump_json(), 0.01, 5)
        monkeypatch.setattr(
            "osprey.agent_runner.primitives.resolve_default_model",
            lambda project_dir: "should-not-be-used",
            raising=True,
        )

        result = runner.invoke(self._get_audit_cmd(), [str(tmp_project), "--model", "picked"])

        assert result.exit_code == 0
        assert "picked" in result.output

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    def test_a_bare_profile_outside_a_repo_is_refused(self, runner, tmp_profile):
        """No project, no provider: better to say so than to review a
        deployment's safety config through whatever endpoint the shell holds."""
        result = runner.invoke(self._get_audit_cmd(), [str(tmp_profile)])

        assert result.exit_code == 1
        assert "--build" in result.output

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    @patch("osprey.cli.audit_cmd.asyncio")
    @pytest.mark.usefixtures("stub_provider_check")
    def test_a_bare_profile_in_a_repo_runs_from_the_repos_build(
        self, mock_asyncio, runner, tmp_path, sample_report, monkeypatch
    ):
        """The render under ``build/`` holds ``config.yml``; the repo root does
        not, so resolving from the root is a FileNotFoundError on any real repo."""
        from osprey.deployment.staleness import BUILD_DIRNAME

        (tmp_path / "profile.yml").write_text("name: test\n")
        (tmp_path / BUILD_DIRNAME).mkdir()
        (tmp_path / BUILD_DIRNAME / "config.yml").write_text("claude_code:\n  provider: mock\n")
        profile = tmp_path / "other-profile.yml"
        profile.write_text("name: test\nprovider: mock\n")
        mock_asyncio.run.return_value = (sample_report.model_dump_json(), 0.01, 5)
        seen: dict = {}

        def fake_resolve(project_dir):
            seen["project_dir"] = project_dir
            return "resolved-model"

        monkeypatch.setattr(
            "osprey.agent_runner.primitives.resolve_default_model", fake_resolve, raising=True
        )

        result = runner.invoke(self._get_audit_cmd(), [str(profile)])

        assert result.exit_code == 0
        assert seen["project_dir"] == tmp_path.resolve() / BUILD_DIRNAME

    @patch("osprey.cli.audit_cmd._SDK_AVAILABLE", True)
    def test_a_bare_profile_in_an_unbuilt_repo_is_refused(self, runner, tmp_path):
        """A repo that has never been built holds no config.yml to resolve a
        provider from — say so rather than crash reading one that is not there."""
        (tmp_path / "profile.yml").write_text("name: test\n")
        profile = tmp_path / "other-profile.yml"
        profile.write_text("name: test\nprovider: mock\n")

        result = runner.invoke(self._get_audit_cmd(), [str(profile)])

        assert result.exit_code == 1
        assert "--build" in result.output


# ---------------------------------------------------------------------------
# Reviewer run tests (through the shared runner)
# ---------------------------------------------------------------------------


def _fake_stream(*events, log: list[str] | None = None, between: str | None = None):
    """Return a stand-in for ``stream_query`` that yields *events* in order.

    When *log* and *between* are given, *between* is appended to *log* after
    the first event is pulled and before the second is yielded.
    """

    async def _stream(*args, **kwargs):
        for index, event in enumerate(events):
            if index == 1 and log is not None and between is not None:
                log.append(between)
            yield event

    return _stream


def _result_event(*, total_cost_usd: float = 0.01, num_turns: int = 1):
    from osprey.agent_runner import ResultEvent

    return ResultEvent(
        subtype="success",
        is_error=False,
        num_turns=num_turns,
        duration_ms=1,
        session_id="stub",
        total_cost_usd=total_cost_usd,
        usage=None,
        result=None,
        api_error_status=None,
    )


class TestReviewerRun:
    """``_run_audit`` drives the shared runner and collects the reviewer's text."""

    def test_the_reviewer_loads_no_project_settings_or_mcp_servers(self, tmp_project, monkeypatch):
        """The reviewer reads the audited project; it must not run as it.

        The project declares an MCP server and a hook, and neither reaches the
        reviewing agent, nor does the run wait for a server it never loads.
        """
        pytest.importorskip("claude_agent_sdk")
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

        from osprey.cli import audit_cmd

        (tmp_project / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"controls": {"command": "true"}}})
        )
        (tmp_project / ".claude").mkdir()
        (tmp_project / ".claude" / "settings.json").write_text(
            json.dumps(
                {
                    "hooks": {
                        "PreToolUse": [
                            {
                                "matcher": "*",
                                "hooks": [{"type": "command", "command": "true"}],
                            }
                        ]
                    }
                }
            )
        )

        async def _response():
            yield AssistantMessage(content=[TextBlock(text="report")], model="m")
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="s",
                total_cost_usd=0.01,
            )

        client = MagicMock()
        client.query = AsyncMock(return_value=None)
        client.receive_response = MagicMock(return_value=_response())
        async_cm = MagicMock()
        async_cm.__aenter__ = AsyncMock(return_value=client)
        async_cm.__aexit__ = AsyncMock(return_value=False)
        captured: dict = {}

        def _client(*, options):
            captured["options"] = options
            return async_cm

        await_ready = AsyncMock(return_value=[])
        expected = Mock(return_value={"controls"})
        monkeypatch.setattr(
            "osprey.agent_runner.primitives.sdk_env", lambda *a, **k: {"CLAUDECODE": ""}
        )
        monkeypatch.setattr(
            "osprey.agent_runner.primitives._resolve_project_spec", lambda *a, **k: None
        )
        monkeypatch.setattr("osprey.agent_runner.primitives.await_mcp_ready", await_ready)
        monkeypatch.setattr("osprey.agent_runner.primitives.expected_mcp_servers", expected)
        monkeypatch.setattr("osprey.agent_runner.runner.ClaudeSDKClient", _client)

        result = asyncio.run(
            audit_cmd._run_audit(tmp_project, "p", model="m", budget=5.0, verbose=False)
        )

        options = captured["options"]
        assert options.setting_sources == []
        assert options.mcp_servers == {}
        assert options.hooks is None
        assert options.max_turns == 30
        assert options.max_budget_usd == 5.0
        assert options.permission_mode == "bypassPermissions"
        assert options.model == "m"
        assert options.cwd == str(tmp_project)
        assert await_ready.await_count == 0
        assert expected.called is False
        assert result == ("report", 0.01, 1)

    def test_a_project_with_telemetry_on_launches_a_reviewer_that_exports_nothing(
        self, tmp_path, monkeypatch
    ):
        """The project's telemetry block does not ride the SDK overlay into the child.

        The real options are built for a project whose provider env carries
        Claude Code's OTLP exporter block; only that resolution and the SDK
        client are replaced. The env the
        reviewer's process receives carries no OTLP exporter, no spool and the
        record off switch.
        """
        pytest.importorskip("claude_agent_sdk")
        from claude_agent_sdk import ResultMessage

        from osprey.cli import audit_cmd

        (tmp_path / "config.yml").write_text(
            "claude_code:\n  provider: anthropic\n", encoding="utf-8"
        )
        monkeypatch.setenv("OSPREY_SPOOL_DIR", str(tmp_path / "spool"))
        # The provider env a project with telemetry on resolves to: routing
        # plus the OTLP exporter block.
        monkeypatch.setattr(
            "osprey.agent_runner.primitives.provider_env_for_project",
            lambda *_a, **_k: {
                "ANTHROPIC_MODEL": "claude-main",
                "ANTHROPIC_BASE_URL": "https://gateway.example",
                "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
                "OTEL_LOGS_EXPORTER": "otlp",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector",
            },
        )
        monkeypatch.setattr(
            "osprey.agent_runner.primitives._resolve_project_spec", lambda *a, **k: None
        )

        async def _response():
            yield ResultMessage(
                subtype="success",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="s",
                total_cost_usd=0.01,
            )

        client = MagicMock()
        client.query = AsyncMock(return_value=None)
        client.receive_response = MagicMock(return_value=_response())
        async_cm = MagicMock()
        async_cm.__aenter__ = AsyncMock(return_value=client)
        async_cm.__aexit__ = AsyncMock(return_value=False)
        captured: dict = {}

        def _client(*, options):
            captured["options"] = options
            captured["config_dir"] = Path(options.env["CLAUDE_CONFIG_DIR"])
            captured["config_dir_existed"] = captured["config_dir"].is_dir()
            return async_cm

        monkeypatch.setattr("osprey.agent_runner.runner.ClaudeSDKClient", _client)

        asyncio.run(audit_cmd._run_audit(tmp_path, "p", model="m", budget=5.0, verbose=False))

        env = captured["options"].env
        assert [k for k in env if k.startswith("OTEL_")] == []
        assert "CLAUDE_CODE_ENABLE_TELEMETRY" not in env
        assert "OSPREY_SPOOL_DIR" not in env
        assert env["OSPREY_AGENT_RECORD_OFF"] == "1"
        # The run's own Claude Code config dir, gone once the review is over.
        assert captured["config_dir_existed"]
        assert not captured["config_dir"].exists()
        # Still routed on the project's provider.
        assert env["ANTHROPIC_BASE_URL"] == "https://gateway.example"

    @pytest.mark.usefixtures("routed_launch")
    def test_verbose_echoes_each_text_block_as_it_arrives(self, tmp_project, monkeypatch):
        from osprey.agent_runner import TextEvent
        from osprey.cli import audit_cmd

        log: list[str] = []
        monkeypatch.setattr(
            "osprey.cli.audit_cmd.stream_query",
            _fake_stream(
                TextEvent(text="a" * 250, parent_tool_use_id=None),
                TextEvent(text="b", parent_tool_use_id=None),
                _result_event(),
                log=log,
                between="pulled second",
            ),
        )
        monkeypatch.setattr(
            "osprey.cli.audit_cmd.output.note", lambda msg: log.append("note:" + msg)
        )

        asyncio.run(audit_cmd._run_audit(tmp_project, "p", model="m", budget=5.0, verbose=True))

        assert log == ["note:" + "a" * 200 + "...", "pulled second", "note:b..."]

    @pytest.mark.usefixtures("routed_launch")
    def test_a_quiet_run_echoes_nothing_and_returns_text_cost_and_turns(
        self, tmp_project, monkeypatch
    ):
        """Only text is collected, a subagent's text included."""
        from osprey.agent_runner import SystemEvent, TextEvent, ThinkingEvent, ToolUseEvent
        from osprey.cli import audit_cmd

        monkeypatch.setattr(
            "osprey.cli.audit_cmd.stream_query",
            _fake_stream(
                ThinkingEvent(text="hmm"),
                ToolUseEvent(tool_use_id="toolu_1", name="Read", input={}, parent_tool_use_id=None),
                TextEvent(text="one", parent_tool_use_id=None),
                SystemEvent(subtype="init", data={}),
                TextEvent(text="two", parent_tool_use_id="toolu_1"),
                _result_event(total_cost_usd=0.02, num_turns=3),
            ),
        )
        note = Mock()
        monkeypatch.setattr("osprey.cli.audit_cmd.output.note", note)

        result = asyncio.run(
            audit_cmd._run_audit(tmp_project, "p", model="m", budget=5.0, verbose=False)
        )

        note.assert_not_called()
        assert result == ("onetwo", 0.02, 3)

    @pytest.mark.usefixtures("routed_launch")
    def test_a_run_without_a_result_leaves_cost_and_turns_unknown(self, tmp_project, monkeypatch):
        from osprey.agent_runner import TextEvent
        from osprey.cli import audit_cmd

        monkeypatch.setattr(
            "osprey.cli.audit_cmd.stream_query",
            _fake_stream(TextEvent(text="x", parent_tool_use_id=None)),
        )

        result = asyncio.run(
            audit_cmd._run_audit(tmp_project, "p", model="m", budget=5.0, verbose=False)
        )

        assert result == ("x", None, None)

    def test_the_verb_refuses_cleanly_without_the_agent_sdk(self, tmp_path):
        """The module loads without the SDK and the verb says what is missing."""
        script = textwrap.dedent(
            """
            import sys

            sys.modules["claude_agent_sdk"] = None

            from click.testing import CliRunner

            from osprey.cli import audit_cmd

            assert audit_cmd._SDK_AVAILABLE is False
            result = CliRunner().invoke(audit_cmd.audit, [sys.argv[1]])
            assert result.exit_code == 1, result.output
            assert "claude-agent-sdk is not installed" in result.stderr, result.stderr
            """
        )

        proc = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# Display tests
# ---------------------------------------------------------------------------


class TestDisplay:
    @pytest.mark.usefixtures("sample_report")
    def test_display_error_finding(self):
        from osprey.cli.audit_cmd import _display_report

        report = AuditReport(
            summary="Issues found",
            overall_risk="high",
            findings=[
                AuditFinding(
                    category="safety",
                    severity="error",
                    title="Critical issue",
                    explanation="Details",
                    file_path="hooks.sh",
                    recommendation="Fix it",
                ),
            ],
        )
        # Just verify it doesn't crash — Rich output goes to console, not capsys
        _display_report(report, json_output=False, verbose=False)

    def test_display_json_output(self, sample_report, capsys):
        from osprey.cli.audit_cmd import _display_report

        _display_report(sample_report, json_output=True, verbose=False)
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["overall_risk"] == "low"

    def test_display_empty_findings(self):
        from osprey.cli.audit_cmd import _display_report

        report = AuditReport(summary="Clean", overall_risk="low", findings=[])
        _display_report(report, json_output=False, verbose=False)

    def test_display_verbose_with_cost(self, sample_report):
        from osprey.cli.audit_cmd import _display_report

        _display_report(
            sample_report,
            json_output=False,
            verbose=True,
            cost=0.05,
            turns=10,
        )


# ---------------------------------------------------------------------------
# A run that leaves nothing in the agent record
# ---------------------------------------------------------------------------


class TestUnrecordedRun:
    """``unrecorded_run`` holds a run whose children record nothing."""

    def test_sets_the_off_switch_and_drops_spool_and_telemetry_env(self):
        from osprey.cli.audit_cmd import unrecorded_run

        env = {
            "OSPREY_SPOOL_DIR": "/spool/alice",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector",
            "OTEL_LOGS_EXPORTER": "otlp",
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "CLAUDE_CONFIG_DIR": "/home/alice/.claude",
            "PATH": "/bin",
        }

        with unrecorded_run(env) as config_dir:
            assert env == {
                "OSPREY_AGENT_RECORD_OFF": "1",
                "CLAUDE_CONFIG_DIR": str(config_dir),
                "PATH": "/bin",
            }
            assert config_dir.is_dir()
            assert config_dir != Path("/home/alice/.claude")

    def test_puts_every_value_back_and_removes_its_config_dir(self):
        from osprey.cli.audit_cmd import unrecorded_run

        env = {
            "OSPREY_SPOOL_DIR": "/spool/alice",
            "OTEL_LOGS_EXPORTER": "otlp",
            "CLAUDE_CONFIG_DIR": "/home/alice/.claude",
        }
        before = dict(env)

        with unrecorded_run(env) as config_dir:
            pass

        assert env == before
        assert not config_dir.exists()

    def test_unsets_what_was_unset_before(self):
        from osprey.cli.audit_cmd import unrecorded_run

        env: dict[str, str] = {}

        with unrecorded_run(env):
            assert env["OSPREY_AGENT_RECORD_OFF"] == "1"

        assert env == {}

    def test_restores_when_the_run_raises(self):
        from osprey.cli.audit_cmd import unrecorded_run

        env = {"OSPREY_SPOOL_DIR": "/spool/alice", "OSPREY_AGENT_RECORD_OFF": "0"}

        with pytest.raises(RuntimeError), unrecorded_run(env) as config_dir:
            raise RuntimeError("the run failed")

        assert env == {"OSPREY_SPOOL_DIR": "/spool/alice", "OSPREY_AGENT_RECORD_OFF": "0"}
        assert not config_dir.exists()

    def test_each_run_gets_a_fresh_config_dir(self):
        from osprey.cli.audit_cmd import unrecorded_run

        with unrecorded_run({}) as first:
            pass
        with unrecorded_run({}) as second:
            pass

        assert first != second


class TestTheAuditLeavesNothingInTheRecord:
    """The reviewer's run reaches no spool, transcript root or OTLP collector."""

    @pytest.mark.usefixtures("routed_launch")
    def test_an_audit_with_a_host_local_record_up_leaves_nothing(
        self, tmp_project, tmp_path, monkeypatch, spool_zone
    ):
        import os

        from osprey.cli import audit_cmd

        transcript_root = a_host_local_record_is_up(monkeypatch, tmp_path)
        seen: list[dict] = []

        async def child_run(*_args, **_kwargs):
            seen.append(act_as_a_recording_child(spool_zone.root))
            yield _result_event()

        monkeypatch.setattr("osprey.cli.audit_cmd.stream_query", child_run)

        asyncio.run(audit_cmd._run_audit(tmp_project, "p", model="m", budget=5.0, verbose=False))

        [child] = seen
        assert child["entry_id"] is None
        assert child["otel"] == []
        assert child["spool_dir"] is None
        assert Path(child["config_dir"]) != transcript_root
        assert files_under(spool_zone.root) == []
        assert files_under(transcript_root) == []
        assert spool_zone.recorded_entries() == []
        # The launch's own environment is back once the review is over.
        assert os.environ["CLAUDE_CONFIG_DIR"] == str(transcript_root)
        assert os.environ["OTEL_LOGS_EXPORTER"] == "otlp"
        assert "OSPREY_AGENT_RECORD_OFF" not in os.environ
