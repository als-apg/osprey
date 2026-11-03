# ruff: noqa: F811 - the launcher tests' fixtures are imported, then requested by name
"""Host launches export Claude Code's OTLP logs to the record they hold, under the host login.

``osprey web``, ``osprey chat`` and ``osprey query`` on the host each hold a
host-local record (``test_host_launch_record.py``). Here: what they hand the
agent while they hold it is the OTLP block for that record — its ``/v1/logs``,
the host's own login — and the block is taken back out when the lease ends.
The container launch paths (web terminal, dispatch worker) are covered in
``test_telemetry_env.py`` and ``tests/dispatch_worker/test_dispatch_api.py``.
"""

from __future__ import annotations

import base64
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.agent_runner import provider_env
from osprey.build.claude_code_telemetry import TELEMETRY_ENV_VARS
from osprey.cli.chat_cmd import chat
from osprey.cli.query_cmd import query
from osprey.cli.web_cmd import web
from tests.cli._scoped_subprocess import patch_subprocess
from tests.cli.test_host_launch_record import (  # noqa: F401 - fixtures requested by name
    _config,
    _deploy,
    _launch_stubs,
    _record_port,
    _restore_process_state,
    fake_ensure,
    real_record,
    runner,
)


def _otlp(environ) -> dict[str, str]:
    return {key: environ[key] for key in TELEMETRY_ENV_VARS if key in environ}


def _assert_exports_to_the_record(otlp: dict[str, str], record) -> None:
    assert otlp["OTEL_LOGS_EXPORTER"] == "otlp"
    assert otlp["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] == f"{record.url}/v1/logs"
    header = otlp["OTEL_EXPORTER_OTLP_HEADERS"].removeprefix("Authorization=Basic ")
    assert base64.b64decode(header).decode() == f"{record.login}:{record.token}"
    assert otlp["OTEL_METRICS_EXPORTER"] == otlp["OTEL_TRACES_EXPORTER"] == "none"


def test_host_record_launch_publishes_the_block_and_takes_it_back(tmp_path, fake_ensure):
    environ: dict[str, str] = {"OSPREY_TERMINAL_SESSION": "chat-7"}
    config = _config(tmp_path)
    with provider_env.host_record_launch(tmp_path, config, environ=environ):
        inside = _otlp(environ)
    _assert_exports_to_the_record(inside, fake_ensure.record)
    assert inside["OTEL_RESOURCE_ATTRIBUTES"] == "terminal.session=chat-7"
    assert _otlp(environ) == {}


@pytest.mark.usefixtures("fake_ensure")
def test_host_record_launch_publishes_no_block_under_the_off_switch(tmp_path):
    environ: dict[str, str] = {"OSPREY_AGENT_RECORD_OFF": "1"}
    with provider_env.host_record_launch(tmp_path, _config(tmp_path), environ=environ):
        assert _otlp(environ) == {}


def test_web_hands_its_sessions_the_record_block(runner, lifecycle_repo, fake_ensure, monkeypatch):
    repo = _deploy(lifecycle_repo, _config(lifecycle_repo), credentials=False)
    seen: dict = {}

    def _run_web(**_kw):
        seen["otlp"] = _otlp(os.environ)

    monkeypatch.setattr("osprey.interfaces.web_terminal.run_web", _run_web)
    monkeypatch.chdir(repo)

    result = runner.invoke(web, ["--shell", "true", "--skip-preflight"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    _assert_exports_to_the_record(seen["otlp"], fake_ensure.record)


def test_chat_hands_the_agent_the_record_block(runner, lifecycle_repo, fake_ensure):
    repo = _deploy(lifecycle_repo, _config(lifecycle_repo), credentials=False)
    launched: list[dict] = []

    def fake_run(_argv, *_args, env=None, **_kwargs):
        launched.append(dict(env or {}))
        return SimpleNamespace(returncode=0)

    with patch_subprocess("osprey.cli.chat_cmd", side_effect=fake_run):
        result = runner.invoke(chat, ["--repo", str(repo)])

    assert result.exit_code == 0, result.output
    otlp = _otlp(launched[0])
    _assert_exports_to_the_record(otlp, fake_ensure.record)
    session = launched[0]["OSPREY_TERMINAL_SESSION"]
    assert otlp["OTEL_RESOURCE_ATTRIBUTES"] == f"terminal.session={session}"


def test_query_hands_the_agent_the_record_block(runner, lifecycle_repo, fake_ensure):
    repo = _deploy(lifecycle_repo, _config(lifecycle_repo), credentials=False)
    seen: dict = {}

    async def _run_query(*_args, **_kw):
        seen["otlp"] = _otlp(os.environ)
        return MagicMock(text_blocks=["ok"], is_error=False, mcp_servers=[], tool_traces=[])

    with (
        patch("osprey.cli.query_cmd.run_query", new=AsyncMock(side_effect=_run_query)),
        patch("osprey.cli.query_cmd.read_only_disallowed_tools", return_value=[]),
        patch("osprey.cli.query_cmd.expected_mcp_servers", return_value=set()),
        patch("osprey.cli.query_cmd.evaluate_verdict", return_value=0),
    ):
        result = runner.invoke(query, ["--repo", str(repo), "hello"])

    assert result.exit_code == 0, result.output
    _assert_exports_to_the_record(seen["otlp"], fake_ensure.record)
