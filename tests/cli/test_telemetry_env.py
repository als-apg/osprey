"""The OTLP block Claude Code is launched with: logs to the agent record, nothing else.

Covers :func:`~osprey.build.claude_code_telemetry.record_telemetry_env` (the
one destination, the sender's own login, the export interval, the gates, the
off switch and the "no record to send to" case), the literals it restates,
its integration into ``resolve()`` and ``load_provider_spec()``, the
conflict-detection exemption for telemetry vars, and the guarantees that no
profile value selects another exporter and no other code builds an OTLP
endpoint. The host launchers are covered in ``test_host_launch_otlp.py``.
"""

from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

import pytest
import yaml

import osprey
from osprey.agent_runner.provider_env import (
    MANAGED_ENV_VARS,
    ClaudeCodeModelResolver,
    ClaudeCodeModelSpec,
    load_provider_spec,
)
from osprey.build import claude_code_telemetry as telemetry
from osprey.build.claude_code_telemetry import (
    CONTENT_GATES,
    TELEMETRY_ENV_VARS,
    record_telemetry_env,
)

URL = "http://127.0.0.1:8790"
LOGIN = "alice:t0ken"
_LAUNCH = {telemetry.RECORD_URL_ENV: URL, telemetry.RECORD_LOGIN_ENV: LOGIN}

#: What a launch with no record to send to exports.
_EXPORT_NOTHING = {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "OTEL_LOGS_EXPORTER": "none",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_TRACES_EXPORTER": "none",
}


def _basic(login: str) -> str:
    return f"Authorization=Basic {base64.b64encode(login.encode()).decode()}"


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


def test_a_launch_with_a_record_exports_only_logs_to_its_v1_logs():
    env = record_telemetry_env({**_LAUNCH, telemetry.TERMINAL_SESSION_ENV: "chat-1"})
    assert env == {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "0",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "none",
        "OTEL_TRACES_EXPORTER": "none",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        "OTEL_EXPORTER_OTLP_ENDPOINT": URL,
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": f"{URL}/v1/logs",
        "OTEL_EXPORTER_OTLP_HEADERS": _basic(LOGIN),
        "OTEL_LOGS_EXPORT_INTERVAL": "1000",
        "OTEL_RESOURCE_ATTRIBUTES": "terminal.session=chat-1",
        "OTEL_LOG_USER_PROMPTS": "0",
        "OTEL_LOG_ASSISTANT_RESPONSES": "0",
        "OTEL_LOG_TOOL_DETAILS": "0",
        "OTEL_LOG_RAW_API_BODIES": "0",
        "OTEL_LOG_TOOL_CONTENT": "0",
    }


def test_the_login_header_is_the_senders_own_login():
    for login in ("alice:t0ken", "dispatch-worker-1:x", "host-builder:y"):
        env = record_telemetry_env({**_LAUNCH, telemetry.RECORD_LOGIN_ENV: login})
        header = env["OTEL_EXPORTER_OTLP_HEADERS"]
        assert header.startswith("Authorization=Basic ")
        assert base64.b64decode(header.removeprefix("Authorization=Basic ")).decode() == login


def test_a_trailing_slash_on_the_address_is_not_doubled():
    env = record_telemetry_env({**_LAUNCH, telemetry.RECORD_URL_ENV: f"{URL}/"})
    assert env["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] == f"{URL}/v1/logs"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == URL


def test_the_terminal_session_attribute_is_empty_until_known():
    env = record_telemetry_env(_LAUNCH)
    assert env["OTEL_RESOURCE_ATTRIBUTES"] == "terminal.session="


def test_every_content_gate_is_written_off():
    env = record_telemetry_env(_LAUNCH)
    assert {gate: env[gate] for gate in CONTENT_GATES} == dict.fromkeys(CONTENT_GATES, "0")


def test_the_off_switch_leaves_no_otlp_setting_at_all():
    assert record_telemetry_env({**_LAUNCH, telemetry.RECORD_OFF_ENV: "1"}) == {}


@pytest.mark.parametrize(
    "launch",
    [
        {},
        {telemetry.RECORD_URL_ENV: URL},
        {telemetry.RECORD_LOGIN_ENV: LOGIN},
        {telemetry.RECORD_URL_ENV: " ", telemetry.RECORD_LOGIN_ENV: LOGIN},
    ],
    ids=["nothing", "no-login", "no-url", "blank-url"],
)
def test_a_launch_with_no_record_exports_nothing(launch):
    assert record_telemetry_env(launch) == _EXPORT_NOTHING


@pytest.mark.parametrize("login", ["no-colon", ":tok", "alice:", "al,ice:tok", "alice:to\nk"])
def test_an_unusable_login_exports_nothing_and_never_echoes_it(login, caplog):
    with caplog.at_level(logging.WARNING, logger=telemetry.__name__):
        env = record_telemetry_env({**_LAUNCH, telemetry.RECORD_LOGIN_ENV: login})
    assert env == _EXPORT_NOTHING
    assert caplog.records, "an unusable login is reported"
    assert all(login not in record.getMessage() for record in caplog.records)


def test_every_written_key_is_a_telemetry_var_with_a_string_value():
    for launch in (_LAUNCH, {}, {**_LAUNCH, telemetry.TERMINAL_SESSION_ENV: "s"}):
        env = record_telemetry_env(launch)
        assert set(env) <= TELEMETRY_ENV_VARS
        assert all(isinstance(value, str) for value in env.values())


def test_telemetry_vars_are_not_managed_vars():
    assert not TELEMETRY_ENV_VARS & MANAGED_ENV_VARS


# ---------------------------------------------------------------------------
# The literals the module restates
# ---------------------------------------------------------------------------


def test_restated_literals_match_their_owners():
    from osprey.agent_record.client import URL_ENV
    from osprey.agent_record.contract import RECORD_OFF_ENV, RECORD_OFF_VALUE
    from osprey.agent_record.credentials import LOGIN_ENV
    from osprey.agent_record.ingest.otlp import LOGS_PATH, TERMINAL_SESSION_ATTRIBUTE
    from osprey.audit.envelope import TERMINAL_SESSION_ENV

    assert telemetry.RECORD_URL_ENV == URL_ENV
    assert telemetry.RECORD_LOGIN_ENV == LOGIN_ENV
    assert (telemetry.RECORD_OFF_ENV, telemetry.RECORD_OFF_VALUE) == (
        RECORD_OFF_ENV,
        RECORD_OFF_VALUE,
    )
    assert telemetry.LOGS_PATH == LOGS_PATH
    assert telemetry.TERMINAL_SESSION_ATTRIBUTE == TERMINAL_SESSION_ATTRIBUTE
    assert telemetry.TERMINAL_SESSION_ENV == TERMINAL_SESSION_ENV


# ---------------------------------------------------------------------------
# No profile value selects another exporter
# ---------------------------------------------------------------------------

#: Every shape a profile's telemetry block could take to send elsewhere, turn
#: content on, or turn the export off.
_PROFILE_BLOCKS = [
    {"enabled": False},
    {"enabled": True, "backend": "gateway", "protocol": "http/json"},
    {
        "enabled": True,
        "backend": "generic",
        "endpoint": "http://collector.example:4318",
        "protocol": "grpc",
        "auth": {"token_env": "COLLECTOR_TOKEN"},
        "signals": ["metrics", "traces"],
        "resource_attributes": {"site": "x", "terminal.session": "forged"},
        "log_user_prompts": True,
        "log_assistant_responses": True,
        "log_tool_details": True,
        "log_raw_api_bodies": True,
        "log_tool_content": True,
        "content_max_length": 99,
    },
]


@pytest.mark.parametrize("block", _PROFILE_BLOCKS, ids=["off", "gateway", "generic"])
def test_no_profile_value_changes_the_block(block):
    environ = {**_LAUNCH, "COLLECTOR_TOKEN": "secret", "ANTHROPIC_API_KEY": "k"}
    spec = ClaudeCodeModelResolver.resolve(
        {"provider": "anthropic", "telemetry": block}, environ=environ
    )
    assert spec is not None
    shipped = {k: v for k, v in spec.env_block.items() if k in TELEMETRY_ENV_VARS}
    assert shipped == record_telemetry_env(environ)
    assert "collector.example" not in " ".join(spec.env_block.values())
    assert "secret" not in " ".join(spec.env_block.values())


# ---------------------------------------------------------------------------
# Resolving the provider and loading its spec
# ---------------------------------------------------------------------------


def test_a_build_time_resolution_exports_nothing():
    spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
    assert spec is not None
    shipped = {k: v for k, v in spec.env_block.items() if k in TELEMETRY_ENV_VARS}
    assert shipped == _EXPORT_NOTHING


def test_without_telemetry_the_env_block_carries_no_telemetry_var():
    spec = ClaudeCodeModelResolver.resolve(
        {"provider": "anthropic"}, include_telemetry=False, environ=_LAUNCH
    )
    assert spec is not None
    assert not set(spec.env_block) & TELEMETRY_ENV_VARS


def test_a_container_launch_exports_under_the_login_its_compose_file_gave_it(tmp_path, monkeypatch):
    """The web terminal's and the worker's path: the env holds the record and the login."""
    (tmp_path / "config.yml").write_text(yaml.safe_dump({"claude_code": {"provider": "anthropic"}}))
    monkeypatch.setenv(telemetry.RECORD_URL_ENV, "http://127.0.0.1:9100")
    monkeypatch.setenv(telemetry.RECORD_LOGIN_ENV, "bob:pw")
    monkeypatch.delenv(telemetry.RECORD_OFF_ENV, raising=False)

    spec = load_provider_spec(tmp_path)

    assert spec is not None
    assert spec.env_block["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"] == "http://127.0.0.1:9100/v1/logs"
    assert spec.env_block["OTEL_EXPORTER_OTLP_HEADERS"] == _basic("bob:pw")


def test_no_conflict_on_preexisting_otel():
    """A pre-existing OTEL export is an override, never a launch-blocking conflict."""
    spec = ClaudeCodeModelSpec(
        provider="anthropic",
        default_model_id="m",
        env_block={"ANTHROPIC_MODEL": "m", **record_telemetry_env(_LAUNCH)},
    )
    environ = {"ANTHROPIC_MODEL": "m", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://elsewhere"}
    assert spec.detect_env_conflicts(environ) == {}


# ---------------------------------------------------------------------------
# No code path builds another OTLP endpoint
# ---------------------------------------------------------------------------

#: An OTLP exporter setting, in any spelling a Python source would write it.
_EXPORTER_SETTING = re.compile(r"OTEL_EXPORTER_OTLP_\w*(ENDPOINT|HEADERS)|OTEL_\w*_EXPORTER\b")

#: Sources that name an exporter setting without building one: a secret-name
#: list that redacts the headers in status output, and a locator that asks
#: whether the session exports at all.
_READS_ONLY = {
    "deployment/status_display.py",
    "mcp_server/workspace/tools/provenance_locator.py",
}


def test_only_the_record_block_builds_an_otlp_exporter_setting():
    root = Path(osprey.__file__).parent
    writers = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if _EXPORTER_SETTING.search(path.read_text(encoding="utf-8"))
    }
    assert writers - _READS_ONLY == {"build/claude_code_telemetry.py"}


# ---------------------------------------------------------------------------
# Tooling runs keep no OTLP env
# ---------------------------------------------------------------------------


def test_an_unrecorded_run_resolves_and_carries_no_otlp_setting():
    """``osprey audit`` and the channel-finder benchmark run unrecorded: no OTLP at all."""
    from osprey.cli.audit_cmd import unrecorded_run

    environ = {
        **_LAUNCH,
        **record_telemetry_env(_LAUNCH),
        "ANTHROPIC_API_KEY": "k",
    }
    with unrecorded_run(environ):
        spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"}, environ=environ)
        assert spec is not None
        assert not set(spec.env_block) & TELEMETRY_ENV_VARS
        assert not [key for key in environ if key.startswith("OTEL_")]
