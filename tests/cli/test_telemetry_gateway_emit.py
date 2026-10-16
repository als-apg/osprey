"""The agent's emit path to the deployment's telemetry gateway.

``backend: gateway`` posts to the gateway address the launch resolved, with the
sender's own login as HTTP Basic, and carries the launch's terminal session as
the ``terminal.session`` resource attribute — the same value the ledger writers
file the launch's records under. A collector named by ``endpoint`` (no backend)
is unchanged byte for byte.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import warnings

import pytest

from osprey.build import claude_code_telemetry as cct
from osprey.build.claude_code_telemetry import (
    ObservabilityCredentialError,
    TelemetryConfigError,
    _build_telemetry_env,
    export_host_terminal_session,
    terminal_session_env,
)

GATEWAY = "http://127.0.0.1:10051"
LOGIN = "alice:s3cret"
_GATEWAY_CFG = {"enabled": True, "backend": "gateway"}
_GENERIC_CFG = {
    "enabled": True,
    "endpoint": "https://collector.example.org:4318",
    "auth": {"token_env": "OPERATOR_OTLP_SECRET"},
    "resource_attributes": {"service.name": "osprey"},
    "log_user_prompts": False,
    "log_assistant_responses": False,
    "log_tool_details": False,
    "log_tool_content": False,
    "log_raw_api_bodies": False,
}


def _gateway_env(**kwargs) -> dict[str, str]:
    kwargs.setdefault("gateway_url", GATEWAY)
    kwargs.setdefault("telemetry_login", LOGIN)
    return _build_telemetry_env(dict(_GATEWAY_CFG, **kwargs.pop("cfg", {})), **kwargs)


def test_gateway_posts_to_the_resolved_address_with_the_senders_login():
    env = _gateway_env(terminal_session="chat-1")
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == GATEWAY
    token = base64.b64encode(LOGIN.encode()).decode()
    assert env["OTEL_EXPORTER_OTLP_HEADERS"] == f"Authorization=Basic {token}"
    assert env["OTEL_RESOURCE_ATTRIBUTES"] == "terminal.session=chat-1"


def test_gateway_keeps_the_blocks_own_resource_attributes_first():
    env = _gateway_env(cfg={"resource_attributes": {"service.name": "osprey"}})
    assert env["OTEL_RESOURCE_ATTRIBUTES"] == "service.name=osprey,terminal.session="


def test_gateway_refuses_a_configured_terminal_session_attribute():
    with pytest.raises(TelemetryConfigError, match="terminal.session"):
        _gateway_env(cfg={"resource_attributes": "terminal.session=pinned"})


@pytest.mark.parametrize("protocol", ["http/protobuf", "http/json"])
def test_gateway_accepts_the_http_protocols(protocol):
    env = _gateway_env(cfg={"protocol": protocol})
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == protocol


def test_grpc_under_the_gateway_is_refused():
    with pytest.raises(TelemetryConfigError, match="grpc"):
        _gateway_env(cfg={"protocol": "grpc"})


def test_an_endpoint_beside_the_gateway_is_refused():
    with pytest.raises(TelemetryConfigError, match="endpoint"):
        _gateway_env(cfg={"endpoint": "http://elsewhere:4318"})


def test_bearer_auth_beside_the_gateway_is_refused():
    with pytest.raises(TelemetryConfigError, match="auth"):
        _gateway_env(cfg={"auth": {"token_env": "X"}})


@pytest.mark.parametrize("login", [None, "", "   "])
def test_a_missing_login_fails_loud(login):
    with pytest.raises(ObservabilityCredentialError, match=cct.TELEMETRY_LOGIN_ENV) as caught:
        _gateway_env(telemetry_login=login)
    assert caught.value.unresolved_vars == (cct.TELEMETRY_LOGIN_ENV,)


@pytest.mark.parametrize("login", ["no-colon", ":pw", "alice:", "alice:a,b", "alice:a\nb"])
def test_a_malformed_login_fails_loud(login):
    with pytest.raises(ObservabilityCredentialError):
        _gateway_env(telemetry_login=login)


def test_a_build_render_defers_the_login():
    env = _gateway_env(telemetry_login=None, defer_unresolved_creds=True)
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in env


def test_the_gateway_needs_a_resolved_address():
    with pytest.raises(TelemetryConfigError, match="gateway address"):
        _gateway_env(gateway_url=None)


@pytest.mark.parametrize("backend", ["openobserve", "otel", "Gateway"])
def test_any_other_backend_is_refused_by_the_ordinary_check(backend):
    with pytest.raises(TelemetryConfigError, match="'gateway'.*'generic'"):
        _build_telemetry_env({"enabled": True, "backend": backend}, gateway_url=GATEWAY)


def test_backend_generic_is_the_same_block_as_no_backend():
    """``generic`` names a collector of its own, exactly as an unset backend does."""
    assert _build_telemetry_env(
        {**_GENERIC_CFG, "backend": "generic"}, auth_token="tok"
    ) == _build_telemetry_env(_GENERIC_CFG, auth_token="tok")


def test_content_capture_to_the_gateway_raises_no_off_host_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _gateway_env()


def test_content_capture_to_a_collector_of_its_own_still_warns():
    cfg = {"enabled": True, "endpoint": "https://collector.example.org:4318"}
    with pytest.warns(UserWarning, match="leave the host"):
        _build_telemetry_env(cfg)


def test_generic_collector_output_is_byte_equal():
    """The ``generic`` golden: a block naming its own collector is untouched by the gateway."""
    env = _build_telemetry_env(
        _GENERIC_CFG,
        auth_token="tok",
        gateway_url=GATEWAY,
        telemetry_login=LOGIN,
        terminal_session="chat-1",
    )
    assert json.dumps(env, sort_keys=True) == json.dumps(
        {
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "https://collector.example.org:4318",
            "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer tok",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "OTEL_LOGS_EXPORTER": "otlp",
            "OTEL_LOG_ASSISTANT_RESPONSES": "0",
            "OTEL_LOG_RAW_API_BODIES": "0",
            "OTEL_LOG_TOOL_CONTENT": "0",
            "OTEL_LOG_TOOL_DETAILS": "0",
            "OTEL_LOG_USER_PROMPTS": "0",
            "OTEL_METRICS_EXPORTER": "otlp",
            "OTEL_RESOURCE_ATTRIBUTES": "service.name=osprey",
            "OTEL_TRACES_EXPORTER": "otlp",
        },
        sort_keys=True,
    )


# --- the terminal session at the launch sites ------------------------------


def test_a_launch_site_fills_the_gateway_session_attribute():
    inherited = _gateway_env(cfg={"resource_attributes": {"service.name": "osprey"}})
    added = terminal_session_env("abc-123", inherited)
    assert added == {
        "OSPREY_TERMINAL_SESSION": "abc-123",
        "OTEL_RESOURCE_ATTRIBUTES": "service.name=osprey,terminal.session=abc-123",
    }


def test_a_launch_site_leaves_a_collector_of_its_owns_attributes_alone():
    inherited = {"OTEL_RESOURCE_ATTRIBUTES": "service.name=osprey"}
    assert terminal_session_env("abc", inherited) == {"OSPREY_TERMINAL_SESSION": "abc"}
    assert terminal_session_env("abc", {}) == {"OSPREY_TERMINAL_SESSION": "abc"}


def test_a_host_launch_mints_a_chat_session_and_no_posture():
    environ: dict[str, str] = {}
    session = export_host_terminal_session(environ)
    assert session.startswith("chat-") and len(session) == len("chat-") + 36
    assert environ == {"OSPREY_TERMINAL_SESSION": session}


def test_a_host_launch_inside_a_terminal_keeps_that_terminals_session():
    environ = {"OSPREY_POSTURE_SESSION": "pty-key"}
    assert export_host_terminal_session(environ) == "pty-key"
    assert environ["OSPREY_TERMINAL_SESSION"] == "pty-key"
    environ = {"OSPREY_TERMINAL_SESSION": "dispatch-r1"}
    assert export_host_terminal_session(environ) == "dispatch-r1"


def test_the_literals_match_their_owners():
    from osprey.audit.envelope import TERMINAL_SESSION_ENV
    from osprey.deployment.telemetry_credentials import LOGIN_ENV

    assert cct.TERMINAL_SESSION_ENV == TERMINAL_SESSION_ENV
    assert cct.TELEMETRY_LOGIN_ENV == LOGIN_ENV


def test_the_resolver_reads_login_and_session_from_the_launch_environment():
    from osprey.agent_runner.provider_env import ClaudeCodeModelResolver

    spec = ClaudeCodeModelResolver.resolve(
        {"provider": "anthropic", "telemetry": dict(_GATEWAY_CFG)},
        environ={"OSPREY_TELEMETRY_LOGIN": LOGIN, "OSPREY_TERMINAL_SESSION": "chat-9"},
        gateway_url=GATEWAY,
    )
    assert spec is not None
    assert spec.env_block["OTEL_EXPORTER_OTLP_ENDPOINT"] == GATEWAY
    assert spec.env_block["OTEL_RESOURCE_ATTRIBUTES"] == "terminal.session=chat-9"


def test_the_resolver_refuses_a_launch_without_a_login():
    from osprey.agent_runner.provider_env import ClaudeCodeModelResolver

    with pytest.raises(ObservabilityCredentialError):
        ClaudeCodeModelResolver.resolve(
            {"provider": "anthropic", "telemetry": dict(_GATEWAY_CFG)},
            environ={},
            gateway_url=GATEWAY,
        )


# --- dispatch: the job's terminal session, never a posture session -----------


def test_a_dispatch_job_carries_the_session_without_a_posture():
    """A dispatch job's ledger ``session`` and OTLP resource carry ``dispatch-<run id>``.

    Run in a child process so the posture and envelope modules read exactly the
    environment a dispatched job is spawned with: ``OSPREY_TERMINAL_SESSION`` set
    by the worker, no posture pair.
    """
    inherited = _gateway_env()
    job_env = terminal_session_env(f"{cct.DISPATCH_SESSION_PREFIX}run-7", inherited)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OSPREY_POSTURE_SESSION", "OSPREY_POSTURE_SOURCE")
    }
    env.update(job_env)
    probe = (
        "import json\n"
        "from osprey.audit import posture\n"
        "from osprey.audit.envelope import terminal_session\n"
        "print(json.dumps([posture.posture_session(), terminal_session()]))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
    )
    posture_session, ledger_session = json.loads(out.stdout.strip().splitlines()[-1])
    assert posture_session is None
    assert ledger_session == "dispatch-run-7"
    assert job_env["OTEL_RESOURCE_ATTRIBUTES"] == "terminal.session=dispatch-run-7"


# --- the session is carried into the sandbox and never pinnable --------------


def test_the_sandbox_child_keeps_the_terminal_session_and_drops_its_neighbours():
    from osprey.mcp_server.sandbox_env import (
        SANDBOX_CHILD_ENV_ALLOW_NAMES,
        drop_sandbox_denied_env,
    )

    assert "OSPREY_TERMINAL_SESSION" in SANDBOX_CHILD_ENV_ALLOW_NAMES
    kept = drop_sandbox_denied_env(
        {
            "OSPREY_TERMINAL_SESSION": "s",
            "OSPREY_TERMINAL_SESSION_LIFETIME": "60",
            "OSPREY_TERMINAL_SESSION_STORE_DIR": "/x",
        }
    )
    assert kept == {"OSPREY_TERMINAL_SESSION": "s"}


def test_a_server_spec_cannot_pin_the_terminal_session():
    from osprey.registry.mcp import NON_PINNABLE_AUDIT_MARKERS

    assert "OSPREY_TERMINAL_SESSION" in NON_PINNABLE_AUDIT_MARKERS
