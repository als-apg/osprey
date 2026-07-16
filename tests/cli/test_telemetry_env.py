"""Tests for OpenObserve/OTEL telemetry env-block generation.

Covers the pure ``_build_telemetry_env`` helper (on/off gating, endpoint
resolution + fail-loud, OpenObserve Basic-auth header, content gates,
protocol/resource passthrough), its integration into ``resolve()``, and the
conflict-detection exemption for telemetry vars.
"""

from __future__ import annotations

import base64

import pytest

from osprey.cli import claude_code_resolver as resolver
from osprey.cli.claude_code_resolver import (
    MANAGED_ENV_VARS,
    ClaudeCodeModelResolver,
    ClaudeCodeModelSpec,
)
from osprey.cli.claude_code_telemetry import (
    TELEMETRY_ENV_VARS,
    TelemetryConfigError,
    _build_telemetry_env,
    _gate_is_on,
    _openobserve_host_override,
)

# ── on / off gating ──────────────────────────────────────────────


def test_absent_is_disabled():
    assert _build_telemetry_env(None) == {}


def test_empty_dict_is_disabled():
    assert _build_telemetry_env({}) == {}


def test_enabled_false_is_disabled():
    assert _build_telemetry_env({"enabled": False, "endpoint": "http://x:5080"}) == {}


def test_enabled_missing_is_disabled():
    """A config with no ``enabled`` key is treated as disabled."""
    assert _build_telemetry_env({"endpoint": "http://x:5080"}) == {}


# ── core env block ───────────────────────────────────────────────


def test_core_keys_present():
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://collector:4318"})
    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert env["OTEL_METRICS_EXPORTER"] == "otlp"
    assert env["OTEL_LOGS_EXPORTER"] == "otlp"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4318"


def test_all_values_are_strings():
    """Never emit bool ``True``/``False`` — only the string ``"1"``."""
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://collector:4318"})
    for value in env.values():
        assert isinstance(value, str)


def test_protocol_default():
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://c:4318"})
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "http/protobuf"


def test_protocol_override():
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://c:4318", "protocol": "grpc"})
    assert env["OTEL_EXPORTER_OTLP_PROTOCOL"] == "grpc"


# ── resource attributes ──────────────────────────────────────────


def test_resource_attributes_string_passthrough():
    env = _build_telemetry_env(
        {
            "enabled": True,
            "endpoint": "http://c:4318",
            "resource_attributes": "service.name=osprey,deployment=als",
        }
    )
    assert env["OTEL_RESOURCE_ATTRIBUTES"] == "service.name=osprey,deployment=als"


def test_resource_attributes_dict_rendered():
    env = _build_telemetry_env(
        {
            "enabled": True,
            "endpoint": "http://c:4318",
            "resource_attributes": {"service.name": "osprey", "deployment": "als"},
        }
    )
    assert env["OTEL_RESOURCE_ATTRIBUTES"] == "service.name=osprey,deployment=als"


def test_resource_attributes_absent():
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://c:4318"})
    assert "OTEL_RESOURCE_ATTRIBUTES" not in env


# ── content gates ────────────────────────────────────────────────

CONTENT_GATES = [
    ("OTEL_LOG_USER_PROMPTS", "log_user_prompts"),
    ("OTEL_LOG_ASSISTANT_RESPONSES", "log_assistant_responses"),
    ("OTEL_LOG_TOOL_DETAILS", "log_tool_details"),
    ("OTEL_LOG_RAW_API_BODIES", "log_raw_api_bodies"),
]


def test_content_gates_default_on():
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://c:4318"})
    for env_var, _cfg_key in CONTENT_GATES:
        assert env[env_var] == "1"


def test_tool_content_never_wired():
    """OTEL_LOG_TOOL_CONTENT requires tracing and is out of scope."""
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://c:4318"})
    assert "OTEL_LOG_TOOL_CONTENT" not in env


@pytest.mark.parametrize("env_var,cfg_key", CONTENT_GATES)
def test_each_content_gate_toggle_drops_exactly_one_key(env_var, cfg_key):
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://c:4318", cfg_key: False})
    assert env_var not in env
    # every OTHER gate stays on
    for other_var, _other_key in CONTENT_GATES:
        if other_var != env_var:
            assert env[other_var] == "1"


def test_content_gate_true_is_on():
    """Explicit ``true`` keeps the gate on (only ``false`` suppresses)."""
    env = _build_telemetry_env(
        {"enabled": True, "endpoint": "http://c:4318", "log_user_prompts": True}
    )
    assert env["OTEL_LOG_USER_PROMPTS"] == "1"


# ── endpoint context + fail-loud (Task 2.3) ──────────────────────


def test_endpoint_verbatim_wins():
    env = _build_telemetry_env(
        {
            "enabled": True,
            "backend": "openobserve",
            "endpoint": "http://explicit:9999/api/x",
            "openobserve": {"user": "u", "password": "p"},
        }
    )
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://explicit:9999/api/x"


def test_openobserve_default_endpoint_localhost():
    env = _build_telemetry_env(
        {
            "enabled": True,
            "backend": "openobserve",
            "openobserve": {"user": "u", "password": "p"},
        },
        in_container=False,
    )
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:5080/api/default"


def test_openobserve_default_endpoint_container_host():
    env = _build_telemetry_env(
        {
            "enabled": True,
            "backend": "openobserve",
            "openobserve": {"user": "u", "password": "p"},
        },
        in_container=True,
    )
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://openobserve:5080/api/default"


def test_openobserve_org_path_honored():
    env = _build_telemetry_env(
        {
            "enabled": True,
            "backend": "openobserve",
            "openobserve": {"user": "u", "password": "p", "org": "als"},
        },
        in_container=False,
    )
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:5080/api/als"


def test_enabled_no_endpoint_non_openobserve_raises():
    with pytest.raises(ValueError):
        _build_telemetry_env({"enabled": True, "backend": "jaeger"})


def test_enabled_no_endpoint_no_backend_raises():
    with pytest.raises(ValueError):
        _build_telemetry_env({"enabled": True})


def test_unresolved_var_in_endpoint_fails_loud():
    with pytest.raises(ValueError):
        _build_telemetry_env({"enabled": True, "endpoint": "http://${OO_HOST}:5080/api/default"})


def test_endpoint_context_and_failloud():
    """Named validation gate: container host, localhost, org path, ${ fail-loud."""
    base = {
        "enabled": True,
        "backend": "openobserve",
        "openobserve": {"user": "u", "password": "p", "org": "als"},
    }
    in_container = _build_telemetry_env(base, in_container=True)
    assert in_container["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://openobserve:5080/api/als"

    on_host = _build_telemetry_env(base, in_container=False)
    assert on_host["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:5080/api/als"

    with pytest.raises(ValueError):
        _build_telemetry_env({"enabled": True, "endpoint": "http://${X}:5080"})


# ── OpenObserve Basic-auth header (Task 2.2) ─────────────────────


def test_basic_auth_header():
    env = _build_telemetry_env(
        {
            "enabled": True,
            "backend": "openobserve",
            "openobserve": {"user": "root@example.com", "password": "s3cr3t"},
        },
        in_container=False,
    )
    headers = env["OTEL_EXPORTER_OTLP_HEADERS"]
    assert headers.startswith("Authorization=Basic ")
    token = headers.split("Authorization=Basic ", 1)[1]
    assert base64.b64decode(token).decode() == "root@example.com:s3cr3t"


def test_basic_auth_missing_creds_raises():
    with pytest.raises(ValueError):
        _build_telemetry_env(
            {"enabled": True, "backend": "openobserve", "openobserve": {"user": "u"}}
        )


def test_basic_auth_blank_creds_raises():
    with pytest.raises(ValueError):
        _build_telemetry_env(
            {
                "enabled": True,
                "backend": "openobserve",
                "openobserve": {"user": "", "password": ""},
            }
        )


def test_basic_auth_no_openobserve_block_raises():
    with pytest.raises(ValueError):
        _build_telemetry_env({"enabled": True, "backend": "openobserve"})


@pytest.mark.parametrize(
    "creds",
    [
        {"user": "${ZO_ROOT_USER_EMAIL}", "password": "p"},
        {"user": "u", "password": "${ZO_ROOT_USER_PASSWORD}"},
    ],
)
def test_creds_failloud_on_unresolved_var(creds):
    """An unresolved ${VAR} in a credential fails loud at resolve() time.

    The config loader leaves the literal ``${VAR}`` when the env var is unset;
    base64-encoding it would silently 401 against OpenObserve at runtime.
    """
    with pytest.raises(ValueError):
        _build_telemetry_env({"enabled": True, "backend": "openobserve", "openobserve": creds})


def test_config_headers_merge_auth_wins():
    """Config headers are merged; computed auth wins on key collision."""
    env = _build_telemetry_env(
        {
            "enabled": True,
            "backend": "openobserve",
            "openobserve": {"user": "u", "password": "p"},
            "headers": {"X-Trace": "abc", "Authorization": "Basic stale"},
        },
        in_container=False,
    )
    headers = env["OTEL_EXPORTER_OTLP_HEADERS"]
    assert "X-Trace=abc" in headers
    expected = base64.b64encode(b"u:p").decode()
    assert f"Authorization=Basic {expected}" in headers
    assert "Basic stale" not in headers


def test_config_headers_string_form():
    """A pre-formatted comma-separated header string is accepted."""
    env = _build_telemetry_env(
        {
            "enabled": True,
            "endpoint": "http://c:4318",
            "headers": "X-Trace=abc,X-Env=prod",
        }
    )
    headers = env["OTEL_EXPORTER_OTLP_HEADERS"]
    assert "X-Trace=abc" in headers
    assert "X-Env=prod" in headers


def test_no_headers_when_none_configured():
    """Non-openobserve backend with no headers emits no HEADERS var."""
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://c:4318"})
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in env


# ── TELEMETRY_ENV_VARS invariants ────────────────────────────────


def test_telemetry_vars_not_in_managed_set():
    """Telemetry vars must NOT be scrubbed as provider/backend selectors."""
    assert TELEMETRY_ENV_VARS.isdisjoint(MANAGED_ENV_VARS)


def test_telemetry_env_vars_covers_all_emitted_keys():
    """Every key the helper can emit is declared in TELEMETRY_ENV_VARS."""
    env = _build_telemetry_env(
        {
            "enabled": True,
            "backend": "openobserve",
            "openobserve": {"user": "u", "password": "p"},
            "resource_attributes": "service.name=osprey",
            "headers": {"X-Trace": "abc"},
        },
        in_container=False,
    )
    assert set(env).issubset(TELEMETRY_ENV_VARS)


# ── resolve() integration (Task 1.2) ─────────────────────────────


def test_resolve_env_block(monkeypatch):
    """resolve() with a provider + enabled telemetry folds telemetry into env_block."""
    monkeypatch.setattr(resolver, "_running_in_container", lambda: False)
    spec = ClaudeCodeModelResolver.resolve(
        {
            "provider": "anthropic",
            "telemetry": {
                "enabled": True,
                "backend": "openobserve",
                "openobserve": {"user": "u", "password": "p"},
            },
        }
    )
    assert spec is not None
    assert spec.env_block["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert spec.env_block["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:5080/api/default"


def test_resolve_container_endpoint(monkeypatch):
    monkeypatch.setattr(resolver, "_running_in_container", lambda: True)
    spec = ClaudeCodeModelResolver.resolve(
        {
            "provider": "anthropic",
            "telemetry": {
                "enabled": True,
                "backend": "openobserve",
                "openobserve": {"user": "u", "password": "p"},
            },
        }
    )
    assert spec.env_block["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://openobserve:5080/api/default"


def test_resolve_no_telemetry_leaves_env_block_clean():
    """Absent telemetry block == disabled; no OTEL vars leak into env_block."""
    spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
    assert spec is not None
    assert not any(k in spec.env_block for k in TELEMETRY_ENV_VARS)


def test_resolve_telemetry_vars_not_added_to_managed_set(monkeypatch):
    """Injecting telemetry must never mutate MANAGED_ENV_VARS."""
    monkeypatch.setattr(resolver, "_running_in_container", lambda: False)
    ClaudeCodeModelResolver.resolve(
        {"provider": "anthropic", "telemetry": {"enabled": True, "endpoint": "http://c:4318"}}
    )
    assert TELEMETRY_ENV_VARS.isdisjoint(MANAGED_ENV_VARS)


# ── conflict-detection exemption (Task 1.4) ──────────────────────


def test_no_conflict_on_preexisting_otel():
    """A differing shell OTEL_* export is NOT flagged as a conflict."""
    spec = ClaudeCodeModelSpec(
        provider="test",
        env_block={
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:5080/api/default",
            "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
            "ANTHROPIC_MODEL": "claude-opus-4-6",
        },
    )
    conflicts = spec.detect_env_conflicts(
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://some-other-collector:4318",
            "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
            "ANTHROPIC_MODEL": "stale-model",
        }
    )
    # No telemetry var is flagged...
    for var in TELEMETRY_ENV_VARS:
        assert var not in conflicts
    # ...but a genuine provider-var mismatch still is.
    assert conflicts["ANTHROPIC_MODEL"] == ("stale-model", "claude-opus-4-6")


# ── OpenObserve host override (deploy-topology aware, F1) ─────────


_OO_CFG = {
    "enabled": True,
    "backend": "openobserve",
    "openobserve": {"user": "u", "password": "p"},
}


def test_host_override_wins_over_derivation():
    """An explicit host beats both the localhost and container-DNS derivation."""
    # Not in a container -> would derive localhost; override wins.
    env = _build_telemetry_env(_OO_CFG, in_container=False, openobserve_host="openobserve")
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://openobserve:5080/api/default"
    # In a container -> would derive "openobserve"; override still wins.
    env2 = _build_telemetry_env(_OO_CFG, in_container=True, openobserve_host="oo-host")
    assert env2["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://oo-host:5080/api/default"


def test_host_override_none_falls_through_to_derivation():
    """No override + not in a container -> localhost (the ALS host-net guard)."""
    env = _build_telemetry_env(_OO_CFG, in_container=False, openobserve_host=None)
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:5080/api/default"


def test_host_override_helper_empty_string_is_none(monkeypatch):
    monkeypatch.delenv("OSPREY_OTEL_OPENOBSERVE_HOST", raising=False)
    assert _openobserve_host_override() is None
    monkeypatch.setenv("OSPREY_OTEL_OPENOBSERVE_HOST", "")
    assert _openobserve_host_override() is None
    monkeypatch.setenv("OSPREY_OTEL_OPENOBSERVE_HOST", "openobserve")
    assert _openobserve_host_override() == "openobserve"


def test_resolve_consults_host_override(monkeypatch):
    """resolve() threads OSPREY_OTEL_OPENOBSERVE_HOST into the endpoint even
    when not detected as in-container (the podman-bridge fix)."""
    monkeypatch.setattr(resolver, "_running_in_container", lambda: False)
    monkeypatch.setattr(resolver, "_openobserve_host_override", lambda: "openobserve")
    spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic", "telemetry": _OO_CFG})
    assert spec.env_block["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://openobserve:5080/api/default"


# ── content-gate truthiness + off-host advisory (F5) ─────────────


@pytest.mark.parametrize("env_var,cfg_key", CONTENT_GATES)
@pytest.mark.parametrize(
    "falsey", [False, "false", "False", "FALSE", "0", "no", "off", " false ", ""]
)
def test_falsey_gate_values_suppress(env_var, cfg_key, falsey):
    """bool False AND false-y strings (incl. ${VAR:-false} -> "false") drop the gate."""
    env = _build_telemetry_env(
        {
            "enabled": True,
            "backend": "openobserve",
            "openobserve": {"user": "u", "password": "p"},
            cfg_key: falsey,
        }
    )
    assert env_var not in env


@pytest.mark.parametrize("truthy", [True, "true", "True", "1", "yes"])
def test_truthy_gate_values_stay_on(truthy):
    env = _build_telemetry_env(
        {
            "enabled": True,
            "backend": "openobserve",
            "openobserve": {"user": "u", "password": "p"},
            "log_user_prompts": truthy,
        }
    )
    assert env["OTEL_LOG_USER_PROMPTS"] == "1"


def test_gate_is_on_helper():
    assert _gate_is_on(None) is True  # missing key -> on
    assert _gate_is_on(True) is True
    assert _gate_is_on(False) is False
    assert _gate_is_on("false") is False
    assert _gate_is_on("FALSE") is False
    assert _gate_is_on("true") is True


def test_warns_on_content_capture_non_openobserve():
    """Full content capture to a non-openobserve backend leaves the host -> warn."""
    with pytest.warns(UserWarning, match="leave the host"):
        _build_telemetry_env({"enabled": True, "endpoint": "http://collector:4318"})


def test_no_warning_for_openobserve_backend(recwarn):
    _build_telemetry_env(
        {"enabled": True, "backend": "openobserve", "openobserve": {"user": "u", "password": "p"}}
    )
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]


def test_no_warning_when_all_content_off(recwarn):
    """A non-openobserve backend with every content gate off must not warn."""
    _build_telemetry_env(
        {
            "enabled": True,
            "endpoint": "http://collector:4318",
            "log_user_prompts": False,
            "log_assistant_responses": False,
            "log_tool_details": False,
            "log_raw_api_bodies": False,
        }
    )
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]


# ── telemetry-specific error type (F4) ───────────────────────────


def test_telemetry_misconfig_raises_telemetry_config_error():
    """Telemetry faults raise TelemetryConfigError (a ValueError subclass), so a
    caller can single out a telemetry misconfig without catching every ValueError."""
    assert issubclass(TelemetryConfigError, ValueError)
    with pytest.raises(TelemetryConfigError):
        _build_telemetry_env(
            {"enabled": True, "backend": "openobserve", "openobserve": {"user": "u"}}
        )
    with pytest.raises(TelemetryConfigError):
        _build_telemetry_env({"enabled": True})
