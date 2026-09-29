"""Tests for OpenObserve/OTEL telemetry env-block generation.

Covers the pure ``_build_telemetry_env`` helper (on/off gating, endpoint
resolution + fail-loud, OpenObserve Basic-auth header, content gates,
protocol/resource passthrough), its integration into ``resolve()``, and the
conflict-detection exemption for telemetry vars.
"""

from __future__ import annotations

import base64

import pytest

from osprey.build import claude_code_resolver as resolver
from osprey.build.claude_code_resolver import (
    MANAGED_ENV_VARS,
    ClaudeCodeModelResolver,
    ClaudeCodeModelSpec,
)
from osprey.build.claude_code_telemetry import (
    _TELEMETRY_CONTENT_GATES,
    TELEMETRY_ENV_VARS,
    ObservabilityCredentialError,
    TelemetryConfigError,
    _build_telemetry_env,
    _gate_is_on,
    _openobserve_host_override,
    _running_in_container,
    telemetry_auth_token_env,
)
from osprey.port_layout import default_port

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
    ("OTEL_LOG_TOOL_CONTENT", "log_tool_content"),
]


def test_content_gates_default_on():
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://c:4318"})
    for env_var, _cfg_key in CONTENT_GATES:
        assert env[env_var] == "1"


def test_traces_are_exported_with_the_tracing_switch():
    """Traces ride the enabled block, with the switch that turns tracing on."""
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://c:4318"})
    assert env["OTEL_TRACES_EXPORTER"] == "otlp"
    assert env["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] == "1"
    assert env["OTEL_LOG_TOOL_CONTENT"] == "1"
    assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in env


def test_content_max_length_is_emitted_as_a_string():
    for raw in (262144, "262144"):
        env = _build_telemetry_env(
            {"enabled": True, "endpoint": "http://c:4318", "content_max_length": raw}
        )
        assert env["CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH"] == "262144"


def test_content_max_length_absent_leaves_claude_codes_default():
    """Unset or empty emits nothing, so Claude Code's own limit applies."""
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://c:4318"})
    assert "CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH" not in env
    env = _build_telemetry_env(
        {"enabled": True, "endpoint": "http://c:4318", "content_max_length": ""}
    )
    assert "CLAUDE_CODE_OTEL_CONTENT_MAX_LENGTH" not in env


@pytest.mark.parametrize("bad", ["big", 0, -1, True, "${OTEL_LIMIT}"])
def test_content_max_length_must_be_a_positive_integer(bad):
    with pytest.raises(TelemetryConfigError, match="content_max_length"):
        _build_telemetry_env(
            {"enabled": True, "endpoint": "http://c:4318", "content_max_length": bad}
        )


@pytest.mark.parametrize("env_var,cfg_key", CONTENT_GATES)
def test_each_content_gate_toggle_zeroes_exactly_one_key(env_var, cfg_key):
    """A disabled gate ships an explicit "0" — omission would let the CLI's
    own fallback chain (e.g. ASSISTANT_RESPONSES ?? USER_PROMPTS) re-enable it."""
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://c:4318", cfg_key: False})
    assert env[env_var] == "0"
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


@pytest.mark.parametrize(
    "creds",
    [
        {"user": "${ZO_ROOT_USER_EMAIL}", "password": "p"},
        {"user": "u", "password": "${ZO_ROOT_USER_PASSWORD}"},
    ],
)
def test_creds_deferred_at_build_time_omit_header(creds, recwarn):
    """``defer_unresolved_creds`` downgrades the hard failure to a warning.

    A build renders a project whose telemetry credentials are supplied by the
    *deployment* (the worker re-resolves them against its own .env at
    agent-spawn — see dispatch_api). Aborting the build there would force every
    such project to hand the builder production secrets. The header is omitted
    rather than encoded from a placeholder, so nothing bogus is baked.
    """
    env = _build_telemetry_env(
        {"enabled": True, "backend": "openobserve", "openobserve": creds},
        defer_unresolved_creds=True,
    )

    assert "OTEL_EXPORTER_OTLP_HEADERS" not in env
    # Telemetry itself stays configured — only the auth header is deferred.
    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert any("unresolved" in str(w.message) for w in recwarn)


def test_creds_deferred_flag_does_not_excuse_missing_creds():
    """Deferral covers an unresolved ${VAR}, not an absent credential.

    A blank/missing credential is a config error at every stage — there is no
    later resolution step that could fill it in.
    """
    with pytest.raises(ValueError):
        _build_telemetry_env(
            {"enabled": True, "backend": "openobserve", "openobserve": {"user": "u"}},
            defer_unresolved_creds=True,
        )


def test_creds_default_still_fails_loud_for_runtime():
    """The default is unchanged: spawn-time callers must still fail loud.

    The dispatch worker re-resolves at agent-spawn with the deployment's own
    .env; there is no later stage, so an unresolved cred there is terminal for
    telemetry and must not be papered over.
    """
    with pytest.raises(ValueError):
        _build_telemetry_env(
            {
                "enabled": True,
                "backend": "openobserve",
                "openobserve": {"user": "${ZO_ROOT_USER_EMAIL}", "password": "p"},
            }
        )


def test_no_headers_when_none_configured():
    """Non-openobserve backend with no headers emits no HEADERS var."""
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://c:4318"})
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in env


def test_headers_key_is_refused_naming_auth_token_env():
    """The free-form header map is gone; the refusal names its replacement."""
    with pytest.raises(TelemetryConfigError, match="claude_code.telemetry.auth.token_env"):
        _build_telemetry_env(
            {
                "enabled": True,
                "endpoint": "http://c:4318",
                "headers": {"Authorization": "Bearer t"},
            }
        )


def test_headers_on_a_disabled_block_is_inert():
    """A disabled block exports nothing, whatever keys it carries."""
    assert _build_telemetry_env({"enabled": False, "headers": {"X-Trace": "abc"}}) == {}


# ── collector bearer token (auth.token_env) ─────────────────────

_COLLECTOR_CFG = {
    "enabled": True,
    "backend": "generic",
    "endpoint": "https://collector.example.org:4318",
    "auth": {"token_env": "OTLP_TOKEN"},
    # Content capture off: an off-host backend warns about it otherwise, and
    # these tests assert on the warnings the token path records.
    **dict.fromkeys(_TELEMETRY_CONTENT_GATES.values(), False),
}


def test_bearer_token_header_from_the_named_variable():
    """The token handed in is sent as a bearer token."""
    env = _build_telemetry_env(_COLLECTOR_CFG, auth_token="s3cret-token")
    assert env["OTEL_EXPORTER_OTLP_HEADERS"] == "Authorization=Bearer s3cret-token"
    assert telemetry_auth_token_env(_COLLECTOR_CFG) == "OTLP_TOKEN"


def test_bearer_token_absent_at_launch_names_the_variable():
    """No token at launch is a credential refusal naming the variable."""
    with pytest.raises(ObservabilityCredentialError) as excinfo:
        _build_telemetry_env(_COLLECTOR_CFG)
    assert excinfo.value.unresolved_vars == ("OTLP_TOKEN",)
    assert "OTLP_TOKEN" in str(excinfo.value)
    assert "claude_code.telemetry.auth.token_env" in str(excinfo.value)


def test_bearer_token_blank_is_refused_like_unset():
    """Compose's ``${VAR:-}`` makes an unset variable a blank one: same fault."""
    with pytest.raises(ObservabilityCredentialError) as excinfo:
        _build_telemetry_env(_COLLECTOR_CFG, auth_token="  ")
    assert excinfo.value.unresolved_vars == ("OTLP_TOKEN",)


def test_bearer_token_deferred_at_build_omits_the_header(recwarn):
    """A build never carries the token, so its absence there says nothing."""
    env = _build_telemetry_env(_COLLECTOR_CFG, defer_unresolved_creds=True)
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in env
    assert len(recwarn) == 0


def test_token_env_holding_a_value_is_refused_without_echoing_it():
    """A value where a name belongs may already be the expanded secret."""
    cfg = {**_COLLECTOR_CFG, "auth": {"token_env": "abc.def-123"}}
    with pytest.raises(TelemetryConfigError) as excinfo:
        _build_telemetry_env(cfg, auth_token="x")
    assert "abc.def-123" not in str(excinfo.value)
    assert "claude_code.telemetry.auth.token_env" in str(excinfo.value)


def test_token_with_a_comma_is_refused_without_echoing_it():
    """A comma would split the comma-joined OTLP header list."""
    with pytest.raises(ObservabilityCredentialError) as excinfo:
        _build_telemetry_env(_COLLECTOR_CFG, auth_token="abc,X-Evil=1")
    assert "abc,X-Evil=1" not in str(excinfo.value)
    assert "OTLP_TOKEN" in str(excinfo.value)


def test_auth_on_the_openobserve_backend_is_refused():
    """The bundled store authenticates with its own account; two sources is a fault."""
    cfg = {**_OO_CFG, "auth": {"token_env": "OTLP_TOKEN"}}
    with pytest.raises(TelemetryConfigError, match="claude_code.telemetry.auth"):
        _build_telemetry_env(cfg, auth_token="x")


@pytest.mark.parametrize(
    "auth",
    [
        {"token_env": "OTLP_TOKEN", "username": "u"},
        {"username": "u", "password_env": "OTLP_PASSWORD"},
        "OTLP_TOKEN",
    ],
)
def test_auth_members_other_than_token_env_are_refused(auth):
    """The collector takes a bearer token and nothing else."""
    cfg = {**_COLLECTOR_CFG, "auth": auth}
    with pytest.raises(TelemetryConfigError, match="auth.token_env"):
        telemetry_auth_token_env(cfg)


def test_resolve_reads_the_token_from_the_supplied_environ_only(monkeypatch):
    """A render passes no environ, so the ambient token never reaches it."""
    monkeypatch.setenv("OTLP_TOKEN", "ambient-token")
    spec = ClaudeCodeModelResolver.resolve(
        {"provider": "anthropic", "telemetry": _COLLECTOR_CFG},
        defer_unresolved_telemetry_creds=True,
    )
    assert spec is not None
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in spec.env_block

    spec = ClaudeCodeModelResolver.resolve(
        {"provider": "anthropic", "telemetry": _COLLECTOR_CFG},
        environ={"OTLP_TOKEN": "handed-token"},
    )
    assert spec is not None
    assert spec.env_block["OTEL_EXPORTER_OTLP_HEADERS"] == "Authorization=Bearer handed-token"


def test_load_provider_spec_reads_the_token_from_the_project_env(tmp_path, monkeypatch):
    """The runtime launch reads the token from the project's ``.env``."""
    import yaml

    from osprey.build.claude_code_resolver import load_provider_spec

    monkeypatch.delenv("OTLP_TOKEN", raising=False)
    (tmp_path / "config.yml").write_text(
        yaml.safe_dump({"claude_code": {"provider": "anthropic", "telemetry": _COLLECTOR_CFG}}),
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("OTLP_TOKEN=from-dotenv\n", encoding="utf-8")
    spec = load_provider_spec(tmp_path)
    assert spec is not None
    assert spec.env_block["OTEL_EXPORTER_OTLP_HEADERS"] == "Authorization=Bearer from-dotenv"


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
            "content_max_length": 262144,
        },
        in_container=False,
    )
    env.update(_build_telemetry_env(_COLLECTOR_CFG, auth_token="t"))
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


class TestContainerDetection:
    """The marker files this module treats as "I am in a container".

    Mirrors ``tests/health/test_derive.py``'s coverage of the twin helper in
    ``osprey.health.derive``: the two are deliberate copies (health must not
    import the build layer), so each needs its own guard or one drifts silently.
    """

    def test_docker_marker_detected(self, monkeypatch):
        monkeypatch.delenv("OSPREY_IN_CONTAINER", raising=False)
        monkeypatch.setattr("os.path.exists", lambda p: p == "/.dockerenv")
        assert _running_in_container() is True

    def test_podman_marker_detected(self, monkeypatch):
        # Podman writes /run/.containerenv and no /.dockerenv, so a Docker-only
        # probe read every podman deployment as a host and derived the
        # localhost OpenObserve default from inside the container.
        monkeypatch.delenv("OSPREY_IN_CONTAINER", raising=False)
        monkeypatch.setattr("os.path.exists", lambda p: p == "/run/.containerenv")
        assert _running_in_container() is True

    def test_no_marker_is_a_host(self, monkeypatch):
        monkeypatch.delenv("OSPREY_IN_CONTAINER", raising=False)
        monkeypatch.setattr("os.path.exists", lambda p: False)
        assert _running_in_container() is False

    def test_operator_override_alone_is_enough(self, monkeypatch):
        monkeypatch.setattr("os.path.exists", lambda p: False)
        monkeypatch.setenv("OSPREY_IN_CONTAINER", "1")
        assert _running_in_container() is True

    def test_the_two_copies_probe_the_same_markers(self, monkeypatch):
        """The health twin and this one must answer identically, marker for marker.

        They are copies by design, which is exactly the arrangement that drifts:
        this asserts agreement on every case rather than trusting two docstrings
        to stay in sync.
        """
        from osprey.health.derive import _in_container

        monkeypatch.delenv("OSPREY_IN_CONTAINER", raising=False)
        for present in ("/.dockerenv", "/run/.containerenv", "/nothing"):
            monkeypatch.setattr("os.path.exists", lambda p, hit=present: p == hit)
            assert _running_in_container() == _in_container(), present


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
        default_model_id="m",
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


def test_openobserve_traces_share_the_base_endpoint():
    """One base endpoint serves every signal.

    The OTLP exporter appends ``/v1/traces`` to the base endpoint, which is
    OpenObserve's traces route, so no per-signal endpoint is emitted.
    """
    env = _build_telemetry_env(_OO_CFG, in_container=False, openobserve_port=15080)
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:15080/api/default"
    assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in env


def test_host_override_wins_over_derivation():
    """An explicit host beats both the localhost and container-DNS derivation."""
    # Not in a container -> would derive localhost; override wins.
    env = _build_telemetry_env(_OO_CFG, in_container=False, openobserve_host="openobserve")
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://openobserve:5080/api/default"
    # In a container -> would derive "openobserve"; override still wins.
    env2 = _build_telemetry_env(_OO_CFG, in_container=True, openobserve_host="oo-host")
    assert env2["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://oo-host:5080/api/default"


def test_host_override_none_falls_through_to_derivation():
    """No override + not in a container -> localhost (the host-network guard)."""
    env = _build_telemetry_env(_OO_CFG, in_container=False, openobserve_host=None)
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:5080/api/default"


def test_resolved_port_is_the_endpoints_port():
    """The endpoint dials the port handed in — never a literal.

    The store is reached on the port it PUBLISHES from the host and from a
    host-networked container, and a project moves that port freely; the 5080
    the image listens on is right only inside the store's own compose network.
    """
    env = _build_telemetry_env(_OO_CFG, in_container=False, openobserve_port=15080)
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:15080/api/default"
    env2 = _build_telemetry_env(
        _OO_CFG, in_container=True, openobserve_host="127.0.0.1", openobserve_port=15080
    )
    assert env2["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://127.0.0.1:15080/api/default"


def test_no_port_falls_back_to_the_listen_port():
    """Omitted, the port is the one the image listens on (5080)."""
    from osprey.build.claude_code_telemetry import OPENOBSERVE_LISTEN_PORT

    env = _build_telemetry_env(_OO_CFG, in_container=True, openobserve_port=None)
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == (
        f"http://openobserve:{OPENOBSERVE_LISTEN_PORT}/api/default"
    )


def test_published_port_reads_the_services_block():
    """``openobserve_published_port`` follows ``services.openobserve.port``.

    With no key it is the layout's ``openobserve`` slot, not the port the image
    listens on: the published port and the listen port are different numbers,
    and only the published one can be dialled from off the store's network.
    """
    from osprey.build.claude_code_telemetry import openobserve_published_port

    assert openobserve_published_port({"services": {"openobserve": {"port": 15080}}}) == 15080
    assert openobserve_published_port({"services": {"openobserve": {"port": "15080"}}}) == 15080
    assert openobserve_published_port({"services": {}}) == default_port("openobserve")
    assert openobserve_published_port({}) == default_port("openobserve")
    assert openobserve_published_port(None) == default_port("openobserve")
    with pytest.raises(TelemetryConfigError, match="services.openobserve.port"):
        openobserve_published_port({"services": {"openobserve": {"port": "five"}}})


def test_both_resolvers_follow_the_port_base(monkeypatch):
    """A moved base moves the store, with no ``services.openobserve.port`` key.

    The number is the layout's: base 20000 + the openobserve offset. Both the
    build-time reader and the runtime one derive it from the config in hand, so
    a second deployment on the host dials its own store rather than the first
    one's.
    """
    from osprey.build.claude_code_telemetry import (
        OPENOBSERVE_PORT_ENV_VAR,
        openobserve_published_port,
        resolve_openobserve_port,
    )

    monkeypatch.delenv(OPENOBSERVE_PORT_ENV_VAR, raising=False)
    moved = {"deployment": {"port_base": 20000}}
    assert openobserve_published_port(moved) == 20050
    assert resolve_openobserve_port(moved) == 20050
    # An explicit port still wins over the block it sits in.
    pinned = {"deployment": {"port_base": 20000}, "services": {"openobserve": {"port": 15080}}}
    assert openobserve_published_port(pinned) == 15080
    assert resolve_openobserve_port(pinned) == 15080


def test_port_override_helper(monkeypatch):
    """``OSPREY_OTEL_OPENOBSERVE_PORT`` is the port half of the host override."""
    from osprey.build.claude_code_telemetry import (
        OPENOBSERVE_PORT_ENV_VAR,
        _openobserve_port_override,
        resolve_openobserve_port,
    )

    published = {"services": {"openobserve": {"port": 15080}}}
    monkeypatch.delenv(OPENOBSERVE_PORT_ENV_VAR, raising=False)
    assert _openobserve_port_override() is None
    assert resolve_openobserve_port(published) == 15080
    monkeypatch.setenv(OPENOBSERVE_PORT_ENV_VAR, "")
    assert _openobserve_port_override() is None
    monkeypatch.setenv(OPENOBSERVE_PORT_ENV_VAR, "5080")
    assert _openobserve_port_override() == 5080
    # The bridge case: the compose author declared the listen port, and it
    # wins over the published port the config names.
    assert resolve_openobserve_port(published) == 5080
    monkeypatch.setenv(OPENOBSERVE_PORT_ENV_VAR, "x")
    with pytest.raises(TelemetryConfigError, match=OPENOBSERVE_PORT_ENV_VAR):
        _openobserve_port_override()


def test_load_provider_spec_dials_the_published_port(tmp_path, monkeypatch):
    """The runtime launch path threads ``services.openobserve.port`` through."""
    import yaml

    from osprey.build.claude_code_resolver import load_provider_spec
    from osprey.build.claude_code_telemetry import OPENOBSERVE_PORT_ENV_VAR

    monkeypatch.delenv(OPENOBSERVE_PORT_ENV_VAR, raising=False)
    monkeypatch.delenv("OSPREY_OTEL_OPENOBSERVE_HOST", raising=False)
    monkeypatch.setattr(resolver, "_running_in_container", lambda: False)
    (tmp_path / "config.yml").write_text(
        yaml.safe_dump(
            {
                "claude_code": {"provider": "anthropic", "telemetry": _OO_CFG},
                "services": {"openobserve": {"port": 15080}},
            }
        ),
        encoding="utf-8",
    )
    spec = load_provider_spec(tmp_path)
    assert spec is not None
    assert spec.env_block["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost:15080/api/default"


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
    """bool False AND false-y strings (incl. ${VAR:-false} -> "false") zero the gate."""
    env = _build_telemetry_env(
        {
            "enabled": True,
            "backend": "openobserve",
            "openobserve": {"user": "u", "password": "p"},
            cfg_key: falsey,
        }
    )
    assert env[env_var] == "0"


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
            "log_tool_content": False,
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


# ── observability-credential error type ──────────────────────────


def test_credential_error_subclass_chain():
    """The credential type nests inside the general one, which nests inside ValueError.

    Every existing handler catches one of the two outer types, so the narrower
    type must stay a subclass of both or those handlers stop seeing the failure
    they already handle today.
    """
    assert issubclass(ObservabilityCredentialError, TelemetryConfigError)
    assert issubclass(ObservabilityCredentialError, ValueError)


@pytest.mark.parametrize(
    "openobserve",
    [
        {"user": "u"},
        {"password": "p"},
        {},
        {"user": "", "password": ""},
    ],
    ids=["password-absent", "user-absent", "block-empty", "both-blank"],
)
def test_missing_or_blank_credential_raises_the_credential_type(openobserve):
    """A credential the operator has to supply reports itself as a credential fault."""
    with pytest.raises(ObservabilityCredentialError):
        _build_telemetry_env(
            {"enabled": True, "backend": "openobserve", "openobserve": openobserve}
        )


@pytest.mark.parametrize(
    "openobserve",
    [
        {"user": "${STORE_USER}", "password": "p"},
        {"user": "u", "password": "${STORE_PASSWORD}"},
    ],
    ids=["user-unresolved", "password-unresolved"],
)
def test_unresolved_credential_var_raises_the_credential_type(openobserve):
    """An unresolved ${VAR} credential is the same kind of fault as a blank one.

    Both are fixed by putting a value in the deployment's secret store, so both
    carry the type whose remedy says so.
    """
    with pytest.raises(ObservabilityCredentialError):
        _build_telemetry_env(
            {"enabled": True, "backend": "openobserve", "openobserve": openobserve}
        )


@pytest.mark.parametrize(
    ("openobserve", "expected"),
    [
        ({"user": "u", "password": "${STORE_PASSWORD}"}, ("STORE_PASSWORD",)),
        ({"user": "${STORE_USER}", "password": "p"}, ("STORE_USER",)),
        # A default the config author wrote is not part of the variable's name.
        ({"user": "u", "password": "${STORE_PASSWORD:-fallback}"}, ("STORE_PASSWORD",)),
        # Two references in one value: both are named, so a caller deciding
        # whether it can supply them itself sees the whole set.
        ({"user": "u", "password": "${A_TOKEN}-${B_TOKEN}"}, ("A_TOKEN", "B_TOKEN")),
    ],
    ids=["password", "user", "with-default", "two-references"],
)
def test_the_refusal_names_the_variables_it_could_not_resolve(openobserve, expected):
    """The names travel on the exception, not only inside its sentence.

    A caller has to decide whether the absent value is one its own deploy
    issues later — that decision is about a variable name, and recovering names
    by parsing prose makes it hostage to the wording of a human-facing message.
    """
    with pytest.raises(ObservabilityCredentialError) as caught:
        _build_telemetry_env(
            {"enabled": True, "backend": "openobserve", "openobserve": openobserve}
        )

    assert caught.value.unresolved_vars == expected


@pytest.mark.parametrize(
    "openobserve",
    [{"user": "u"}, {"user": "u", "password": ""}],
    ids=["absent", "blank"],
)
def test_a_missing_credential_names_no_variable(openobserve):
    """There is nothing to name, and nothing any later step could fill in — so a
    caller reading the names cannot mistake this for a value that is merely early."""
    with pytest.raises(ObservabilityCredentialError) as caught:
        _build_telemetry_env(
            {"enabled": True, "backend": "openobserve", "openobserve": openobserve}
        )

    assert caught.value.unresolved_vars == ()


def test_deferred_credential_still_raises_the_credential_type_when_absent():
    """Deferral covers an unresolved ${VAR}; an absent credential keeps its type."""
    with pytest.raises(ObservabilityCredentialError):
        _build_telemetry_env(
            {"enabled": True, "backend": "openobserve", "openobserve": {"user": "u"}},
            defer_unresolved_creds=True,
        )


@pytest.mark.parametrize(
    "telemetry",
    [
        {"enabled": True, "backend": "jaeger"},
        {"enabled": True},
        {"enabled": True, "endpoint": "http://${COLLECTOR_HOST}:4318"},
        {
            "enabled": True,
            "backend": "openobserve",
            "protocol": "grpc",
            "openobserve": {"user": "u", "password": "p"},
        },
    ],
    ids=["no-endpoint-other-backend", "no-endpoint-no-backend", "unresolved-var", "grpc-derived"],
)
def test_endpoint_faults_keep_the_general_telemetry_type(telemetry):
    """An endpoint fault is not a credential fault, and must not borrow its type.

    A handler that offers a credential remedy would send the operator to the
    secret store for a problem that lives in the endpoint config.
    """
    with pytest.raises(TelemetryConfigError) as caught:
        _build_telemetry_env(telemetry)
    assert not isinstance(caught.value, ObservabilityCredentialError)


def test_existing_broad_handlers_still_catch_a_credential_fault():
    """The callers that catch TelemetryConfigError or ValueError keep working."""
    cfg = {"enabled": True, "backend": "openobserve", "openobserve": {"user": "u"}}
    with pytest.raises(TelemetryConfigError):
        _build_telemetry_env(cfg)
    with pytest.raises(ValueError):
        _build_telemetry_env(cfg)


def test_deferred_var_credential_warns_not_raises_through_resolve(monkeypatch, recwarn):
    """Pins the build-time catch site's reachability contract through resolve().

    ``build_cmd.py``'s render call sets ``defer_unresolved_telemetry_creds=True``
    around ``load_provider_spec()`` and catches ``ValueError``. An unresolved
    ``${VAR}`` credential must warn rather than raise all the way through
    ``resolve()`` — not just at the ``_build_telemetry_env``/
    ``_openobserve_auth_header`` leaf — or the comment at that catch site goes
    stale silently. A missing/blank credential is a separate arm (see
    ``test_deferred_credential_still_raises_the_credential_type_when_absent``)
    and keeps raising even when deferred.
    """
    monkeypatch.setattr(resolver, "_running_in_container", lambda: False)
    spec = ClaudeCodeModelResolver.resolve(
        {
            "provider": "anthropic",
            "telemetry": {
                "enabled": True,
                "backend": "openobserve",
                "openobserve": {"user": "${STORE_USER}", "password": "p"},
            },
        },
        defer_unresolved_telemetry_creds=True,
    )
    assert spec is not None
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in spec.env_block
    assert spec.env_block["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert any("unresolved" in str(w.message) for w in recwarn)


class TestTelemetryPortIsATelemetryInput:
    """A malformed ``services.openobserve.port`` is a telemetry fault, and a
    caller that asked for no telemetry must not see it: the dispatch worker's
    degrade-and-retry re-resolves with ``include_telemetry=False`` precisely
    to keep the provider's auth when telemetry is misconfigured."""

    _CONFIG = (
        "api:\n"
        "  providers:\n"
        "    anthropic:\n"
        "      api_key: ${ANTHROPIC_API_KEY}\n"
        "claude_code:\n"
        "  provider: anthropic\n"
        "  telemetry:\n"
        "    enabled: true\n"
        "    backend: openobserve\n"
        "services:\n"
        "  openobserve:\n"
        "    port: not-a-port\n"
    )

    def test_without_telemetry_the_port_is_never_resolved(self, tmp_path, monkeypatch):
        from osprey.build.claude_code_resolver import load_provider_spec

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        (tmp_path / "config.yml").write_text(self._CONFIG)

        spec = load_provider_spec(tmp_path, include_telemetry=False)

        assert spec.provider == "anthropic"
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in spec.env_block

    def test_with_telemetry_the_port_fault_is_loud(self, tmp_path, monkeypatch):
        import pytest

        from osprey.build.claude_code_resolver import load_provider_spec
        from osprey.build.claude_code_telemetry import TelemetryConfigError

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        (tmp_path / "config.yml").write_text(self._CONFIG)

        with pytest.raises(TelemetryConfigError, match="services.openobserve.port"):
            load_provider_spec(tmp_path, include_telemetry=True)


# ── claude_code.telemetry.signals ─────────────────────────────────

_EXPORTERS = {
    "metrics": "OTEL_METRICS_EXPORTER",
    "logs": "OTEL_LOGS_EXPORTER",
    "traces": "OTEL_TRACES_EXPORTER",
}
_SIGNAL_KEYS = {*_EXPORTERS.values(), "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"}


def _signals_cfg(signals) -> dict:
    """An enabled openobserve block with inline credentials and ``signals``."""
    return {
        "enabled": True,
        "backend": "openobserve",
        "openobserve": {"user": "u", "password": "p"},
        "signals": signals,
    }


def _signals_env(signals) -> dict[str, str]:
    return _build_telemetry_env(_signals_cfg(signals), in_container=False)


def test_absent_signals_export_all_three():
    """Unset, ``None`` and the full list give the same block: every signal on."""
    absent = _build_telemetry_env(
        {"enabled": True, "backend": "openobserve", "openobserve": {"user": "u", "password": "p"}},
        in_container=False,
    )
    assert _signals_env(None) == absent
    assert _signals_env(["metrics", "logs", "traces"]) == absent
    for exporter in _EXPORTERS.values():
        assert absent[exporter] == "otlp"
    assert absent["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] == "1"


@pytest.mark.parametrize(
    "signals",
    [["metrics", "logs"], ["traces"], ["logs"], ["metrics", "traces"]],
)
def test_a_signal_left_out_is_exported_as_none(signals):
    """Each exporter is ``otlp`` when listed and ``none`` otherwise, never missing."""
    env = _signals_env(signals)
    for signal, exporter in _EXPORTERS.items():
        assert env[exporter] == ("otlp" if signal in signals else "none"), exporter
    expected_switch = "1" if "traces" in signals else "0"
    assert env["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] == expected_switch


def test_signals_change_nothing_but_the_exporters():
    """Endpoint, headers, protocol and content gates do not follow ``signals``."""
    default = _signals_env(None)
    narrowed = _signals_env(["metrics"])
    assert set(narrowed) == set(default)
    for key in set(default) - _SIGNAL_KEYS:
        assert narrowed[key] == default[key], key


def test_a_repeated_signal_counts_once():
    assert _signals_env(["logs", "logs"]) == _signals_env(["logs"])


@pytest.mark.parametrize(
    ("signals", "named"),
    [
        (["metrics", "spans"], "spans"),
        (["Traces"], "Traces"),
        ("metrics", "must be a list"),
        ({"metrics": True}, "must be a list"),
    ],
)
def test_signals_outside_the_three_are_refused(signals, named):
    with pytest.raises(TelemetryConfigError, match="claude_code.telemetry.signals") as exc:
        _signals_env(signals)
    assert named in str(exc.value)


def test_empty_signals_are_refused_toward_enabled_false():
    with pytest.raises(TelemetryConfigError, match="enabled: false"):
        _signals_env([])


def test_signals_are_inert_while_telemetry_is_off():
    """A disabled block is not validated, as with every other key in it."""
    assert _build_telemetry_env({"enabled": False, "signals": []}) == {}


def test_build_refuses_empty_signals(tmp_path, monkeypatch):
    """The build's own spec load stops on an empty list."""
    import yaml

    from osprey.build.claude_code_resolver import load_provider_spec

    monkeypatch.setattr(resolver, "_running_in_container", lambda: False)
    (tmp_path / "config.yml").write_text(
        yaml.safe_dump(
            {"claude_code": {"provider": "anthropic", "telemetry": {**_OO_CFG, "signals": []}}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(TelemetryConfigError, match="claude_code.telemetry.signals is empty"):
        load_provider_spec(
            tmp_path, defer_unresolved_telemetry_creds=True, defer_unresolved_base_url=True
        )


def _resolve_with_signals(monkeypatch, signals) -> ClaudeCodeModelSpec:
    monkeypatch.setattr(resolver, "_running_in_container", lambda: False)
    spec = ClaudeCodeModelResolver.resolve(
        {"provider": "anthropic", "telemetry": _signals_cfg(signals)}
    )
    assert spec is not None
    return spec


def test_a_left_out_signal_overrides_a_shell_export(monkeypatch):
    """A stale shell export of a left-out signal is overwritten at launch."""
    spec = _resolve_with_signals(monkeypatch, ["metrics", "logs"])
    environ = {"OTEL_TRACES_EXPORTER": "otlp", "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1"}
    resolver.inject_provider_env(environ, spec)
    assert environ["OTEL_TRACES_EXPORTER"] == "none"
    assert environ["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] == "0"
    assert spec.detect_env_conflicts({"OTEL_TRACES_EXPORTER": "otlp"}) == {}


def test_a_left_out_signal_overrides_the_project_env_file(tmp_path, monkeypatch):
    """A ``.env`` line for a left-out signal is overwritten at launch."""
    spec = _resolve_with_signals(monkeypatch, ["metrics", "logs"])
    (tmp_path / ".env").write_text("OTEL_TRACES_EXPORTER=otlp\n", encoding="utf-8")
    environ: dict[str, str] = {}
    resolver.inject_provider_env(environ, spec, project_dir=tmp_path)
    assert environ["OTEL_TRACES_EXPORTER"] == "none"
