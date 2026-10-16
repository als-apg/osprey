"""Tests for the OTEL telemetry env block Claude Code is launched with.

Covers the pure ``_build_telemetry_env`` helper (on/off gating, endpoint
resolution + fail-loud, the gateway sender login, the collector bearer token,
content gates, protocol/resource passthrough), its integration into
``resolve()`` and ``load_provider_spec()``, and the conflict-detection
exemption for telemetry vars. The gateway's terminal-session attribute and the
launch sites are covered in ``test_telemetry_gateway_emit.py``.
"""

from __future__ import annotations

import base64

import pytest

from osprey.agent_runner import provider_env as resolver
from osprey.agent_runner.provider_env import (
    MANAGED_ENV_VARS,
    ClaudeCodeModelResolver,
    ClaudeCodeModelSpec,
)
from osprey.build.claude_code_telemetry import (
    _TELEMETRY_CONTENT_GATES,
    TELEMETRY_ENV_VARS,
    TELEMETRY_LOGIN_ENV,
    ObservabilityCredentialError,
    TelemetryConfigError,
    _build_telemetry_env,
    _gate_is_on,
    telemetry_auth_token_env,
)
from osprey.deployment.telemetry_config import (
    GATEWAY_HOST_ENV,
    GATEWAY_PORT_ENV,
    gateway_port,
)

GATEWAY = "http://127.0.0.1:10051"
LOGIN = "operator:s3cr3t"
_GW_CFG = {"enabled": True, "backend": "gateway"}


def _gw_env(cfg: dict | None = None, **kwargs) -> dict[str, str]:
    """The block a gateway sender with a login and a resolved address gets."""
    kwargs.setdefault("gateway_url", GATEWAY)
    kwargs.setdefault("telemetry_login", LOGIN)
    return _build_telemetry_env({**_GW_CFG, **(cfg or {})}, **kwargs)


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


# ── endpoint resolution + fail-loud ──────────────────────────────


def test_endpoint_verbatim_wins():
    env = _build_telemetry_env({"enabled": True, "endpoint": "http://explicit:9999/api/x"})
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://explicit:9999/api/x"


def test_backend_generic_uses_the_endpoint_verbatim():
    env = _build_telemetry_env(
        {"enabled": True, "backend": "generic", "endpoint": "http://explicit:9999/api/x"}
    )
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://explicit:9999/api/x"


def test_gateway_endpoint_is_the_address_the_launch_resolved():
    """The gateway names no endpoint of its own; one base URL serves every signal."""
    env = _gw_env()
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == GATEWAY
    assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in env


def test_enabled_no_endpoint_unknown_backend_raises():
    with pytest.raises(ValueError):
        _build_telemetry_env({"enabled": True, "backend": "jaeger"})


def test_enabled_no_endpoint_no_backend_raises():
    with pytest.raises(ValueError):
        _build_telemetry_env({"enabled": True})


def test_unresolved_var_in_endpoint_fails_loud():
    with pytest.raises(ValueError):
        _build_telemetry_env({"enabled": True, "endpoint": "http://${OO_HOST}:4318/v1"})


def test_unresolved_var_in_the_gateway_address_fails_loud():
    with pytest.raises(TelemetryConfigError, match="unresolved"):
        _gw_env(gateway_url="http://${GATEWAY_HOST}:4318")


# ── gateway sender login (HTTP Basic) ────────────────────────────


def test_gateway_basic_auth_header():
    headers = _gw_env()["OTEL_EXPORTER_OTLP_HEADERS"]
    assert headers.startswith("Authorization=Basic ")
    token = headers.split("Authorization=Basic ", 1)[1]
    assert base64.b64decode(token).decode() == LOGIN


@pytest.mark.parametrize("login", [None, "", "   "], ids=["absent", "empty", "blank"])
def test_gateway_missing_login_raises(login):
    with pytest.raises(ObservabilityCredentialError):
        _gw_env(telemetry_login=login)


def test_gateway_login_deferred_at_build_time_omits_header(recwarn):
    """A build never carries the login, so its absence there is the correct state.

    The header is omitted rather than encoded from nothing; telemetry itself
    stays configured, and nothing warns.
    """
    env = _gw_env(telemetry_login=None, defer_unresolved_creds=True)
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in env
    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert len(recwarn) == 0


def test_gateway_login_deferral_does_not_excuse_a_malformed_login():
    """Deferral covers an absent login, not one that is present and wrong."""
    with pytest.raises(ObservabilityCredentialError):
        _gw_env(telemetry_login="no-colon", defer_unresolved_creds=True)


def test_no_headers_when_none_configured():
    """A collector with no auth emits no HEADERS var."""
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


def test_token_env_with_a_trailing_newline_is_refused():
    cfg = {**_COLLECTOR_CFG, "auth": {"token_env": "OTLP_TOKEN\n"}}
    with pytest.raises(TelemetryConfigError, match="claude_code.telemetry.auth.token_env"):
        telemetry_auth_token_env(cfg)


def test_token_with_a_comma_is_refused_without_echoing_it():
    """A comma would split the comma-joined OTLP header list."""
    with pytest.raises(ObservabilityCredentialError) as excinfo:
        _build_telemetry_env(_COLLECTOR_CFG, auth_token="abc,X-Evil=1")
    assert "abc,X-Evil=1" not in str(excinfo.value)
    assert "OTLP_TOKEN" in str(excinfo.value)


def test_auth_on_the_gateway_backend_is_refused():
    """Each gateway sender authenticates with its own login; two sources is a fault."""
    with pytest.raises(TelemetryConfigError, match="claude_code.telemetry.auth"):
        _gw_env({"auth": {"token_env": "OTLP_TOKEN"}}, auth_token="x")


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

    from osprey.agent_runner.provider_env import load_provider_spec

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
    env = _gw_env(
        {"resource_attributes": "service.name=osprey", "content_max_length": 262144},
        terminal_session="chat-1",
    )
    env.update(_build_telemetry_env(_COLLECTOR_CFG, auth_token="t"))
    assert set(env).issubset(TELEMETRY_ENV_VARS)


# ── resolve() integration ────────────────────────────────────────


def test_resolve_env_block():
    """resolve() with a provider + enabled telemetry folds telemetry into env_block."""
    spec = ClaudeCodeModelResolver.resolve(
        {"provider": "anthropic", "telemetry": dict(_GW_CFG)},
        environ={TELEMETRY_LOGIN_ENV: LOGIN},
        gateway_url=GATEWAY,
    )
    assert spec is not None
    assert spec.env_block["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert spec.env_block["OTEL_EXPORTER_OTLP_ENDPOINT"] == GATEWAY


def test_resolve_no_telemetry_leaves_env_block_clean():
    """Absent telemetry block == disabled; no OTEL vars leak into env_block."""
    spec = ClaudeCodeModelResolver.resolve({"provider": "anthropic"})
    assert spec is not None
    assert not any(k in spec.env_block for k in TELEMETRY_ENV_VARS)


def test_resolve_telemetry_vars_not_added_to_managed_set():
    """Injecting telemetry must never mutate MANAGED_ENV_VARS."""
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


# ── load_provider_spec(): the gateway address from where the launch runs ──


def _write_gateway_project(tmp_path, extra: dict | None = None) -> None:
    import yaml

    (tmp_path / "config.yml").write_text(
        yaml.safe_dump(
            {"claude_code": {"provider": "anthropic", "telemetry": dict(_GW_CFG)}, **(extra or {})}
        ),
        encoding="utf-8",
    )


@pytest.fixture
def _host_launch(monkeypatch):
    """A launch on the host: a login, and no rendered gateway address."""
    monkeypatch.delenv(GATEWAY_HOST_ENV, raising=False)
    monkeypatch.delenv(GATEWAY_PORT_ENV, raising=False)
    monkeypatch.setenv(TELEMETRY_LOGIN_ENV, LOGIN)


@pytest.mark.usefixtures("_host_launch")
def test_a_host_launch_dials_the_published_gateway_port(tmp_path):
    from osprey.agent_runner.provider_env import load_provider_spec

    _write_gateway_project(tmp_path, {"services": {"telemetry_gateway": {"port": 15051}}})
    spec = load_provider_spec(tmp_path)
    assert spec is not None
    assert spec.env_block["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://127.0.0.1:15051"


@pytest.mark.usefixtures("_host_launch")
def test_a_host_launch_follows_the_port_base(tmp_path):
    """A moved base moves the gateway, with no ``services.telemetry_gateway.port`` key."""
    from osprey.agent_runner.provider_env import load_provider_spec

    moved = {"deployment": {"port_base": 20000}}
    _write_gateway_project(tmp_path, moved)
    spec = load_provider_spec(tmp_path)
    assert spec is not None
    assert spec.env_block["OTEL_EXPORTER_OTLP_ENDPOINT"] == (
        f"http://127.0.0.1:{gateway_port(moved)}"
    )
    assert gateway_port(moved) != gateway_port({})


@pytest.mark.usefixtures("_host_launch")
def test_a_host_launch_on_a_concrete_bind_address_dials_that_address(tmp_path):
    """A port published on one address is not reachable on loopback."""
    from osprey.agent_runner.provider_env import load_provider_spec

    _write_gateway_project(
        tmp_path,
        {
            "deployment": {"bind_address": "10.0.0.5"},
            "services": {"telemetry_gateway": {"port": 15051}},
        },
    )
    spec = load_provider_spec(tmp_path)
    assert spec is not None
    assert spec.env_block["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://10.0.0.5:15051"


@pytest.mark.usefixtures("_host_launch")
def test_a_rendered_container_dials_the_host_its_environment_names(tmp_path, monkeypatch):
    """A bridge container names the service; its port defaults to the listen port."""
    from osprey.agent_runner.provider_env import load_provider_spec

    _write_gateway_project(tmp_path, {"services": {"telemetry_gateway": {"port": 15051}}})
    monkeypatch.setenv(GATEWAY_HOST_ENV, "telemetry_gateway")
    spec = load_provider_spec(tmp_path)
    assert spec.env_block["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://telemetry_gateway:4318"
    monkeypatch.setenv(GATEWAY_PORT_ENV, "15051")
    spec = load_provider_spec(tmp_path)
    assert spec.env_block["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://telemetry_gateway:15051"


@pytest.mark.usefixtures("_host_launch")
def test_a_malformed_gateway_port_in_the_environment_is_loud(tmp_path, monkeypatch):
    from osprey.agent_runner.provider_env import load_provider_spec

    _write_gateway_project(tmp_path)
    monkeypatch.setenv(GATEWAY_HOST_ENV, "telemetry_gateway")
    monkeypatch.setenv(GATEWAY_PORT_ENV, "x")
    with pytest.raises(TelemetryConfigError, match=GATEWAY_PORT_ENV):
        load_provider_spec(tmp_path)


def test_a_host_launch_without_a_login_is_refused(tmp_path, monkeypatch):
    from osprey.agent_runner.provider_env import load_provider_spec

    monkeypatch.delenv(GATEWAY_HOST_ENV, raising=False)
    monkeypatch.delenv(TELEMETRY_LOGIN_ENV, raising=False)
    _write_gateway_project(tmp_path)
    with pytest.raises(ObservabilityCredentialError) as caught:
        load_provider_spec(tmp_path)
    assert caught.value.unresolved_vars == (TELEMETRY_LOGIN_ENV,)


# ── load_provider_spec(): the login a host launch sends under ────

_BUILT_LOGIN = "builder:filed-by-the-build"


def _built_repo(root, monkeypatch) -> object:
    """A built repo whose render records ``builder`` and whose root files its login.

    Returns the render. The launch environment carries no login of its own.
    """
    import yaml

    from osprey.deployment.telemetry_credentials import sender_file

    monkeypatch.delenv(GATEWAY_HOST_ENV, raising=False)
    monkeypatch.delenv(GATEWAY_PORT_ENV, raising=False)
    monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
    monkeypatch.delenv(TELEMETRY_LOGIN_ENV, raising=False)
    (root / "profile.yml").write_text("project_name: p\n", encoding="utf-8")
    render = root / "build"
    render.mkdir()
    (render / "config.yml").write_text(
        yaml.safe_dump(
            {
                "claude_code": {"provider": "anthropic", "telemetry": dict(_GW_CFG)},
                "deployment": {"host_identity": "builder"},
            }
        ),
        encoding="utf-8",
    )
    (root / sender_file("builder")).write_text(
        f"{TELEMETRY_LOGIN_ENV}={_BUILT_LOGIN}\n", encoding="utf-8"
    )
    return render


def _basic(login: str) -> str:
    return "Authorization=Basic " + base64.b64encode(login.encode()).decode()


def test_a_host_launch_sends_under_the_login_the_build_filed(tmp_path, monkeypatch):
    from osprey.agent_runner.provider_env import load_provider_spec

    render = _built_repo(tmp_path, monkeypatch)
    spec = load_provider_spec(render, env_dir=tmp_path)
    assert spec is not None
    assert spec.env_block["OTEL_EXPORTER_OTLP_HEADERS"] == _basic(_BUILT_LOGIN)


def test_the_agent_runner_on_the_host_sends_under_the_built_login(tmp_path, monkeypatch):
    """The runner's own environment, with nothing exported before it, authenticates."""
    from osprey.agent_runner.primitives import sdk_env

    render = _built_repo(tmp_path, monkeypatch)
    assert sdk_env(render)["OTEL_EXPORTER_OTLP_HEADERS"] == _basic(_BUILT_LOGIN)


def test_a_rendered_container_sends_under_its_own_login_only(tmp_path, monkeypatch):
    """A container's login is the one its compose file set; the repo's files are not read."""
    from osprey.agent_runner.provider_env import load_provider_spec

    render = _built_repo(tmp_path, monkeypatch)
    monkeypatch.setenv(GATEWAY_HOST_ENV, "telemetry_gateway")
    with pytest.raises(ObservabilityCredentialError):
        load_provider_spec(render, env_dir=tmp_path)
    monkeypatch.setenv(TELEMETRY_LOGIN_ENV, LOGIN)
    spec = load_provider_spec(render, env_dir=tmp_path)
    assert spec.env_block["OTEL_EXPORTER_OTLP_HEADERS"] == _basic(LOGIN)


def test_a_build_time_resolution_carries_no_login(tmp_path, monkeypatch):
    from osprey.agent_runner.provider_env import load_provider_spec

    render = _built_repo(tmp_path, monkeypatch)
    spec = load_provider_spec(render, env_dir=tmp_path, defer_unresolved_telemetry_creds=True)
    assert spec is not None
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in spec.env_block


# ── content-gate truthiness + off-host advisory ──────────────────


@pytest.mark.parametrize("env_var,cfg_key", CONTENT_GATES)
@pytest.mark.parametrize(
    "falsey", [False, "false", "False", "FALSE", "0", "no", "off", " false ", ""]
)
def test_falsey_gate_values_suppress(env_var, cfg_key, falsey):
    """bool False AND false-y strings (incl. ${VAR:-false} -> "false") zero the gate."""
    env = _gw_env({cfg_key: falsey})
    assert env[env_var] == "0"


@pytest.mark.parametrize("truthy", [True, "true", "True", "1", "yes"])
def test_truthy_gate_values_stay_on(truthy):
    env = _gw_env({"log_user_prompts": truthy})
    assert env["OTEL_LOG_USER_PROMPTS"] == "1"


def test_gate_is_on_helper():
    assert _gate_is_on(None) is True  # missing key -> on
    assert _gate_is_on(True) is True
    assert _gate_is_on(False) is False
    assert _gate_is_on("false") is False
    assert _gate_is_on("FALSE") is False
    assert _gate_is_on("true") is True


def test_warns_on_content_capture_to_a_collector_of_its_own():
    """Full content capture to a collector outside the deployment leaves the host -> warn."""
    with pytest.warns(UserWarning, match="leave the host"):
        _build_telemetry_env({"enabled": True, "endpoint": "http://collector:4318"})


def test_no_warning_for_the_gateway_backend(recwarn):
    _gw_env()
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]


def test_no_warning_when_all_content_off(recwarn):
    """A collector of its own with every content gate off must not warn."""
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


# ── telemetry-specific error type ────────────────────────────────


def test_telemetry_misconfig_raises_telemetry_config_error():
    """Telemetry faults raise TelemetryConfigError (a ValueError subclass), so a
    caller can single out a telemetry misconfig without catching every ValueError."""
    assert issubclass(TelemetryConfigError, ValueError)
    with pytest.raises(TelemetryConfigError):
        _gw_env(telemetry_login=None)
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


@pytest.mark.parametrize("login", [None, "", "  "], ids=["absent", "empty", "blank"])
def test_a_missing_login_names_the_login_variable(login):
    """The names travel on the exception, not only inside its sentence.

    A caller deciding what the operator must supply decides about a variable
    name; recovering it by parsing prose would make it hostage to the wording.
    """
    with pytest.raises(ObservabilityCredentialError) as caught:
        _gw_env(telemetry_login=login)
    assert caught.value.unresolved_vars == (TELEMETRY_LOGIN_ENV,)
    assert TELEMETRY_LOGIN_ENV in str(caught.value)


@pytest.mark.parametrize(
    "login",
    ["no-colon", ":hunter2x", "identity:", "alice:pa,ss", "alice:pa\nss"],
    ids=["no-colon", "no-identity", "no-password", "comma", "newline"],
)
def test_a_malformed_login_names_no_variable_and_never_echoes_the_value(login):
    """A login that is set but wrong is not fixed by setting a variable."""
    with pytest.raises(ObservabilityCredentialError) as caught:
        _gw_env(telemetry_login=login)
    assert caught.value.unresolved_vars == ()
    if ":" in login and login.split(":", 1)[1]:
        assert login.split(":", 1)[1] not in str(caught.value)


@pytest.mark.parametrize(
    ("telemetry", "kwargs"),
    [
        ({"enabled": True, "backend": "jaeger"}, {}),
        ({"enabled": True}, {}),
        ({"enabled": True, "endpoint": "http://${COLLECTOR_HOST}:4318"}, {}),
        (
            {"enabled": True, "backend": "gateway", "protocol": "grpc"},
            {"gateway_url": GATEWAY, "telemetry_login": LOGIN},
        ),
        ({"enabled": True, "backend": "gateway"}, {"telemetry_login": LOGIN}),
    ],
    ids=[
        "unknown-backend",
        "no-endpoint-no-backend",
        "unresolved-var",
        "grpc-to-gateway",
        "no-gateway-address",
    ],
)
def test_endpoint_faults_keep_the_general_telemetry_type(telemetry, kwargs):
    """An endpoint fault is not a credential fault, and must not borrow its type.

    A handler that offers a credential remedy would send the operator to the
    secret store for a problem that lives in the endpoint config.
    """
    with pytest.raises(TelemetryConfigError) as caught:
        _build_telemetry_env(telemetry, **kwargs)
    assert not isinstance(caught.value, ObservabilityCredentialError)


def test_existing_broad_handlers_still_catch_a_credential_fault():
    """The callers that catch TelemetryConfigError or ValueError keep working."""
    with pytest.raises(TelemetryConfigError):
        _gw_env(telemetry_login=None)
    with pytest.raises(ValueError):
        _gw_env(telemetry_login=None)


def test_a_deferred_login_is_omitted_not_raised_through_resolve(recwarn):
    """Pins the build-time catch site's reachability contract through resolve().

    ``build_cmd.py``'s render sets ``defer_unresolved_telemetry_creds=True``
    around ``load_provider_spec()``. An absent login must omit the header all
    the way through ``resolve()`` — not just at the ``_build_telemetry_env``
    leaf — because a build never carries a sender's login.
    """
    spec = ClaudeCodeModelResolver.resolve(
        {"provider": "anthropic", "telemetry": dict(_GW_CFG)},
        defer_unresolved_telemetry_creds=True,
        environ={},
        gateway_url=GATEWAY,
    )
    assert spec is not None
    assert "OTEL_EXPORTER_OTLP_HEADERS" not in spec.env_block
    assert spec.env_block["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert len(recwarn) == 0


class TestTelemetryPortIsATelemetryInput:
    """A malformed ``services.telemetry_gateway.port`` is a telemetry fault, and a
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
        "    backend: gateway\n"
        "services:\n"
        "  telemetry_gateway:\n"
        "    port: not-a-port\n"
    )

    @pytest.fixture(autouse=True)
    def _host_launch(self, monkeypatch):
        monkeypatch.delenv(GATEWAY_HOST_ENV, raising=False)
        monkeypatch.setenv(TELEMETRY_LOGIN_ENV, LOGIN)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    def test_without_telemetry_the_port_is_never_resolved(self, tmp_path):
        from osprey.agent_runner.provider_env import load_provider_spec

        (tmp_path / "config.yml").write_text(self._CONFIG)

        spec = load_provider_spec(tmp_path, include_telemetry=False)

        assert spec.provider == "anthropic"
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in spec.env_block

    def test_with_telemetry_the_port_fault_is_loud(self, tmp_path):
        from osprey.agent_runner.provider_env import load_provider_spec

        (tmp_path / "config.yml").write_text(self._CONFIG)

        with pytest.raises(TelemetryConfigError, match="services.telemetry_gateway.port"):
            load_provider_spec(tmp_path, include_telemetry=True)


# ── claude_code.telemetry.signals ─────────────────────────────────

_EXPORTERS = {
    "metrics": "OTEL_METRICS_EXPORTER",
    "logs": "OTEL_LOGS_EXPORTER",
    "traces": "OTEL_TRACES_EXPORTER",
}
_SIGNAL_KEYS = {*_EXPORTERS.values(), "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"}


def _signals_cfg(signals) -> dict:
    """An enabled gateway block with ``signals``."""
    return {**_GW_CFG, "signals": signals}


def _signals_env(signals) -> dict[str, str]:
    return _gw_env(_signals_cfg(signals))


def test_absent_signals_export_all_three():
    """Unset, ``None`` and the full list give the same block: every signal on."""
    absent = _gw_env()

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


def test_build_refuses_empty_signals(tmp_path):
    """The build's own spec load stops on an empty list."""
    import yaml

    from osprey.agent_runner.provider_env import load_provider_spec

    (tmp_path / "config.yml").write_text(
        yaml.safe_dump(
            {"claude_code": {"provider": "anthropic", "telemetry": {**_GW_CFG, "signals": []}}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(TelemetryConfigError, match="claude_code.telemetry.signals is empty"):
        load_provider_spec(
            tmp_path, defer_unresolved_telemetry_creds=True, defer_unresolved_base_url=True
        )


def _resolve_with_signals(signals) -> ClaudeCodeModelSpec:
    spec = ClaudeCodeModelResolver.resolve(
        {"provider": "anthropic", "telemetry": _signals_cfg(signals)},
        environ={TELEMETRY_LOGIN_ENV: LOGIN},
        gateway_url=GATEWAY,
    )
    assert spec is not None
    return spec


def test_a_left_out_signal_overrides_a_shell_export():
    """A stale shell export of a left-out signal is overwritten at launch."""
    spec = _resolve_with_signals(["metrics", "logs"])
    environ = {"OTEL_TRACES_EXPORTER": "otlp", "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1"}
    resolver.inject_provider_env(environ, spec)
    assert environ["OTEL_TRACES_EXPORTER"] == "none"
    assert environ["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] == "0"
    assert spec.detect_env_conflicts({"OTEL_TRACES_EXPORTER": "otlp"}) == {}


def test_a_left_out_signal_overrides_the_project_env_file(tmp_path):
    """A ``.env`` line for a left-out signal is overwritten at launch."""
    spec = _resolve_with_signals(["metrics", "logs"])
    (tmp_path / ".env").write_text("OTEL_TRACES_EXPORTER=otlp\n", encoding="utf-8")
    environ: dict[str, str] = {}
    resolver.inject_provider_env(environ, spec, project_dir=tmp_path)
    assert environ["OTEL_TRACES_EXPORTER"] == "none"
