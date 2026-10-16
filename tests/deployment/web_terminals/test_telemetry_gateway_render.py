"""What the web-terminal render hands each container about the telemetry gateway.

Terminals run under ``network_mode: host``, so each is told where the gateway
PUBLISHES — the interface ``deployment.bind_address`` names (loopback for a
wildcard), on the published port — and is handed its own login file by
``env_file``. Containers that send no OTLP (nginx) get neither, and the shared
``.env.users`` carrier never names a login.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from osprey.build.claude_code_telemetry import TELEMETRY_LOGIN_ENV, _resolve_telemetry_endpoint
from osprey.deployment.telemetry_config import (
    GATEWAY_HOST_ENV,
    GATEWAY_PORT_ENV,
    gateway_port,
    runtime_gateway_url,
)
from osprey.deployment.telemetry_credentials import SENDER_FILE_PREFIX
from osprey.deployment.web_terminals.render import render_web_terminals

_CONFIG: dict = {
    "project_name": "demo_controls",
    "registry": {"url": "registry.example.org/demo/profiles"},
    "deploy": {"host": "demo-deploy", "fqdn": "demo-deploy.example.org"},
    "modules": {
        "web_terminals": {
            "enabled": True,
            "users": ["alice", "bob"],
            "default_persona": "assistant",
            "personas": {"assistant": {"project": "demo_controls-assistant"}},
        }
    },
    "deployed_services": ["telemetry_store", "telemetry_gateway"],
}


def _compose(**overrides) -> dict:
    config = copy.deepcopy(_CONFIG)
    config.update(overrides)
    artifacts = render_web_terminals(config, facility_name="Demo Facility")
    return yaml.safe_load(artifacts["docker-compose.web.yml"]), config


def _environment(service: dict) -> dict[str, str]:
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return {str(k): str(v) for k, v in env.items()}
    return dict(item.split("=", 1) for item in env if "=" in item)


def _env_files(service: dict) -> list[str]:
    files = service.get("env_file") or []
    return [files] if isinstance(files, str) else list(files)


def _terminals(compose: dict) -> dict[str, dict]:
    return {name: svc for name, svc in compose["services"].items() if name.startswith("web-")}


@pytest.mark.parametrize(
    ("deployment", "expected_host"),
    [
        ({}, "127.0.0.1"),
        ({"bind_address": "0.0.0.0"}, "127.0.0.1"),
        ({"bind_address": "10.0.0.5"}, "10.0.0.5"),
    ],
    ids=["default-bind", "wildcard-bind", "concrete-bind"],
)
def test_a_terminal_dials_the_gateway_where_it_publishes(deployment, expected_host):
    """A port published on one interface does not answer on loopback."""
    compose, config = _compose(deployment=deployment)
    terminals = _terminals(compose)
    assert terminals, "the render produced no terminal"
    for name, service in terminals.items():
        assert service["network_mode"] == "host", name
        env = _environment(service)
        assert env[GATEWAY_HOST_ENV] == expected_host, name
        assert env[GATEWAY_PORT_ENV] == str(gateway_port(config)), name
        # The emitter inside resolves exactly that address.
        endpoint = _resolve_telemetry_endpoint(
            {"enabled": True, "backend": "gateway"},
            gateway_url=runtime_gateway_url(config, env),
        )
        assert endpoint == f"http://{expected_host}:{gateway_port(config)}", name


def test_each_terminal_is_handed_its_own_login_file():
    compose, _ = _compose()
    for name, service in _terminals(compose).items():
        files = _env_files(service)
        logins = [f for f in files if f.startswith(SENDER_FILE_PREFIX)]
        assert len(logins) == 1, (name, files)
        assert ".env.users" in files, name


def test_no_endpoint_or_login_reaches_a_container_that_sends_nothing():
    """nginx emits no OTLP: no gateway address, no login file, no login."""
    compose, _ = _compose()
    others = {n: s for n, s in compose["services"].items() if not n.startswith("web-")}
    assert others, "the render produced no non-terminal service"
    for name, service in others.items():
        env = _environment(service)
        assert GATEWAY_HOST_ENV not in env, name
        assert GATEWAY_PORT_ENV not in env, name
        assert TELEMETRY_LOGIN_ENV not in env, name
        assert not [f for f in _env_files(service) if f.startswith(SENDER_FILE_PREFIX)], name


def test_the_login_is_never_interpolated_into_a_container():
    """The login reaches a terminal by its own file only, never ``${VAR}``."""
    compose, _ = _compose()
    for name, service in compose["services"].items():
        assert TELEMETRY_LOGIN_ENV not in _environment(service), name


def test_without_a_gateway_nothing_is_stated():
    compose, _ = _compose(deployed_services=[])
    for name, service in compose["services"].items():
        env = _environment(service)
        assert GATEWAY_HOST_ENV not in env, name
        assert not [f for f in _env_files(service) if f.startswith(SENDER_FILE_PREFIX)], name
