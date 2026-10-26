"""What the web-terminal render hands each container about the telemetry gateway.

Terminals run under ``network_mode: host``, so each is told where the gateway
PUBLISHES — the interface ``deployment.bind_address`` names (loopback for a
wildcard), on the published port. Containers that send no OTLP (nginx) are told
nothing, and no container is handed a gateway login file.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from osprey.deployment.telemetry_config import (
    GATEWAY_HOST_ENV,
    GATEWAY_PORT_ENV,
    gateway_port,
    runtime_gateway_url,
)
from osprey.deployment.telemetry_credentials import SENDER_FILE_PREFIX
from osprey.deployment.web_terminals.render import render_web_terminals
from tests._renders import tier_render

_CONFIG: dict = {
    "project_name": "demo_controls",
    "registry": {"url": "registry.example.org/demo/profiles"},
    "deploy": {"host": "demo-deploy", "fqdn": "demo-deploy.example.org"},
    "modules": {
        "web_terminals": {
            "enabled": True,
            "users": ["alice", "bob"],
            "default_tier": "assistant",
            "renders": {"assistant": tier_render("assistant", "demo_controls-assistant")},
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
        # A reader inside resolves exactly that address.
        assert runtime_gateway_url(config, env) == (
            f"http://{expected_host}:{gateway_port(config)}"
        ), name


def test_no_container_is_handed_a_gateway_login_file():
    """No terminal sends to the gateway with a login of its own."""
    compose, _ = _compose()
    for name, service in compose["services"].items():
        assert not [f for f in _env_files(service) if f.startswith(SENDER_FILE_PREFIX)], name


def test_no_endpoint_reaches_a_container_that_sends_nothing():
    """nginx emits no OTLP: no gateway address."""
    compose, _ = _compose()
    others = {n: s for n, s in compose["services"].items() if not n.startswith("web-")}
    assert others, "the render produced no non-terminal service"
    for name, service in others.items():
        env = _environment(service)
        assert GATEWAY_HOST_ENV not in env, name
        assert GATEWAY_PORT_ENV not in env, name


def test_without_a_gateway_nothing_is_stated():
    compose, _ = _compose(deployed_services=[])
    for name, service in compose["services"].items():
        env = _environment(service)
        assert GATEWAY_HOST_ENV not in env, name
        assert not [f for f in _env_files(service) if f.startswith(SENDER_FILE_PREFIX)], name
