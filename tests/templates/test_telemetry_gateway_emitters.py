"""Only the containers that send OTLP are told where the gateway is, or given a login.

Renders every bundled service template with the gateway deployed. The dispatch
workers emit and so carry the gateway address and their own login file; the
archive, the stores and every fixed service send nothing and get neither.
"""

from __future__ import annotations

import yaml
from test_render_defaults_golden import (
    TEMPLATES,
    _pinned_context,
    _probe_repo,
    _raw_default_config,
    _render_templates,
)

from osprey.build.claude_code_telemetry import TELEMETRY_LOGIN_ENV
from osprey.deployment.telemetry_config import GATEWAY_HOST_ENV, GATEWAY_PORT_ENV
from osprey.deployment.telemetry_credentials import SENDER_FILE_PREFIX

#: The service templates whose containers run an agent that emits OTLP.
_EMITTER_TEMPLATES = frozenset({"dispatch_worker.yml"})


def _render_all_with_the_gateway() -> dict[str, str]:
    with _probe_repo() as repo_root:
        config = _raw_default_config(repo_root)
        deployed = sorted({*config.get("deployed_services", []), "telemetry_gateway"})
        return _render_templates(
            _pinned_context({**config, "deployed_services": deployed}), TEMPLATES
        )


def _environment(service: dict) -> dict[str, str]:
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return {str(k): str(v) for k, v in env.items()}
    return dict(str(item).split("=", 1) for item in env if "=" in str(item))


def _env_files(service: dict) -> list[str]:
    files = service.get("env_file") or []
    return [files] if isinstance(files, str) else [str(f) for f in files]


def test_only_emitters_get_a_gateway_address_or_a_login():
    rendered = _render_all_with_the_gateway()
    assert _EMITTER_TEMPLATES <= set(rendered)
    for name, text in rendered.items():
        services = (yaml.safe_load(text) or {}).get("services") or {}
        for service_name, service in services.items():
            env = _environment(service)
            logins = [f for f in _env_files(service) if f.startswith(SENDER_FILE_PREFIX)]
            where = f"{name}:{service_name}"
            assert TELEMETRY_LOGIN_ENV not in env, where
            if name in _EMITTER_TEMPLATES:
                assert GATEWAY_HOST_ENV in env and GATEWAY_PORT_ENV in env, where
                assert logins == [f"{SENDER_FILE_PREFIX}{service_name}"], where
            else:
                assert GATEWAY_HOST_ENV not in env, where
                assert GATEWAY_PORT_ENV not in env, where
                assert logins == [], where
