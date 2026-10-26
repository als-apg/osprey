"""Only the containers that send OTLP are given the record's address and a record login.

Renders every bundled service template with the old telemetry gateway still
deployed. The dispatch workers run an agent, so each carries the agent record's
address and its own record login file, which is what its exporter sends with;
no service is handed a telemetry-gateway sender login in any form.
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

from osprey.agent_record.client import URL_ENV
from osprey.deployment.telemetry_credentials import LOGIN_ENV as SENDER_LOGIN_ENV
from osprey.deployment.telemetry_credentials import SENDER_FILE_PREFIX

#: The prefix of a record login file (``.env.agent-record.login.<login>``).
RECORD_LOGIN_FILE_PREFIX = ".env.agent-record.login."

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


def test_only_emitters_get_the_record_address_and_a_record_login():
    rendered = _render_all_with_the_gateway()
    assert _EMITTER_TEMPLATES <= set(rendered)
    for name, text in rendered.items():
        services = (yaml.safe_load(text) or {}).get("services") or {}
        for service_name, service in services.items():
            env = _environment(service)
            files = _env_files(service)
            where = f"{name}:{service_name}"
            assert SENDER_LOGIN_ENV not in env, where
            assert not [f for f in files if f.startswith(SENDER_FILE_PREFIX)], where
            if name in _EMITTER_TEMPLATES:
                assert env.get(URL_ENV, "").startswith("http"), where
                logins = [f for f in files if f.startswith(RECORD_LOGIN_FILE_PREFIX)]
                assert logins == [f"{RECORD_LOGIN_FILE_PREFIX}{service_name}"], where
