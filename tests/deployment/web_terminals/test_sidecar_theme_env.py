"""The login page's theme and facility name cross the compose overlay.

The auth sidecar sits one hop before the terminals, and it had no way to know
either: it fell back to the framework palette and the bare wordmark while
everything behind it carried the deployment's own. The two values reach it over
the same seam every other sidecar setting uses — the rendered ``environment:``
block — under the names the per-user terminals already read, and are read back
by the sidecar's own parser here so a rename on either side fails.

The theme *id* travels, never rendered CSS: the design system stays the one
producer of palettes and the sidecar resolves the id through it.
"""

from __future__ import annotations

import yaml

from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.services.auth_sidecar.app import ENV_WEB_APP_NAME, ENV_WEB_THEME, AuthSettings


def _config(*, facility_name: str | None = "Demo Light Source", theme: str | None = None) -> dict:
    """A sidecar-bearing render config, optionally naming a facility and a theme."""
    config: dict = {
        "facility": {"prefix": "dls", "timezone": "America/Los_Angeles"},
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": ["alice", "bob"],
                # The render refuses any auth method over cleartext otherwise;
                # that gate has its own coverage and is not this file's subject.
                "auth": {"method": "password", "allow_insecure_http": True},
            }
        },
    }
    if facility_name is not None:
        config["facility"]["name"] = facility_name
    if theme is not None:
        config["web"] = {"theme": theme}
    return config


def _sidecar_env(config: dict) -> dict[str, str]:
    """The sidecar's rendered environment, as the mapping its parser reads."""
    overlay = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])
    lines = overlay["services"]["auth"]["environment"]
    return dict(line.split("=", 1) for line in lines)


def test_the_configured_theme_and_facility_name_reach_the_sidecar() -> None:
    env = _sidecar_env(_config(theme="desy-light"))

    assert env[ENV_WEB_THEME] == "desy-light"
    assert env[ENV_WEB_APP_NAME] == "Demo Light Source"


def test_the_rendered_values_round_trip_through_the_sidecars_own_parser() -> None:
    settings = AuthSettings.from_env(_sidecar_env(_config(theme="desy-light")))

    assert settings.web_theme == "desy-light"
    assert settings.web_app_name == "Demo Light Source"


def test_neither_line_is_emitted_when_the_deployment_names_neither() -> None:
    """An unset value emits no line, so the page keeps its built-in fallbacks."""
    env = _sidecar_env(_config(facility_name=None))

    assert ENV_WEB_THEME not in env
    assert ENV_WEB_APP_NAME not in env


def test_a_facility_name_carrying_yaml_punctuation_arrives_whole() -> None:
    """``: `` and `` #`` are YAML structure in a bare scalar, but not in a name.

    The compose ``- KEY=value`` list form has no per-value quoting, so the whole
    ``KEY=value`` is quoted as one scalar — the same shape the per-user
    containers use for this variable.
    """
    env = _sidecar_env(_config(facility_name="Ring: Two #2"))

    assert env[ENV_WEB_APP_NAME] == "Ring: Two #2"
