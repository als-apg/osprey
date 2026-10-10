"""Unit tests for the web-terminal container-name convention.

``naming.py`` names every web-terminal container from the project name
``resolve_project_name()`` returns, and is the single Python edit point for the
container names the compose template (``docker-compose.web.yml.j2``) declares.
These tests lock in the exact string each helper emits for a resolved project
name, that the user is recovered from a compose service key and never from a
container name, and that the module stays in sync with the template lines it
mirrors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.deployment.compose_generator import resolve_project_name
from osprey.deployment.web_terminals.naming import (
    WEB_SERVICE_PREFIX,
    web_container_name,
    web_service_user,
)
from osprey.deployment.web_terminals.render import render_web_terminals

WEB_TEMPLATE = (
    Path(__file__).parents[3]
    / "src"
    / "osprey"
    / "templates"
    / "modules"
    / "web_terminals"
    / "docker-compose.web.yml.j2"
)

# The literal jinja fragments the template uses for container names and the
# per-user service key. naming.py exists to reproduce these after substitution.
TEMPLATE_USER_PATTERN = "{{ project_name }}-web-{{ svc.user }}"
TEMPLATE_SERVICE_KEY = "  web-{{ svc.user }}:"


def test_name_format():
    assert web_container_name("control-assistant", "alice") == "control-assistant-web-alice"


def test_helper_takes_the_project_name():
    """The one input is the project name, addressable as ``project``."""
    assert web_container_name(project="demo", user="alice") == "demo-web-alice"


def test_name_follows_the_resolved_project_name():
    """A config's containers carry the name ``resolve_project_name`` gives it."""
    cfg = {"project_name": "control-assistant"}
    project = resolve_project_name(cfg)
    assert web_container_name(resolve_project_name(cfg), "alice") == f"{project}-web-alice"


def test_name_follows_the_project_root_fallback():
    """With no ``project_name``, the name comes from the project directory."""
    cfg = {"project_root": "/srv/deployments/beamline-assistant"}
    assert web_container_name(resolve_project_name(cfg), "alice") == "beamline-assistant-web-alice"


def test_name_format_keeps_the_project_name_verbatim():
    assert web_container_name("uitf_assistant", "alice") == "uitf_assistant-web-alice"


@pytest.mark.parametrize("user", ["alice", "user_01", "a-b", "OPS"])
def test_user_passed_through_verbatim(user):
    """The module performs no sanitization; whatever user string it is given
    lands unchanged after the project name."""
    assert web_container_name("site", user) == f"site-web-{user}"


@pytest.mark.parametrize(
    ("service", "user"),
    [
        ("web-alice", "alice"),
        ("web-a-b", "a-b"),
        ("nginx", None),
        ("auth", None),
        ("web-", None),
        ("dispatch-worker", None),
        ("webhook", None),
    ],
)
def test_service_key_names_its_user(service, user):
    """A per-user terminal is told by its compose service key, which carries no
    deployment name: the nginx and auth siblings and every base service are
    never read as a user."""
    assert web_service_user(service) == user


def test_service_user_does_not_depend_on_the_container_name():
    """The same terminal is recognized whatever deployment named its container,
    which is what lets a container from an earlier naming scheme be found."""
    for project in ("ex", "site", "uitf_assistant"):
        name = web_container_name(project, "alice")
        assert name.endswith(f"-{WEB_SERVICE_PREFIX}alice")
    assert web_service_user(f"{WEB_SERVICE_PREFIX}alice") == "alice"


def test_template_still_carries_the_patterns_this_module_mirrors():
    """Guard against the template drifting from naming.py: the exact jinja
    fragments naming.py reproduces must still appear in the compose template."""
    text = WEB_TEMPLATE.read_text(encoding="utf-8")
    assert f"container_name: {TEMPLATE_USER_PATTERN}" in text
    assert TEMPLATE_SERVICE_KEY in text
    assert "container_name: {{ project_name }}-nginx" in text
    assert "container_name: {{ project_name }}-auth" in text


def test_no_container_name_is_spelled_on_the_facility_prefix():
    """``facility.prefix`` names no container: every ``container_name`` line is
    spelled on the compose project."""
    text = WEB_TEMPLATE.read_text(encoding="utf-8")
    names = [line.strip() for line in text.splitlines() if "container_name:" in line]
    assert names
    assert all("{{ project_name }}-" in line for line in names), names
    assert all("facility_prefix" not in line for line in names), names


def test_module_output_matches_rendered_template_pattern():
    """Substituting the template's jinja placeholders must yield exactly what
    web_container_name produces — the sync contract, checked by construction."""
    rendered = TEMPLATE_USER_PATTERN.replace("{{ project_name }}", "site").replace(
        "{{ svc.user }}", "alice"
    )
    assert rendered == web_container_name("site", "alice")


def _rendered_config(project_name: str, facility_token: str) -> dict:
    """A password-auth config, so the render emits the proxy, the sidecar and the terminals."""
    return {
        "project_name": project_name,
        "facility": {"prefix": facility_token},
        "system": {"timezone": "UTC"},
        "registry": {"url": "registry.example.org/profiles"},
        "deploy": {"host": "deploy", "fqdn": "deploy.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "users": ["alice", "bob"],
                "default_persona": "assistant",
                "personas": {"assistant": {"project": "demo-assistant"}},
                "auth": {"method": "password", "allow_insecure_http": True},
            }
        },
    }


def test_every_rendered_container_is_named_by_the_project_name():
    """Each container the compose overlay names starts with ``resolve_project_name()``
    and carries nothing from the facility token."""
    config = _rendered_config("beamline-ops", "zq")
    project = resolve_project_name(config)

    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])
    names = {key: svc["container_name"] for key, svc in compose["services"].items()}

    assert names == {
        "nginx": f"{project}-nginx",
        "auth": f"{project}-auth",
        "web-alice": web_container_name(project, "alice"),
        "web-bob": web_container_name(project, "bob"),
    }
    assert all(name.startswith(f"{project}-") for name in names.values())
    assert not any("zq" in name for name in names.values())
