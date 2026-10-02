"""Unit tests for the web-terminal container-name convention.

``naming.py`` names every web-terminal container from the project name
``resolve_project_name()`` returns. These tests lock in the exact string each
helper emits for a resolved project name, and the composition invariant that the
per-user name is prefix-addressable (orphan discovery relies on
``startswith(prefix)``).
"""

from __future__ import annotations

import pytest
import yaml

from osprey.deployment.compose_generator import resolve_project_name
from osprey.deployment.web_terminals.naming import (
    web_container_name,
    web_container_prefix,
)
from osprey.deployment.web_terminals.render import render_web_terminals


def test_prefix_format():
    assert web_container_prefix("control-assistant") == "control-assistant-web-"


def test_name_format():
    assert web_container_name("control-assistant", "alice") == "control-assistant-web-alice"


def test_helpers_take_the_project_name():
    """The one input is the project name, addressable as ``project``."""
    assert web_container_prefix(project="demo") == "demo-web-"
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


def test_name_is_prefix_plus_user():
    """The full name is exactly the prefix concatenated with the user — the
    property that lets orphan discovery reconstruct/match names from the prefix."""
    assert web_container_name("demo", "bob") == web_container_prefix("demo") + "bob"


def test_name_starts_with_prefix():
    """Orphan discovery in lifecycle.py prefix-matches on web_container_prefix;
    every real per-user name MUST satisfy that match."""
    prefix = web_container_prefix("demo")
    assert web_container_name("demo", "carol").startswith(prefix)


def test_nginx_sibling_is_not_swept_by_web_prefix():
    """The reverse proxy is named ``<project>-nginx`` (a different pattern), so it
    must NOT prefix-match the web-terminal prefix — otherwise orphan discovery
    would decommission the shared nginx as a stray user terminal."""
    nginx_name = "demo-nginx"
    assert not nginx_name.startswith(web_container_prefix("demo"))


def test_distinct_projects_do_not_collide():
    """Two projects whose names are not prefixes of one another get disjoint
    container namespaces — a name from one never prefix-matches the other's
    discovery prefix."""
    assert not web_container_name("demo2", "alice").startswith(web_container_prefix("demo1"))
    assert not web_container_name("demo1", "alice").startswith(web_container_prefix("demo2"))


def test_empty_user_yields_bare_prefix():
    """Documents the boundary: an empty user degenerates to just the prefix
    (no sanitization or fallback happens in this module)."""
    assert web_container_name("demo", "") == web_container_prefix("demo")


@pytest.mark.parametrize("user", ["alice", "user_01", "a-b", "OPS"])
def test_user_passed_through_verbatim(user):
    """The module performs no sanitization; whatever user string it is given
    lands unchanged after the prefix."""
    assert web_container_name("demo", user) == f"demo-web-{user}"


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
