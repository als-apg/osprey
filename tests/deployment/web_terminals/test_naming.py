"""Unit tests for the web-terminal container-name convention.

``naming.py`` is the single Python edit point for the container names the
compose template (``docker-compose.web.yml.j2``) declares. These tests lock in
the exact string each helper emits, that the user is recovered from a compose
service key and never from a container name, and that the module stays in sync
with the template lines it mirrors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.deployment.web_terminals.naming import (
    WEB_SERVICE_PREFIX,
    web_container_name,
    web_service_user,
)

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
