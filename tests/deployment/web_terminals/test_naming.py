"""Unit tests for the web-terminal container-name convention.

``naming.py`` is the single Python edit point for the container names the
compose template (``docker-compose.web.yml.j2``) declares. These tests lock in
three things: the exact string format each helper emits, the composition
invariant that the per-user name is prefix-addressable (orphan discovery relies
on ``startswith(prefix)``), and that the module stays in sync with the template
line it mirrors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.deployment.web_terminals.naming import (
    web_container_name,
    web_container_prefix,
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

# The literal jinja fragment the template uses for a per-user terminal's
# container_name. naming.py exists to reproduce exactly this after substitution.
TEMPLATE_USER_PATTERN = "{{ facility_prefix }}-web-{{ svc.user }}"


def test_prefix_format():
    assert web_container_prefix("als") == "als-web-"


def test_name_format():
    assert web_container_name("als", "alice") == "als-web-alice"


def test_name_is_prefix_plus_user():
    """The full name is exactly the prefix concatenated with the user — the
    property that lets orphan discovery reconstruct/match names from the prefix."""
    assert web_container_name("als", "bob") == web_container_prefix("als") + "bob"


def test_name_starts_with_prefix():
    """Orphan discovery in lifecycle.py prefix-matches on web_container_prefix;
    every real per-user name MUST satisfy that match."""
    prefix = web_container_prefix("demo")
    assert web_container_name("demo", "carol").startswith(prefix)


def test_nginx_sibling_is_not_swept_by_web_prefix():
    """The reverse proxy is named ``<prefix>-nginx`` (a different pattern), so it
    must NOT prefix-match the web-terminal prefix — otherwise orphan discovery
    would decommission the shared nginx as a stray user terminal."""
    nginx_name = "als-nginx"
    assert not nginx_name.startswith(web_container_prefix("als"))


def test_distinct_facilities_do_not_collide():
    """Two facilities whose prefixes are not prefixes of one another get
    disjoint container namespaces — a name from one never prefix-matches the
    other's discovery prefix."""
    assert not web_container_name("als2", "alice").startswith(web_container_prefix("als1"))
    assert not web_container_name("als1", "alice").startswith(web_container_prefix("als2"))


def test_empty_user_yields_bare_prefix():
    """Documents the boundary: an empty user degenerates to just the prefix
    (no sanitization or fallback happens in this module)."""
    assert web_container_name("als", "") == web_container_prefix("als")


@pytest.mark.parametrize("user", ["alice", "user_01", "a-b", "OPS"])
def test_user_passed_through_verbatim(user):
    """The module performs no sanitization; whatever user string it is given
    lands unchanged after the prefix."""
    assert web_container_name("als", user) == f"als-web-{user}"


def test_template_still_carries_the_pattern_this_module_mirrors():
    """Guard against the template drifting from naming.py: the exact jinja
    fragment naming.py reproduces must still appear in the compose template."""
    text = WEB_TEMPLATE.read_text(encoding="utf-8")
    assert f"container_name: {TEMPLATE_USER_PATTERN}" in text


def test_module_output_matches_rendered_template_pattern():
    """Substituting the template's jinja placeholders must yield exactly what
    web_container_name produces — the sync contract, checked by construction."""
    rendered = TEMPLATE_USER_PATTERN.replace("{{ facility_prefix }}", "als").replace(
        "{{ svc.user }}", "alice"
    )
    assert rendered == web_container_name("als", "alice")
