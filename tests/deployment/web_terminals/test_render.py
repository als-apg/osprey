"""Tests for multi-user web-terminal artifact rendering (osprey.deployment.web_terminals.render)."""

from __future__ import annotations

import copy

import pytest
import yaml

from osprey.deployment.web_terminals.ports import allocate_ports
from osprey.deployment.web_terminals.render import render_web_terminals

_BASE_PORTS = {"web": 9091, "artifact": 9291, "ariel": 9391, "lattice": 9491}


def _config(users: list[str], groups: list[dict] | None = None) -> dict:
    """Build a minimal-but-complete facility config exercising every field
    render_web_terminals() reads."""
    web_terminals: dict = {
        "enabled": True,
        "nginx_port": 9080,
        "web_base_port": _BASE_PORTS["web"],
        "artifact_base_port": _BASE_PORTS["artifact"],
        "ariel_base_port": _BASE_PORTS["ariel"],
        "lattice_base_port": _BASE_PORTS["lattice"],
        "users": users,
    }
    if groups is not None:
        web_terminals["landing"] = {"groups": groups}
    return {
        "facility": {
            "name": "Demo Light Source",
            "prefix": "dls",
            "timezone": "America/Los_Angeles",
        },
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {"web_terminals": web_terminals},
    }


_MULTI_USER_CONFIG = _config(["alice", "bob", "carol"])


def test_render_returns_exactly_three_artifacts() -> None:
    """render_web_terminals() produces the compose overlay, nginx fragment, and landing page."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert set(artifacts.keys()) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }


def test_render_is_deterministic() -> None:
    """Same config in -> byte-identical output out, twice."""
    # Arrange
    config_a = copy.deepcopy(_MULTI_USER_CONFIG)
    config_b = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    first = render_web_terminals(config_a)
    second = render_web_terminals(config_b)

    # Assert
    assert first == second


def test_compose_parses_as_yaml_with_one_service_and_one_volume_pair_per_user() -> None:
    """The compose overlay is valid YAML: one web-<user> service + nginx, one volume pair per user."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert set(compose["services"].keys()) == {"nginx", *(f"web-{u}" for u in users)}
    assert set(compose["volumes"].keys()) == {
        volume for u in users for volume in (f"{u}-claude-config", f"{u}-agent-data")
    }


def test_compose_service_env_has_terminal_user_and_8087_constant_per_service() -> None:
    """Each per-user service sets OSPREY_TERMINAL_USER, and the 8087 web-internal
    default is referenced (in the healthcheck commentary) once per service."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    compose_text = artifacts["docker-compose.web.yml"]
    compose = yaml.safe_load(compose_text)

    # Assert
    for user in users:
        env = compose["services"][f"web-{user}"]["environment"]
        assert f"OSPREY_TERMINAL_USER={user}" in env
    # The healthcheck commentary mentions the fixed 8087 default twice per
    # per-user service block (docker-compose.web.yml.j2's healthcheck comment).
    assert compose_text.count("8087") == 2 * len(users)


def test_compose_ports_are_four_non_colliding_families_matching_allocate_ports() -> None:
    """Every user's four ports must match allocate_ports() exactly and never collide."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    seen_ports: set[int] = set()
    for index, user in enumerate(users):
        expected = allocate_ports(_BASE_PORTS, index)
        env = compose["services"][f"web-{user}"]["environment"]
        env_map = dict(item.split("=", 1) for item in env if "=" in item)
        actual = {
            "web": int(env_map["OSPREY_WEB_PORT"]),
            "artifact": int(env_map["OSPREY_ARTIFACT_SERVER_PORT"]),
            "ariel": int(env_map["OSPREY_ARIEL_PORT"]),
            "lattice": int(env_map["OSPREY_LATTICE_DASHBOARD_PORT"]),
        }
        assert actual == expected
        for port in actual.values():
            assert port not in seen_ports, f"port {port} collides across users"
            seen_ports.add(port)


def test_nginx_fragment_has_one_routing_location_per_user() -> None:
    """nginx.conf.j2 emits one `location = /<user>` redirect block per configured user."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert "server {" in nginx_conf
    assert f"listen {config['modules']['web_terminals']['nginx_port']};" in nginx_conf
    for user in users:
        assert f"location = /{user} {{" in nginx_conf


def test_landing_renders_one_card_per_user_in_users_group() -> None:
    """The auto-populated `users` group renders one landing card per configured user."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    landing_html = artifacts["nginx/landing.html"]

    # Assert
    assert landing_html.count('class="landing-card-label"') == len(users)
    for user in users:
        assert f">{user}<" in landing_html
        assert f'href="/{user}"' in landing_html


def test_landing_zero_users_suppresses_users_group_and_shows_empty_state() -> None:
    """0-user config: no cards render, and the generic empty-state message shows."""
    # Arrange
    config = copy.deepcopy(_config([]))

    # Act
    artifacts = render_web_terminals(config)
    landing_html = artifacts["nginx/landing.html"]

    # Assert
    assert 'class="landing-card-label"' not in landing_html
    assert "No terminals or links are configured yet." in landing_html


def test_landing_single_user_renders_exactly_one_card() -> None:
    """1-user config: exactly one card renders, no empty-state message."""
    # Arrange
    config = copy.deepcopy(_config(["solo"]))

    # Act
    artifacts = render_web_terminals(config)
    landing_html = artifacts["nginx/landing.html"]

    # Assert
    assert landing_html.count('class="landing-card-label"') == 1
    assert "No terminals or links are configured yet." not in landing_html


def test_landing_escapes_html_special_characters_in_user_and_link_labels() -> None:
    """Unsafe characters in a user name or link label must be escaped, never raw."""
    # Arrange
    unsafe_user = "<script>alert('u')</script>"
    groups = [
        {"type": "users"},
        {
            "type": "links",
            "label": "Facility <Tools> & More",
            "links": [{"label": "Elog & Status", "url": "https://elog.example.org"}],
        },
    ]
    config = copy.deepcopy(_config([unsafe_user], groups=groups))

    # Act
    artifacts = render_web_terminals(config)
    landing_html = artifacts["nginx/landing.html"]

    # Assert
    assert unsafe_user not in landing_html
    assert "&lt;script&gt;" in landing_html
    assert "Facility <Tools> & More" not in landing_html
    assert "Facility &lt;Tools&gt; &amp; More" in landing_html
    assert "Elog &amp; Status" in landing_html


def test_landing_transform_renders_links_group_items() -> None:
    """A config `links` group's items must appear as landing cards with their own URLs."""
    # Arrange
    groups = [
        {"type": "users"},
        {
            "type": "links",
            "label": "Facility Tools",
            "links": [
                {"label": "Elog", "url": "https://elog.example.org"},
                {"label": "Status Page", "url": "https://status.example.org"},
            ],
        },
    ]
    config = copy.deepcopy(_config(["alice"], groups=groups))

    # Act
    artifacts = render_web_terminals(config)
    landing_html = artifacts["nginx/landing.html"]

    # Assert
    assert "Facility Tools" in landing_html
    assert ">Elog<" in landing_html
    assert 'href="https://elog.example.org"' in landing_html
    assert ">Status Page<" in landing_html
    assert 'href="https://status.example.org"' in landing_html
    # one users-group card (alice) + two links-group cards
    assert landing_html.count('class="landing-card-label"') == 3


def test_render_missing_deploy_fqdn_raises_when_users_configured() -> None:
    """landing_url can't be resolved without deploy.fqdn once at least one user exists."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    del config["deploy"]["fqdn"]

    # Act / Assert
    with pytest.raises(ValueError, match="deploy.fqdn"):
        render_web_terminals(config)


def test_render_missing_deploy_fqdn_is_fine_with_zero_users() -> None:
    """landing_url is never needed when there are no per-user services to bake it into."""
    # Arrange
    config = copy.deepcopy(_config([]))
    del config["deploy"]["fqdn"]

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert "docker-compose.web.yml" in artifacts


def test_render_missing_port_family_raises_value_error() -> None:
    """A user list that can't fully resolve allocate_ports() must fail loudly, not silently."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    del config["modules"]["web_terminals"]["lattice_base_port"]

    # Act / Assert
    with pytest.raises(ValueError, match="lattice"):
        render_web_terminals(config)


def test_removing_user_regenerates_without_their_service_route_and_volumes() -> None:
    """C8: dropping a user from users[] regenerates without their service/route/card/volume,
    while other users are unaffected. Phase-1 generation must never emit a directive that
    tears down a removed user's data — volume teardown is an explicit, documented Phase-3 step.
    """
    # Arrange — render the full roster, then regenerate with 'bob' removed
    full = render_web_terminals(_config(["alice", "bob", "carol"]))
    reduced = render_web_terminals(_config(["alice", "carol"]))

    full_compose = yaml.safe_load(full["docker-compose.web.yml"])
    reduced_compose = yaml.safe_load(reduced["docker-compose.web.yml"])

    # Assert — the removal is real (sanity: bob WAS present in the full render)
    assert "web-bob" in full_compose["services"]
    assert "bob-claude-config" in full_compose["volumes"]

    # bob is gone from every generated consumer (service, route, landing card, volumes)
    for path, content in reduced.items():
        assert "bob" not in content, f"removed user 'bob' still present in {path}"
    assert "web-bob" not in reduced_compose["services"]
    assert "bob-claude-config" not in reduced_compose.get("volumes", {})
    assert "bob-agent-data" not in reduced_compose.get("volumes", {})

    # the remaining users are untouched across all three artifacts
    for user in ("alice", "carol"):
        assert f"web-{user}" in reduced_compose["services"]
        assert f"{user}-claude-config" in reduced_compose["volumes"]
        assert f"/{user}" in reduced["nginx/nginx.conf"]

    # generation is pure artifact emission — it never issues a volume-destroying directive
    for content in reduced.values():
        assert "down -v" not in content
        assert "volume rm" not in content
        assert "volume prune" not in content
