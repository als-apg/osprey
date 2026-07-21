"""Tests for multi-user web-terminal artifact rendering (osprey.deployment.web_terminals.render)."""

from __future__ import annotations

import copy
import re

import pytest
import yaml

from osprey.deployment.web_terminals.ports import (
    PANEL_ENV_VARS,
    allocate_ports,
    base_ports_from_config,
)
from osprey.deployment.web_terminals.render import _auth_tls_context, render_web_terminals

# The four classic config-set families; the effective per-family base set the
# render actually allocates from also carries every registry default
# (channel_finder, okf, ...) — resolved once here so `allocate_ports(_BASE_PORTS,
# i)` calls throughout this module see the same full set the render does.
_CONFIGURED_BASE_PORTS = {"web": 9091, "artifact": 9291, "ariel": 9391, "lattice": 9491}
_BASE_PORTS = base_ports_from_config(
    {f"{family}_base_port": port for family, port in _CONFIGURED_BASE_PORTS.items()}
)


def _config(users: list[str], groups: list[dict] | None = None) -> dict:
    """Build a minimal-but-complete facility config exercising every field
    render_web_terminals() reads."""
    web_terminals: dict = {
        "enabled": True,
        "nginx_port": 9080,
        "web_base_port": _CONFIGURED_BASE_PORTS["web"],
        "artifact_base_port": _CONFIGURED_BASE_PORTS["artifact"],
        "ariel_base_port": _CONFIGURED_BASE_PORTS["ariel"],
        "lattice_base_port": _CONFIGURED_BASE_PORTS["lattice"],
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


def test_nginx_image_defaults_when_config_omits_it() -> None:
    """With no `nginx_image` in config, the nginx service uses the public default tag."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    assert "nginx_image" not in config["modules"]["web_terminals"]

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert compose["services"]["nginx"]["image"] == "nginx:1.27-alpine"


def test_nginx_image_custom_value_lands_on_the_nginx_service() -> None:
    """A configured `nginx_image` overrides the nginx service image; hosts that pull
    only from a private mirror point it at their own registry."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    custom = "registry.example.com:5050/mirrors/nginx:1.27-alpine"
    config["modules"]["web_terminals"]["nginx_image"] = custom

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert compose["services"]["nginx"]["image"] == custom


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


def test_compose_ports_are_non_colliding_families_matching_allocate_ports() -> None:
    """Every user's ports — one per derived family — must match allocate_ports()
    exactly and never collide. Iterates PANEL_ENV_VARS (registry-derived), so a
    newly registered companion server is asserted here without editing this test."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]
    effective_base_ports = base_ports_from_config(config["modules"]["web_terminals"])

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    seen_ports: set[int] = set()
    for index, user in enumerate(users):
        expected = allocate_ports(effective_base_ports, index)
        env = compose["services"][f"web-{user}"]["environment"]
        env_map = dict(item.split("=", 1) for item in env if "=" in item)
        actual = {"web": int(env_map["OSPREY_WEB_PORT"])}
        for family, env_var in PANEL_ENV_VARS.items():
            actual[family] = int(env_map[env_var])
        assert actual == expected
        for port in actual.values():
            assert port not in seen_ports, f"port {port} collides across users"
            seen_ports.add(port)


def test_per_user_services_bind_loopback_not_0_0_0_0() -> None:
    """C3: each per-user service's four app processes bind 127.0.0.1 (loopback),
    never 0.0.0.0 — nginx (bind-nginx-reverse-proxy) becomes the only off-host
    path. `network_mode: host` is retained (EPICS CA UDP broadcast)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    compose_text = artifacts["docker-compose.web.yml"]
    compose = yaml.safe_load(compose_text)

    # Assert
    assert "0.0.0.0" not in compose_text
    assert compose["services"]["nginx"]["network_mode"] == "host"
    for user in users:
        service = compose["services"][f"web-{user}"]
        assert service["network_mode"] == "host"
        env = dict(item.split("=", 1) for item in service["environment"] if "=" in item)
        assert env["OSPREY_TERMINAL_BIND_HOST"] == "127.0.0.1"


def test_nginx_fragment_has_one_routing_location_per_user() -> None:
    """nginx.conf.j2 emits one `location /u/<user>/` proxy block per configured user."""
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
        assert f"location /u/{user}/ {{" in nginx_conf


def test_nginx_fragment_has_proxy_pass_to_each_users_loopback_web_upstream() -> None:
    """Each `location /u/<user>/` block proxy_passes to that user's own loopback
    `web` port (bind-per-user-apps-loopback) — not a redirect to a distinct origin."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert "return 302" not in nginx_conf  # Phase-1 redirect scheme is gone
    for index in range(len(users)):
        expected_port = allocate_ports(_BASE_PORTS, index)["web"]
        assert f"proxy_pass http://127.0.0.1:{expected_port}/;" in nginx_conf


def test_nginx_fragment_has_websocket_upgrade_machinery() -> None:
    """The `/ws/terminal` and panel WebSocket handshakes need proxy_http_version 1.1
    plus the Upgrade/Connection headers, backed by a `map $http_upgrade
    $connection_upgrade` directive declared ONCE at http/top-level (NOT nested
    inside `server{}` — a `map` directive there would fail `nginx -t`)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert nginx_conf.count("map $http_upgrade $connection_upgrade") == 1
    map_index = nginx_conf.index("map $http_upgrade $connection_upgrade")
    server_index = nginx_conf.index("server {")
    assert map_index < server_index, "map{} must sit at http/top-level, before server{}"

    assert "proxy_http_version 1.1;" in nginx_conf
    assert "proxy_set_header Upgrade $http_upgrade;" in nginx_conf
    assert "proxy_set_header Connection $connection_upgrade;" in nginx_conf


def test_nginx_fragment_proxy_disables_buffering_and_raises_read_timeout_for_sse() -> None:
    """Now that every request under `/u/<user>/` genuinely flows through nginx
    (unlike Phase-1's redirect, which never put nginx in the data path), nginx's
    DEFAULT proxy buffering and 60s read timeout would apply to the app's
    heartbeat-less Server-Sent-Events streams (`/api/files/events`, chat SSE) too
    — batching/delaying `data:` lines behind nginx's buffer and tearing down an
    idle SSE connection every 60s. Each proxied `/u/<user>/` location must
    disable buffering and raise the read timeout well above that default.
    WebSocket traffic is unaffected (it bypasses proxy buffering once the
    Upgrade handshake completes), so this doesn't undo the map/upgrade-header
    machinery covered by the sibling test above."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert — one occurrence per proxied user location (not a stray global
    # setting), and the timeout is meaningfully raised, not left at nginx's
    # 60s default.
    assert nginx_conf.count("proxy_buffering off;") == len(users)
    timeout_matches = re.findall(r"proxy_read_timeout (\d+)s;", nginx_conf)
    assert len(timeout_matches) == len(users)
    for value in timeout_matches:
        assert int(value) > 60, "proxy_read_timeout must be raised above nginx's 60s default"


def test_nginx_fragment_has_trailing_slash_redirect_per_user() -> None:
    """A no-trailing-slash `/u/<user>` bookmark 301-redirects into `/u/<user>/`
    rather than silently falling through to the landing page at `/`."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    for user in users:
        assert f"location = /u/{user} {{" in nginx_conf
        assert f"return 301 /u/{user}/;" in nginx_conf


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
        assert f'href="/u/{user}/"' in landing_html


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


def test_render_missing_web_base_port_raises_value_error() -> None:
    """A user list that can't fully resolve allocate_ports() must fail loudly, not
    silently. Only `web` can be missing — every companion family carries a
    registry default (see test_render_missing_companion_base_port_uses_default)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    del config["modules"]["web_terminals"]["web_base_port"]

    # Act / Assert
    with pytest.raises(ValueError, match="web"):
        render_web_terminals(config)


def test_render_missing_companion_base_port_uses_registry_default() -> None:
    """A config omitting a companion family's base port (e.g. written before that
    panel existed) must render with the registry default, not fail — the
    zero-migration guarantee that keeps feature parity from breaking old configs."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    del config["modules"]["web_terminals"]["lattice_base_port"]

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert — lattice family renders from its registry default for every user.
    from osprey.registry.web import FRAMEWORK_WEB_SERVERS

    default_base = FRAMEWORK_WEB_SERVERS["lattice_dashboard"].multi_user_base_port
    for index, user in enumerate(["alice", "bob", "carol"]):
        env = compose["services"][f"web-{user}"]["environment"]
        assert f"OSPREY_LATTICE_DASHBOARD_PORT={default_base + index}" in env


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


def test_object_form_users_render_with_explicit_index() -> None:
    """Object-form users (`{"name": ..., "index": ...}`) drive port allocation from
    their own `index` field, not their position in the list."""
    # Arrange
    config = copy.deepcopy(_config([{"name": "alice", "index": 5}]))

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    expected = allocate_ports(_BASE_PORTS, 5)
    env = dict(
        item.split("=", 1)
        for item in compose["services"]["web-alice"]["environment"]
        if "=" in item
    )
    assert int(env["OSPREY_WEB_PORT"]) == expected["web"]
    assert int(env["OSPREY_ARTIFACT_SERVER_PORT"]) == expected["artifact"]


def test_mixed_object_and_bare_users_render_together() -> None:
    """A roster mixing object-form and legacy bare-string entries renders both:
    the object-form entry uses its explicit index, the bare entry falls back to
    its position in the list."""
    # Arrange
    config = copy.deepcopy(_config([{"name": "alice", "index": 5}, "bob"]))

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])
    landing_html = artifacts["nginx/landing.html"]

    # Assert
    assert set(compose["services"].keys()) == {"nginx", "web-alice", "web-bob"}
    alice_env = dict(
        item.split("=", 1)
        for item in compose["services"]["web-alice"]["environment"]
        if "=" in item
    )
    bob_env = dict(
        item.split("=", 1) for item in compose["services"]["web-bob"]["environment"] if "=" in item
    )
    assert int(alice_env["OSPREY_WEB_PORT"]) == allocate_ports(_BASE_PORTS, 5)["web"]
    # 'bob' is the second list entry (position 1) -> falls back to positional index 1.
    assert int(bob_env["OSPREY_WEB_PORT"]) == allocate_ports(_BASE_PORTS, 1)["web"]
    assert ">alice<" in landing_html
    assert ">bob<" in landing_html


def test_legacy_bare_users_render_identically_to_before() -> None:
    """A bare-string-only roster is unaffected by the object-form parsing path:
    ports and landing cards still come from positional index, byte-for-byte the
    same as the pre-existing behavior covered elsewhere in this file."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    for index, user in enumerate(("alice", "bob", "carol")):
        expected = allocate_ports(_BASE_PORTS, index)
        env = dict(
            item.split("=", 1)
            for item in compose["services"][f"web-{user}"]["environment"]
            if "=" in item
        )
        assert int(env["OSPREY_WEB_PORT"]) == expected["web"]


def test_malformed_user_entries_are_dropped() -> None:
    """A non-string, non-well-formed-object entry is dropped rather than raising
    (well-formedness is lint's job, not render's) -- mirrors the prior silent-drop
    behavior for non-string entries."""
    # Arrange
    config = copy.deepcopy(
        _config([{"name": "alice", "index": 0}, {"name": "no-index"}, 42, None, "bob"])
    )

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert set(compose["services"].keys()) == {"nginx", "web-alice", "web-bob"}
    # 'bob' is the 5th raw list entry (position 4). A bare entry keeps its RAW
    # position rather than compacting over the dropped junk ahead of it -- pinned
    # here so the semantic is locked (raw position also prevents a bare fallback
    # from colliding with an object entry's explicit index in a mixed roster).
    bob_env = dict(
        item.split("=", 1) for item in compose["services"]["web-bob"]["environment"] if "=" in item
    )
    assert int(bob_env["OSPREY_WEB_PORT"]) == allocate_ports(_BASE_PORTS, 4)["web"]


# ---------------------------------------------------------------------------
# display_name -> OSPREY_WEB_APP_NAME (per-user window/tab title seam)
# ---------------------------------------------------------------------------


def _service_env(compose: dict, service: str) -> dict[str, str]:
    """Parse a compose service's `- KEY=value` environment list into a dict."""
    return dict(
        item.split("=", 1) for item in compose["services"][service]["environment"] if "=" in item
    )


def test_display_name_emits_app_name_env_line_only_for_the_user_that_sets_it() -> None:
    """A user's `display_name` renders an `OSPREY_WEB_APP_NAME` env line for that
    service; a user without one omits the line entirely (app falls back to config
    web.app_name)."""
    # Arrange
    config = copy.deepcopy(
        _config([{"name": "alice", "index": 0, "display_name": "Operations"}, "bob"])
    )

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert _service_env(compose, "web-alice")["OSPREY_WEB_APP_NAME"] == "Operations"
    assert "OSPREY_WEB_APP_NAME" not in _service_env(compose, "web-bob")


def test_no_display_name_anywhere_emits_no_app_name_env_line() -> None:
    """The common (bare-string) roster emits no OSPREY_WEB_APP_NAME line at all —
    the seam is inert until a user opts in."""
    # Act
    artifacts = render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert "OSPREY_WEB_APP_NAME" not in artifacts["docker-compose.web.yml"]


def test_display_name_with_spaces_and_colon_is_yaml_quoted_and_round_trips() -> None:
    """A `display_name` containing spaces and a `": "` (which would derail an
    unquoted compose `- KEY=value` scalar into a mapping) is safely quoted so the
    parsed env value is the exact original string."""
    # Arrange
    tricky = "Operations: Control Room"
    config = copy.deepcopy(_config([{"name": "alice", "index": 0, "display_name": tricky}]))

    # Act
    artifacts = render_web_terminals(config)
    compose_text = artifacts["docker-compose.web.yml"]
    compose = yaml.safe_load(compose_text)

    # Assert — the whole KEY=value is a single double-quoted YAML scalar, and it
    # parses back to the exact display_name (no embedded quotes, no split mapping).
    assert '- "OSPREY_WEB_APP_NAME=Operations: Control Room"' in compose_text
    assert _service_env(compose, "web-alice")["OSPREY_WEB_APP_NAME"] == tricky


def test_display_name_with_double_quote_is_escaped_and_round_trips() -> None:
    """A `display_name` containing a double quote is escaped inside the quoted YAML
    scalar rather than prematurely closing it."""
    # Arrange
    tricky = 'The "Main" Console'
    config = copy.deepcopy(_config([{"name": "alice", "index": 0, "display_name": tricky}]))

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert _service_env(compose, "web-alice")["OSPREY_WEB_APP_NAME"] == tricky


def test_catalog_present_all_users_on_default_persona_is_byte_identical_image_and_mount() -> None:
    """A `personas` catalog can exist without moving any user off the default persona
    (no roster entry sets `persona:`, so every entry falls back to `default_persona`).
    resolve_personas()'s registry-mode default-persona branch must still produce the
    SAME unsuffixed `<registry_url>/web-terminal:latest` image and the SAME
    `/app/<facility_prefix>-assistant` agent-data mount root a no-catalog config
    produces — introducing a catalog is zero-migration until a user is actually
    reassigned to a non-default persona."""
    # Arrange
    baseline_config = copy.deepcopy(_MULTI_USER_CONFIG)
    catalog_config = copy.deepcopy(_MULTI_USER_CONFIG)
    catalog_config["modules"]["web_terminals"]["default_persona"] = "assistant"
    catalog_config["modules"]["web_terminals"]["personas"] = {
        "assistant": {
            "project": "dls-assistant",
            "project_path": "../dls-assistant",
            "build_profile": "profiles/assistant.yml",
        },
    }
    users = baseline_config["modules"]["web_terminals"]["users"]

    # Act
    baseline = render_web_terminals(baseline_config)
    catalog = render_web_terminals(catalog_config)
    baseline_compose = yaml.safe_load(baseline["docker-compose.web.yml"])
    catalog_compose = yaml.safe_load(catalog["docker-compose.web.yml"])

    # Assert
    for user in users:
        baseline_svc = baseline_compose["services"][f"web-{user}"]
        catalog_svc = catalog_compose["services"][f"web-{user}"]
        assert catalog_svc["image"] == baseline_svc["image"]
        assert catalog_svc["volumes"] == baseline_svc["volumes"]
        assert (
            catalog_svc["image"]
            == "git.dls.example.org:5050/physics/production/dls-profiles/web-terminal:latest"
        )
        assert f"{user}-agent-data:/app/dls-assistant/_agent_data" in catalog_svc["volumes"]


def test_persona_extra_mounts_render_as_extra_per_user_volume_lines() -> None:
    """A persona's `extra_mounts` render as additional `volumes:` entries on every
    user of that persona, after the two default (claude-config, agent-data) mounts."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    web_terminals = config["modules"]["web_terminals"]
    web_terminals["default_persona"] = "assistant"
    web_terminals["personas"] = {
        "assistant": {
            "project": "dls-assistant",
            "project_path": "../dls-assistant",
            "build_profile": "profiles/assistant.yml",
        },
        "gui": {
            "project": "dls-gui",
            "project_path": "../dls-gui",
            "build_profile": "profiles/gui.yml",
            "extra_mounts": ["/opt/site-data:/app/site-data:ro", "shared-cache:/app/cache"],
        },
    }
    web_terminals["users"] = [
        {"name": "alice", "index": 0},  # default persona, no extra_mounts
        {"name": "bob", "index": 1, "persona": "gui"},
    ]

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert — bob (gui) carries the two default mounts plus the persona's two
    bob_volumes = compose["services"]["web-bob"]["volumes"]
    assert bob_volumes == [
        "bob-claude-config:/data/claude-config",
        "bob-agent-data:/app/dls-gui/_agent_data",
        "/opt/site-data:/app/site-data:ro",
        "shared-cache:/app/cache",
    ]
    # alice (default persona, no extra_mounts) keeps exactly the two default mounts
    assert compose["services"]["web-alice"]["volumes"] == [
        "alice-claude-config:/data/claude-config",
        "alice-agent-data:/app/dls-assistant/_agent_data",
    ]


def test_no_extra_mounts_leaves_only_the_two_default_volume_lines() -> None:
    """A no-personas config (the zero-migration default) emits exactly the two
    default per-user volume lines — the extra_mounts loop adds nothing."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    for user in config["modules"]["web_terminals"]["users"]:
        assert compose["services"][f"web-{user}"]["volumes"] == [
            f"{user}-claude-config:/data/claude-config",
            f"{user}-agent-data:/app/dls-assistant/_agent_data",
        ]


def test_auth_default_none() -> None:
    """No `web_terminals.auth` stanza -> auth_method defaults to 'none' (v1 has no auth)."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_method"] == "none"


def test_auth_default_none_reads_configured_method() -> None:
    """A configured `web_terminals.auth.method` is read through, not clobbered by the default."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {"method": "oauth2_proxy"}

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_method"] == "oauth2_proxy"


def test_auth_method_non_string_falls_back_to_none() -> None:
    """A malformed (non-str) `auth.method` (well-formedness is lint's job, not this
    function's) must not leak into the rendered seam as-is — it falls back to the
    same inert "none" default as an absent/empty method."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {"method": {"nested": "not-a-string"}}

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_method"] == "none"


def test_tls_default_off() -> None:
    """No `web_terminals.tls` stanza -> tls_enabled defaults to False (v1 stays http)."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["tls_enabled"] is False
    assert context["tls_cert"] is None
    assert context["tls_key"] is None


def test_tls_default_off_reads_configured_stanza() -> None:
    """A configured `web_terminals.tls` stanza is read through untouched."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["tls_enabled"] is True
    assert context["tls_cert"] == "/etc/nginx/certs/dls.crt"
    assert context["tls_key"] == "/etc/nginx/certs/dls.key"


def test_render_succeeds_with_auth_default_none_and_tls_default_off() -> None:
    """A config with no auth/tls stanzas at all still renders the full artifact set — 1.4 only
    establishes the config contract, it must not change Phase-1 render behavior."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    assert "auth" not in config["modules"]["web_terminals"]
    assert "tls" not in config["modules"]["web_terminals"]

    # Act
    artifacts = render_web_terminals(config)

    # Assert — round-trips exactly as before (defaults are inert; no seam rendering here)
    assert set(artifacts.keys()) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }
    yaml.safe_load(artifacts["docker-compose.web.yml"])


def test_render_tolerates_non_dict_auth_and_tls_stanzas() -> None:
    """`_as_dict`-style defensive reads: a malformed (non-dict) auth/tls value must not raise —
    it falls back to the same defaults as an absent stanza. Well-formedness is lint's job."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["auth"] = "not-a-dict"
    config["modules"]["web_terminals"]["tls"] = ["also", "not", "a", "dict"]

    # Act
    artifacts = render_web_terminals(config)
    context = _auth_tls_context(config["modules"]["web_terminals"])

    # Assert
    assert context["auth_method"] == "none"
    assert context["tls_enabled"] is False
    assert "docker-compose.web.yml" in artifacts


def test_nginx_seam_default_gated_off_emits_no_ssl_listen_or_auth_request() -> None:
    """C1: with no `auth`/`tls` stanzas (the v1 default posture), the rendered nginx
    fragment must carry NEITHER a `listen 443 ssl` block NOR an `auth_request`
    directive — the seam is fully inert until a facility config opts in."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    assert "auth" not in config["modules"]["web_terminals"]
    assert "tls" not in config["modules"]["web_terminals"]

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert "listen 443 ssl" not in nginx_conf
    assert "ssl_certificate" not in nginx_conf
    assert "auth_request" not in nginx_conf


def test_nginx_seam_tls_enabled_emits_ssl_listen_and_configured_cert_paths() -> None:
    """C2 (render half): `tls.enabled: true` emits `listen 443 ssl` plus
    `ssl_certificate`/`ssl_certificate_key` referencing the configured paths,
    inside the `server{}` block (not at http/top-level)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert "listen 443 ssl;" in nginx_conf
    assert "ssl_certificate /etc/nginx/certs/dls.crt;" in nginx_conf
    assert "ssl_certificate_key /etc/nginx/certs/dls.key;" in nginx_conf
    server_index = nginx_conf.index("server {")
    ssl_index = nginx_conf.index("listen 443 ssl;")
    assert server_index < ssl_index, "the ssl listener must sit inside server{}, not above it"


def test_nginx_seam_tls_enabled_without_cert_or_key_raises_value_error() -> None:
    """`tls.enabled: true` without both `cert` and `key` can't render a coherent
    `ssl_certificate`/`ssl_certificate_key` pair — fail loudly at render time rather
    than emit a directive pointing at a missing path (which `nginx -t` would reject
    anyway, but with a far less actionable error)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["tls"] = {"enabled": True}

    # Act / Assert
    with pytest.raises(ValueError, match="tls"):
        render_web_terminals(config)


def test_nginx_seam_auth_method_set_emits_auth_request_per_user_location() -> None:
    """C2 (render half): a configured `auth.method` (anything but the "none" default)
    emits an `auth_request` directive inside every proxied `/u/<user>/` location, plus
    the single internal target location it subrequests."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]
    config["modules"]["web_terminals"]["auth"] = {"method": "oauth2_proxy"}

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert nginx_conf.count("auth_request /_osprey_auth;") == len(users)
    assert nginx_conf.count("internal;") == 1
    # Fail-closed: the stub target denies (403), never silently authorizes (200) —
    # setting auth.method without wiring a real backend must lock users out.
    assert "return 403;" in nginx_conf
    assert "return 200;" not in nginx_conf


def test_nginx_seam_auth_none_omits_auth_request_even_with_tls_enabled() -> None:
    """The two seams are independently gated: TLS on with `auth.method` left at its
    "none" default must still omit `auth_request` entirely."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert "listen 443 ssl;" in nginx_conf
    assert "auth_request" not in nginx_conf


# ---------------------------------------------------------------------------
# Task 2.5: mcp.topology fail-closed
# ---------------------------------------------------------------------------


def test_mcp_topology_omitted_renders_unchanged() -> None:
    """No `web_terminals.mcp` stanza at all -> today's behavior, byte-identical
    to a config that has never heard of the topology key (zero migration)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    assert "mcp" not in config["modules"]["web_terminals"]

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert set(artifacts.keys()) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }


def test_mcp_topology_explicit_per_container_stdio_renders_unchanged() -> None:
    """Explicitly spelling out the default value is equally inert."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["mcp"] = {"topology": "per_container_stdio"}

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert set(artifacts.keys()) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }


def test_mcp_topology_shared_http_raises_value_error_scoped_to_shared_tier() -> None:
    """`shared_http` is a recognized-but-rejected value (Task 2.4 owns the lint
    ERROR twin of this check): render must fail closed, and the message must be
    scoped to the shared framework-MCP tier — it must never read as though it
    were rejecting HTTP MCP transport in general, since a facility's own
    `claude_code.servers` custom `url` entries are a separate, unaffected,
    already-supported path."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["mcp"] = {"topology": "shared_http"}

    # Act / Assert
    with pytest.raises(ValueError, match="mcp.topology") as exc_info:
        render_web_terminals(config)
    message = str(exc_info.value)
    assert "shared_http" in message
    assert "per_container_stdio" in message
    # Scoped to the shared framework-MCP tier, not a blanket HTTP-transport ban.
    assert "claude_code.servers" in message
    assert "unaffected" in message


def test_mcp_topology_unknown_value_raises_value_error() -> None:
    """Any value other than `per_container_stdio` is fail-closed, not just the one
    named-in-the-schema `shared_http` alternative."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["mcp"] = {"topology": "some_future_value"}

    # Act / Assert
    with pytest.raises(ValueError, match="mcp.topology"):
        render_web_terminals(config)


def test_mcp_topology_malformed_mcp_stanza_falls_back_to_default() -> None:
    """A non-dict `mcp` stanza (well-formedness is lint's job, not this
    function's) must not raise here — it falls back to the same default as an
    absent stanza, matching the `_as_dict`-style defensive-read convention used
    throughout this module (see `_auth_tls_context`)."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["mcp"] = "not-a-dict"

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert "docker-compose.web.yml" in artifacts


def test_mcp_topology_custom_url_server_config_is_a_separate_namespace() -> None:
    """Regression guard against over-broad validation: a facility config that
    also carries a project-level `claude_code.servers` custom `url` entry
    alongside `modules.web_terminals` must render exactly as it would without
    that entry — the topology gate reads only `modules.web_terminals.mcp`, and
    must never inspect, walk, or reject on the unrelated `claude_code.servers`
    key. The corresponding `.mcp.json` rendering assertion (that the url-server
    entry itself renders unchanged) lives in
    `tests/registry/test_mcp.py::test_topology_default_leaves_custom_url_server_untouched`.
    """
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["claude_code"] = {
        "servers": {
            "remote-api": {"url": "http://remote:8001/sse"},
        }
    }
    assert "mcp" not in config["modules"]["web_terminals"]

    # Act
    artifacts = render_web_terminals(config)

    # Assert — unaffected by the sibling claude_code.servers stanza
    assert set(artifacts.keys()) == {
        "docker-compose.web.yml",
        "nginx/nginx.conf",
        "nginx/landing.html",
    }


def test_nginx_landing_location_is_exact_match_only() -> None:
    """The landing page is served for `/` ONLY — every other unmatched path 404s.

    A prefix `location /` catch-all would answer stray API calls (e.g. a panel
    fetch that lost its `/u/<user>` prefix) with 200 + landing HTML, hiding the
    bug behind a downstream JSON parse error instead of a visible 404.
    """
    # Act
    artifacts = render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG))
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    assert "location = / {" in nginx_conf
    assert not re.search(r"location / \{", nginx_conf)
