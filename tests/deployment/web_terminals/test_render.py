"""Tests for multi-user web-terminal artifact rendering (osprey.deployment.web_terminals.render)."""

from __future__ import annotations

import copy
import re
from unittest.mock import patch

import pytest
import yaml
from jinja2 import Template

from osprey.deployment.web_terminals.auth_credentials import AUTH_ENV_FILENAME
from osprey.deployment.web_terminals.personas import env_var_suffix
from osprey.deployment.web_terminals.ports import (
    PANEL_ENV_VARS,
    allocate_ports,
    base_ports_from_config,
)
from osprey.deployment.web_terminals.render import _auth_tls_context, render_web_terminals

# The sidecar's own env-var names, imported rather than spelled out: the compose
# service and the app factory are the two ends of one contract, and a rename on
# either side must fail here instead of silently rendering a variable nothing reads.
from osprey.services.auth_sidecar.app import (
    ENV_EXTERNAL_ORIGIN,
    ENV_METHOD,
    ENV_OIDC_CLIENT_ID_ENV,
    ENV_OIDC_CLIENT_SECRET_ENV,
    ENV_OIDC_ISSUER,
    ENV_OIDC_SUBJECT_PREFIX,
    ENV_PW_HASH_PREFIX,
    ENV_SESSION_LIFETIME,
    ENV_SESSION_SECRET,
    ENV_STATE_SECRET,
    ENV_TLS_ENABLED,
    ENV_USERS,
)

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


# ---------------------------------------------------------------------------
# Task 1.3: one external-origin derivation behind landing_url and OIDC
# ---------------------------------------------------------------------------


def _tls_config(users: list[str] | None = None) -> dict:
    """A config with TLS terminated at nginx (cert/key set, as the render demands)."""
    config = copy.deepcopy(_config(users) if users is not None else _MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }
    return config


def _capture_render_contexts(config: dict) -> dict[str, dict]:
    """Render `config`, capturing the Jinja context each template was given.

    `external_origin` is exported for templates that don't consume it yet (the
    nginx TLS-redirect block and the compose sidecar service land later), so the
    context itself is the only place its export can be pinned today.
    """
    captured: dict[str, dict] = {}

    def _record(self, **context):  # type: ignore[no-untyped-def]
        captured[self.name] = context
        return ""

    with patch.object(Template, "render", _record):
        render_web_terminals(config)
    return captured


def test_external_origin_without_tls_is_plain_http_on_the_published_nginx_port() -> None:
    """No TLS: nginx publishes one plain port, and that port is part of the origin."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    contexts = _capture_render_contexts(config)

    # Assert
    assert contexts["nginx.conf.j2"]["external_origin"] == "http://dls-deploy.dls.example.org:9080"


def test_external_origin_with_tls_is_https_with_443_left_implicit() -> None:
    """TLS on: the 443 server is the sole content server, and 443 is left implicit —
    the canonical origin serialization, and the form an OIDC client registration is
    normally written in (an IdP compares `redirect_uri` character for character)."""
    # Arrange
    config = _tls_config()

    # Act
    contexts = _capture_render_contexts(config)

    # Assert — the nginx_port never appears in a TLS origin
    assert contexts["nginx.conf.j2"]["external_origin"] == "https://dls-deploy.dls.example.org"


def test_external_origin_is_exported_to_both_the_nginx_and_compose_contexts() -> None:
    """One derivation, exported to every consumer: the nginx server blocks and the
    compose overlay (whose auth sidecar builds its OIDC redirect_uri from it)."""
    # Arrange
    config = _tls_config()

    # Act
    contexts = _capture_render_contexts(config)

    # Assert
    origin = "https://dls-deploy.dls.example.org"
    assert contexts["nginx.conf.j2"]["external_origin"] == origin
    assert contexts["docker-compose.web.yml.j2"]["external_origin"] == origin


def test_landing_url_is_the_external_origin_verbatim_under_tls() -> None:
    """The divergence guard this task exists for: `landing_url` is not assembled
    separately, so it cannot drift from the origin the OIDC redirect_uri uses."""
    # Arrange
    config = _tls_config()

    # Act
    contexts = _capture_render_contexts(config)
    compose_ctx = contexts["docker-compose.web.yml.j2"]

    # Assert
    assert compose_ctx["landing_url"] == compose_ctx["external_origin"]


def test_landing_url_baked_into_containers_follows_tls_into_https() -> None:
    """Regression: OSPREY_TERMINAL_LANDING_URL used to be hardcoded to
    `http://<fqdn>:<nginx_port>` regardless of TLS, so a TLS deployment shipped every
    container a "back to landing" link on the plain origin — which under the TLS
    posture only ever redirects, and would never carry a Secure session cookie."""
    # Arrange
    config = _tls_config()

    # Act
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])

    # Assert
    landing_env = [
        env
        for env in compose["services"]["web-alice"]["environment"]
        if env.startswith("OSPREY_TERMINAL_LANDING_URL=")
    ]
    assert landing_env == ["OSPREY_TERMINAL_LANDING_URL=https://dls-deploy.dls.example.org"]


def test_landing_url_without_tls_is_unchanged_from_today() -> None:
    """The plain-HTTP deployment (every current facility) renders byte-identically."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])

    # Assert
    assert (
        "OSPREY_TERMINAL_LANDING_URL=http://dls-deploy.dls.example.org:9080"
        in compose["services"]["web-alice"]["environment"]
    )


def test_external_origin_is_required_by_auth_even_with_an_empty_roster() -> None:
    """`deploy.fqdn` stops being users-only once auth is on: the sidecar needs an
    origin to build its redirect_uri from, so a roster-less auth deployment must fail
    loudly here rather than at the IdP with a redirect_uri mismatch."""
    # Arrange
    config = _tls_config(users=[])
    del config["deploy"]["fqdn"]
    config["modules"]["web_terminals"]["auth"] = {"method": "oidc"}

    # Act / Assert
    with pytest.raises(ValueError, match="deploy.fqdn"):
        render_web_terminals(config)


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


# ---------------------------------------------------------------------------
# Landing page: design-system token values baked from web.theme
# ---------------------------------------------------------------------------


def _landing_style(config: dict) -> str:
    """The landing page's inline <style> block, up to the first static rule."""
    html = render_web_terminals(config)["nginx/landing.html"]
    return html[html.index("<style>") : html.index("* { box-sizing")]


def _themed_config(theme: str | None) -> dict:
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    if theme is not None:
        config.setdefault("web", {})["theme"] = theme
    return config


def test_landing_bakes_the_configured_theme_family_in_both_modes() -> None:
    """A family-valued web.theme yields dark under :root plus a light media query,
    so the page follows the viewer's OS — the same thing the terminals behind it do
    when configured with a bare family."""
    # Act
    style = _landing_style(_themed_config("desy"))

    # Assert — DESY dark canvas under :root, DESY light behind the query
    assert "--bg-primary: #091720;" in style
    assert "@media (prefers-color-scheme: light)" in style
    assert "--bg-primary: #f3f8fc;" in style
    assert "color-scheme: dark;" in style
    assert "color-scheme: light;" in style


def test_landing_honors_a_pinned_mode_with_no_media_query() -> None:
    """A concrete-id web.theme pins the mode: one block, and no OS-following."""
    # Act
    style = _landing_style(_themed_config("desy-light"))

    # Assert
    assert "--bg-primary: #f3f8fc;" in style
    assert "color-scheme: light;" in style
    assert "prefers-color-scheme" not in style
    assert "color-scheme: dark;" not in style


def test_landing_defaults_to_the_main_family_when_web_theme_is_absent() -> None:
    """No `web` section at all still themes the page — from the main family."""
    # Act
    style = _landing_style(_themed_config(None))

    # Assert — main family's canvases, and still both modes
    assert "--bg-primary: #111217;" in style
    assert "--bg-primary: #f4f5f5;" in style


def test_landing_falls_back_on_an_unknown_web_theme() -> None:
    """A typo must not leave the page unstyled — it resolves like the terminals do,
    to the main family, and pins nothing."""
    # Act
    style = _landing_style(_themed_config("no-such-theme"))

    # Assert
    assert "--bg-primary: #111217;" in style
    assert "@media (prefers-color-scheme: light)" in style


def test_landing_carries_no_hardcoded_pre_token_palette() -> None:
    """The hand-written hexes this page used before the token system are gone.

    Guards the actual regression risk: a later edit reintroducing a literal
    colour would look fine on the default theme and be wrong on every other one.
    """
    # Act
    style = _landing_style(_themed_config("desy"))

    # Assert — the old teal accent and slate canvas
    assert "#4fd1c5" not in style
    assert "#0a0f1a" not in style


def test_landing_defines_every_variable_its_stylesheet_uses() -> None:
    """Every `var(--x)` in the page must be baked; a standalone file has no
    stylesheet to fall back to, so an undefined one renders as nothing."""
    # Arrange
    html = render_web_terminals(_themed_config("desy"))["nginx/landing.html"]

    # Act
    used = set(re.findall(r"var\((--[a-z0-9-]+)\)", html))
    defined = set(re.findall(r"^\s+(--[a-z0-9-]+):", html, re.MULTILINE))

    # Assert
    assert used, "expected the landing stylesheet to use custom properties"
    assert used <= defined, f"undefined custom properties: {sorted(used - defined)}"


def test_landing_theme_values_are_pattern_checked_before_inlining() -> None:
    """A value that could break out of <style> fails the render rather than
    shipping. HTML escaping would not catch a `</style>` sequence."""
    # Arrange
    config = _themed_config("desy")

    # Act / Assert
    with patch(
        "osprey.interfaces.design_system.theme_config.theme_css_variables",
        return_value={"--bg-primary": "#000</style><script>alert(1)</script>"},
    ):
        with pytest.raises(ValueError, match="not safe to inline"):
            render_web_terminals(config)


def test_landing_render_fails_loudly_on_a_renamed_token() -> None:
    """A token the page needs but the theme no longer defines must break the
    render, not leave a deployed page quietly off-palette."""
    # Arrange
    from osprey.interfaces.design_system.theme_config import MissingThemeVariableError

    config = _themed_config("desy")

    # Act / Assert
    with patch(
        "osprey.deployment.web_terminals.render._LANDING_THEME_VARIABLES",
        ("--bg-primary", "--no-such-token"),
    ):
        with pytest.raises(MissingThemeVariableError, match="no-such-token"):
            render_web_terminals(config)


# ---------------------------------------------------------------------------
# theme -> OSPREY_WEB_THEME (per-user default theme seam)
# ---------------------------------------------------------------------------


def test_theme_emits_theme_env_line_only_for_the_user_that_sets_it() -> None:
    """A user's `theme` renders an `OSPREY_WEB_THEME` env line for that service; a
    user without one omits the line entirely (app falls back to config web.theme)."""
    # Arrange
    config = copy.deepcopy(_config([{"name": "alice", "index": 0, "theme": "desy-light"}, "bob"]))

    # Act
    artifacts = render_web_terminals(config)
    compose = yaml.safe_load(artifacts["docker-compose.web.yml"])

    # Assert
    assert _service_env(compose, "web-alice")["OSPREY_WEB_THEME"] == "desy-light"
    assert "OSPREY_WEB_THEME" not in _service_env(compose, "web-bob")


def test_no_theme_anywhere_emits_no_theme_env_line() -> None:
    """The common (bare-string) roster emits no OSPREY_WEB_THEME line at all — the
    seam is inert until a user opts in."""
    # Act
    artifacts = render_web_terminals(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert "OSPREY_WEB_THEME" not in artifacts["docker-compose.web.yml"]


def test_theme_and_display_name_are_independent_per_user() -> None:
    """Each optional per-user field is emitted on its own; setting one must not
    drag the other in or leave it out."""
    # Arrange
    config = copy.deepcopy(
        _config(
            [
                {"name": "alice", "index": 0, "theme": "desy"},
                {"name": "bob", "index": 1, "display_name": "Operations"},
            ]
        )
    )

    # Act
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])

    # Assert
    alice, bob = _service_env(compose, "web-alice"), _service_env(compose, "web-bob")
    assert alice["OSPREY_WEB_THEME"] == "desy"
    assert "OSPREY_WEB_APP_NAME" not in alice
    assert bob["OSPREY_WEB_APP_NAME"] == "Operations"
    assert "OSPREY_WEB_THEME" not in bob


def test_theme_value_is_yaml_quoted_and_round_trips() -> None:
    """Quoted the same way as OSPREY_WEB_APP_NAME: a value containing a `": "`
    would otherwise derail the unquoted compose `- KEY=value` scalar into a
    mapping. Theme ids never contain one today — this guards the quoting itself,
    so a hand-written config can't break the compose file."""
    # Arrange
    tricky = "not: a theme"
    config = copy.deepcopy(_config([{"name": "alice", "index": 0, "theme": tricky}]))

    # Act
    artifacts = render_web_terminals(config)
    compose_text = artifacts["docker-compose.web.yml"]
    compose = yaml.safe_load(compose_text)

    # Assert
    assert '- "OSPREY_WEB_THEME=not: a theme"' in compose_text
    assert _service_env(compose, "web-alice")["OSPREY_WEB_THEME"] == tricky


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
    web_terminals["auth"] = {"method": "password"}

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_method"] == "password"


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


# ---------------------------------------------------------------------------
# Task 1.2: the full auth stanza — `_auth_tls_context` is the single parser
# ---------------------------------------------------------------------------


def test_auth_context_defaults_are_inert_without_an_auth_stanza() -> None:
    """No `auth` stanza -> every auth key resolves to a default that renders nothing:
    no method, no image pin, no OIDC issuer. The two knobs that must still hold a
    usable value (the sidecar's port and the session lifetime) carry the documented
    defaults so a facility enabling auth needs only `method`."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_method"] == "none"
    assert context["auth_allow_insecure_http"] is False
    assert context["auth_image"] is None
    assert context["auth_oidc_issuer"] is None
    assert context["auth_port"] == 9070
    assert context["auth_session_lifetime"] == 12 * 60 * 60


def test_auth_context_reads_every_configured_scalar() -> None:
    """A fully specified `auth` stanza is read through field for field."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {
        "method": "password",
        "port": 9401,
        "session_lifetime": 3600,
        "allow_insecure_http": True,
        "image": "git.dls.example.org:5050/physics/production/dls-auth:2026.8.1",
    }

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_method"] == "password"
    assert context["auth_port"] == 9401
    assert context["auth_session_lifetime"] == 3600
    assert context["auth_allow_insecure_http"] is True
    assert context["auth_image"].endswith("dls-auth:2026.8.1")


def test_auth_context_reads_oidc_issuer_and_credential_env_var_names() -> None:
    """`oidc` mode names its issuer inline and its client credentials only by env-var
    name — the credentials themselves live in `.env.auth`, never in config.yml."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {
        "method": "oidc",
        "allow_insecure_http": True,
        "oidc": {
            "issuer": "https://sso.dls.example.org/realms/facility",
            "client_id_env": "DLS_SSO_CLIENT_ID",
            "client_secret_env": "DLS_SSO_CLIENT_SECRET",
        },
    }

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_method"] == "oidc"
    assert context["auth_oidc_issuer"] == "https://sso.dls.example.org/realms/facility"
    assert context["auth_oidc_client_id_env"] == "DLS_SSO_CLIENT_ID"
    assert context["auth_oidc_client_secret_env"] == "DLS_SSO_CLIENT_SECRET"


def test_auth_context_oidc_credential_env_var_names_have_defaults() -> None:
    """A facility that doesn't rename them gets the framework's own env-var names, so
    an `oidc` stanza only has to carry the issuer."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {
        "method": "oidc",
        "allow_insecure_http": True,
        "oidc": {"issuer": "https://sso.dls.example.org/realms/facility"},
    }

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_oidc_client_id_env"] == "OSPREY_AUTH_OIDC_CLIENT_ID"
    assert context["auth_oidc_client_secret_env"] == "OSPREY_AUTH_OIDC_CLIENT_SECRET"


def test_auth_context_unknown_method_raises_value_error() -> None:
    """An `auth.method` naming a method no sidecar implements is a hard error, not a
    forward-compatible passthrough: rendering it would emit an `auth_request` seam
    that nothing can answer, locking the deployment out with an unactionable 403."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {"method": "oauth2_proxy"}

    # Act / Assert
    with pytest.raises(ValueError, match="auth.method") as exc_info:
        _auth_tls_context(web_terminals)
    message = str(exc_info.value)
    assert "oauth2_proxy" in message
    # The message names the alternatives, so the fix is obvious from the error alone.
    assert "password" in message
    assert "oidc" in message


def test_auth_context_malformed_scalars_fall_back_to_defaults() -> None:
    """Wrong-typed values follow this module's defensive-read convention (lint is the
    authoritative gate on well-formedness) rather than crashing the render. `bool` is
    called out explicitly: it satisfies `isinstance(x, int)`, so an unguarded read
    would turn `port: true` into port 1."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {
        "port": True,
        "session_lifetime": -1,
        "image": "",
        "oidc": "not-a-dict",
    }

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    assert context["auth_port"] == 9070
    assert context["auth_session_lifetime"] == 12 * 60 * 60
    assert context["auth_image"] is None
    assert context["auth_oidc_issuer"] is None
    assert context["auth_oidc_client_id_env"] == "OSPREY_AUTH_OIDC_CLIENT_ID"


def test_auth_context_never_reads_a_credential_out_of_config() -> None:
    """Secrets belong in `.env.auth`; the parser reads env-var *names* only. A config
    that pastes a literal client secret (or password hash) in must not carry it into
    any render context, where it would land in a committed compose/nginx artifact."""
    # Arrange
    web_terminals = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    web_terminals["auth"] = {
        "method": "oidc",
        "allow_insecure_http": True,
        "oidc": {
            "issuer": "https://sso.dls.example.org/realms/facility",
            "client_secret": "s3cr3t-should-never-render",
        },
        "password_hash": "scrypt.16384.8.1.salt.hash",
    }

    # Act
    context = _auth_tls_context(web_terminals)

    # Assert
    rendered_values = " ".join(str(value) for value in context.values())
    assert "s3cr3t-should-never-render" not in rendered_values
    assert "scrypt." not in rendered_values


def test_render_auth_context_gate_method_without_tls_raises() -> None:
    """Fail-closed render gate: authentication over cleartext HTTP would ship session
    cookies in the open, so the render refuses rather than emitting it — the same
    posture as the tls cert/key fail-fast right beside it."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["auth"] = {"method": "password"}

    # Act / Assert
    with pytest.raises(ValueError, match="allow_insecure_http") as exc_info:
        render_web_terminals(config)
    assert "tls.enabled" in str(exc_info.value)


def test_render_auth_context_gate_allow_insecure_http_permits_the_render() -> None:
    """The escape hatch is explicit and per-deployment: a facility terminating TLS at
    an upstream proxy opts in by name."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["auth"] = {
        "method": "password",
        "allow_insecure_http": True,
    }

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert "auth_request" in artifacts["nginx/nginx.conf"]


def test_render_auth_context_gate_is_satisfied_by_tls_enabled() -> None:
    """The ordinary way to pass the gate is to actually enable TLS."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["auth"] = {"method": "oidc"}
    config["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert "auth_request" in artifacts["nginx/nginx.conf"]


def test_render_auth_context_gate_does_not_fire_for_method_none() -> None:
    """The gate is about protecting session cookies; with no authentication there are
    none, so plain-HTTP deployments keep rendering exactly as they do today."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    config["modules"]["web_terminals"]["auth"] = {"method": "none"}

    # Act
    artifacts = render_web_terminals(config)

    # Assert
    assert "auth_request" not in artifacts["nginx/nginx.conf"]


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


def test_nginx_seam_auth_method_set_gates_every_proxied_user_location() -> None:
    """C2 (render half): a configured `auth.method` (anything but the "none" default)
    leaves NO proxied `/u/<user>/` location ungated.

    The per-user target shape each `auth_request` names is pinned by the
    `auth_location` tests below; this one guards the coverage property they
    can't — that the count of gated locations equals the count of proxied ones,
    so a user added through a code path that forgets the gate fails here.
    Fail-closed is asserted alongside it: nothing in the render authorizes on
    its own, so a sidecar that is down or unconfigured denies rather than
    admits.
    """
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)
    users = config["modules"]["web_terminals"]["users"]
    config["modules"]["web_terminals"]["auth"] = {
        "method": "password",
        "allow_insecure_http": True,
    }

    # Act
    artifacts = render_web_terminals(config)
    nginx_conf = artifacts["nginx/nginx.conf"]

    # Assert
    directives = _directives(nginx_conf)
    assert directives.count("auth_request ") == len(users)
    assert directives.count("    location /u/") == len(users)
    assert "return 200;" not in directives


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


# ---------------------------------------------------------------------------
# Task 3.1: per-user internal auth locations
# ---------------------------------------------------------------------------


def _auth_config(users: list[str] | None = None, **auth: object) -> dict:
    """A rendering config with authentication on.

    `allow_insecure_http` is set unless the caller enables TLS, because the
    render refuses `auth.method` over cleartext otherwise — a gate these tests
    are not about (it has its own coverage above).
    """
    config = _config(users if users is not None else ["alice", "bob", "carol"])
    stanza: dict = {"method": "password", "allow_insecure_http": True}
    stanza.update(auth)
    config["modules"]["web_terminals"]["auth"] = stanza
    return config


def _render_nginx(config: dict) -> str:
    return render_web_terminals(config)["nginx/nginx.conf"]


def _directives(conf: str) -> str:
    """`conf` with its comment lines dropped.

    Counting and absence assertions have to ignore comment prose, which in this
    template frequently names the very construct being counted or ruled out —
    the comment above `error_page` explains what the `auth_request` above it
    does, and would otherwise be counted as a second `auth_request`.
    """
    return "\n".join(line for line in conf.splitlines() if not line.lstrip().startswith("#"))


def test_auth_location_is_rendered_once_per_roster_user() -> None:
    """Each user gets their OWN internal auth target, not a shared one.

    A single shared target could only learn which user is being authorized from
    the request itself (`$request_uri`/`X-Original-URI`) — client-controlled
    input on the authorization path. Per-user locations make the identity a
    property of which location matched, which the client cannot influence.
    """
    # Arrange
    users = ["alice", "bob", "carol"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    for user in users:
        assert f"location = /_osprey_auth/{user} {{" in nginx_conf
    assert nginx_conf.count("location = /_osprey_auth/") == len(users)
    # The pre-sidecar shared stub is gone, not merely supplemented.
    assert "location = /_osprey_auth {" not in nginx_conf


def test_auth_location_is_named_by_each_users_own_auth_request() -> None:
    """`/u/<user>/` subrequests `<user>`'s target — never another user's.

    Pinning the pairing (not just the presence of both directives) is what
    catches a loop that renders every `auth_request` against the first user.
    """
    # Arrange
    users = ["alice", "bob", "carol"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    for user in users:
        location = re.search(rf"location /u/{user}/ \{{(.*?)\n    \}}", nginx_conf, flags=re.DOTALL)
        assert location is not None, f"no proxied location rendered for {user}"
        assert f"auth_request /_osprey_auth/{user};" in location.group(1)
    assert _directives(nginx_conf).count("auth_request ") == len(users)


def test_auth_location_verifies_against_the_sidecar_with_a_render_time_username() -> None:
    """The target proxies to the sidecar's `/verify`, with `?user=` baked in.

    The username is a literal from the roster, so the sidecar is asked about
    exactly the user whose prefix was requested — there is no code path where a
    crafted URL, header, or cookie chooses a different one.
    """
    # Act
    nginx_conf = _render_nginx(_auth_config(["alice", "bob"]))

    # Assert
    assert "proxy_pass http://127.0.0.1:9070/verify?user=alice;" in nginx_conf
    assert "proxy_pass http://127.0.0.1:9070/verify?user=bob;" in nginx_conf


def test_auth_location_follows_the_configured_auth_port() -> None:
    """A facility that moves the sidecar off 9070 is routed to the new port —
    the template reads `auth_port`, it never hardcodes the default."""
    # Act
    nginx_conf = _render_nginx(_auth_config(["alice"], port=9471))

    # Assert
    assert "proxy_pass http://127.0.0.1:9471/verify?user=alice;" in nginx_conf
    assert ":9070/verify" not in nginx_conf


def test_auth_location_is_internal_only() -> None:
    """Every auth target carries `internal;`, so a browser cannot request one
    directly — the subrequest is the only way in."""
    # Arrange
    users = ["alice", "bob", "carol"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    for user in users:
        target = re.search(
            rf"location = /_osprey_auth/{user} \{{(.*?)\n    \}}", nginx_conf, flags=re.DOTALL
        )
        assert target is not None, f"no auth target rendered for {user}"
        assert "internal;" in target.group(1)
    assert nginx_conf.count("internal;") == len(users)


def test_auth_location_suppresses_the_subrequest_body() -> None:
    """`proxy_pass_request_body off` + a cleared `Content-Length` on every auth
    target.

    Without both, nginx announces the parent POST's body to a sidecar that
    never reads it, and the subrequest hangs until it times out — every
    authenticated POST in the deployment stalls. This assertion is the only
    cheap guard for that; the failure mode is invisible to a syntax check.
    """
    # Arrange
    users = ["alice", "bob"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    assert nginx_conf.count("proxy_pass_request_body off;") == len(users)
    assert nginx_conf.count('proxy_set_header Content-Length "";') == len(users)


def test_auth_location_verify_subrequest_fails_fast() -> None:
    """Every verify target states short timeouts instead of inheriting 60s.

    This subrequest sits in front of every gated request, so its worst case is
    the deployment's worst case: on nginx's defaults, one wedged sidecar holds
    each open terminal's next request for a minute before failing it. A
    loopback signature check has no legitimate reason to take that long.
    """
    # Arrange
    users = ["alice", "bob", "carol"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    for user in users:
        body = _location_body(nginx_conf, f"location = /_osprey_auth/{user}")
        assert "proxy_connect_timeout 2s;" in body
        assert "proxy_read_timeout 5s;" in body
    # The gated terminal locations keep the long SSE/WebSocket read timeout —
    # the short one belongs to the subrequest only.
    assert "proxy_read_timeout 3600s;" in _location_body(nginx_conf, "location /u/alice/")


def test_auth_location_absent_when_method_is_none() -> None:
    """`auth.method: none` renders no auth target, no `auth_request`, and no
    reference to the sidecar's port — byte-for-byte the pre-auth output.

    (The wider byte-equality guarantee is pinned by the golden fixtures in
    test_golden_render.py; this asserts the property directly for the explicit
    `method: none` spelling as well as the absent-stanza default.)
    """
    # Arrange
    default_config = copy.deepcopy(_MULTI_USER_CONFIG)
    explicit_none = copy.deepcopy(_MULTI_USER_CONFIG)
    explicit_none["modules"]["web_terminals"]["auth"] = {"method": "none"}

    # Act
    default_conf = _render_nginx(default_config)
    explicit_conf = _render_nginx(explicit_none)

    # Assert
    assert explicit_conf == default_conf
    assert "_osprey_auth" not in default_conf
    assert "auth_request" not in default_conf
    assert "9070" not in default_conf


def test_auth_location_zero_users_renders_no_auth_targets() -> None:
    """An empty roster with auth on has nobody to authorize: no targets, and —
    critically — no location that answers 200 in their absence."""
    # Act
    nginx_conf = _render_nginx(_auth_config([]))

    # Assert
    assert "_osprey_auth" not in nginx_conf
    assert "return 200;" not in nginx_conf


# ---------------------------------------------------------------------------
# Task 3.2: public /auth locations
# ---------------------------------------------------------------------------


def _location_body(nginx_conf: str, header: str) -> str:
    """The directives inside a top-level `location …` block, by its header line."""
    match = re.search(rf"{re.escape(header)} \{{(.*?)\n    \}}", nginx_conf, flags=re.DOTALL)
    assert match is not None, f"no `{header}` block in the rendered fragment"
    return match.group(1)


def test_auth_public_location_proxies_the_login_surface_to_the_sidecar() -> None:
    """`/auth/` reaches the sidecar's own `/auth/` prefix — the login page, the
    credential POST, the OIDC callback, logout, and the page's assets all ride
    this one prefix location rather than an enumeration that drifts from the
    sidecar's router."""
    # Act
    nginx_conf = _render_nginx(_auth_config())

    # Assert
    assert "    location /auth/ {" in nginx_conf
    assert "proxy_pass http://127.0.0.1:9070/auth/;" in _location_body(
        nginx_conf, "location /auth/"
    )


def test_auth_public_location_follows_the_configured_auth_port() -> None:
    """The public surface and the internal verify targets are pointed at the same
    configured sidecar port — neither hardcodes the 9070 default."""
    # Act
    nginx_conf = _render_nginx(_auth_config(port=9471))

    # Assert
    assert "proxy_pass http://127.0.0.1:9471/auth/;" in nginx_conf
    assert ":9070" not in nginx_conf


def test_auth_public_location_is_not_behind_auth_request() -> None:
    """The login flow must answer without a session — gating it behind one is a
    locked door whose key is inside the room. Same for the landing page at `= /`,
    which is both the identity chooser and what the nginx healthcheck probes."""
    # Act
    nginx_conf = _render_nginx(_auth_config())

    # Assert
    assert "auth_request" not in _location_body(nginx_conf, "location /auth/")
    assert "auth_request" not in _location_body(nginx_conf, "location = /")


def test_auth_public_location_is_not_internal() -> None:
    """`/auth/` is browser-reachable: an `internal` marker here would 404 every
    login attempt (the exact failure a copy-paste from the verify targets makes,
    and one no syntax check catches)."""
    # Act
    nginx_conf = _render_nginx(_auth_config())

    # Assert
    assert "internal;" not in _location_body(nginx_conf, "location /auth/")


def test_auth_public_location_sets_explicit_short_proxy_timeouts() -> None:
    """The login surface states its own timeouts instead of taking whatever the
    terminal locations need.

    The `/u/<user>/` locations run `proxy_read_timeout 3600s` to keep SSE and
    WebSocket streams alive; the login flow is a few short round trips to a
    loopback service, so a wedged sidecar should surface in seconds rather than
    hang a browser tab.
    """
    # Act
    body = _location_body(_render_nginx(_auth_config()), "location /auth/")

    # Assert
    assert "proxy_connect_timeout 5s;" in body
    assert "proxy_send_timeout 15s;" in body
    assert "proxy_read_timeout 15s;" in body
    assert "3600s" not in body


def test_auth_public_location_does_not_expose_the_verify_endpoint() -> None:
    """`/verify` is reachable only through the `internal` per-user targets. It
    must not be proxied under the public prefix, where a browser could ask the
    sidecar about any username it liked."""
    # Act
    nginx_conf = _render_nginx(_auth_config())

    # Assert
    assert "location /verify" not in nginx_conf
    assert "location = /verify" not in nginx_conf
    # Every directive (as opposed to comment prose) naming `/verify` is the
    # proxy_pass target of an internal per-user location.
    verify_lines = [line for line in _directives(nginx_conf).splitlines() if "/verify" in line]
    assert verify_lines, "the verify targets themselves should still be rendered"
    assert all("/verify?user=" in line for line in verify_lines)


def test_auth_public_location_absent_when_method_is_none() -> None:
    """No sidecar, no public login surface — a `none` deployment renders no
    `/auth/` location at all."""
    # Act
    nginx_conf = _render_nginx(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert "location /auth/" not in nginx_conf


def test_auth_public_location_renders_for_an_empty_roster() -> None:
    """The login surface is a property of the sidecar, not of the roster: it is
    rendered even with zero users, so a roster added later needs no second
    render path (and an empty deployment still serves a coherent origin)."""
    # Act
    nginx_conf = _render_nginx(_auth_config([]))

    # Assert
    assert "    location /auth/ {" in nginx_conf


# ---------------------------------------------------------------------------
# Task 3.5: upstream cookie stripping
# ---------------------------------------------------------------------------


def test_cookie_strip_on_every_proxied_user_location() -> None:
    """No browser cookie reaches a per-user container.

    One cookie jar serves the whole origin, so an operator with several users
    unlocked would otherwise hand each container a session cookie naming all of
    them — and these containers run agent-generated code, so anything readable
    inside one has effectively left the perimeter.
    """
    # Arrange
    users = ["alice", "bob", "carol"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    for user in users:
        assert 'proxy_set_header Cookie "";' in _location_body(nginx_conf, f"location /u/{user}/")


def test_cookie_strip_spares_the_sidecar_locations() -> None:
    """The sidecar is the one upstream that MUST see the cookie — it is the
    thing that reads the session. Stripping it there would deny every request
    and lock the deployment out."""
    # Act
    nginx_conf = _render_nginx(_auth_config(["alice"]))

    # Assert
    assert "Cookie" not in _location_body(nginx_conf, "location /auth/")
    assert "Cookie" not in _directives(_location_body(nginx_conf, "location = /_osprey_auth/alice"))


def test_cookie_strip_absent_when_method_is_none() -> None:
    """Without authentication there is no auth cookie to keep out of a
    container, and the render stays byte-identical to its pre-auth self."""
    # Act
    nginx_conf = _render_nginx(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert "Cookie" not in nginx_conf


# ---------------------------------------------------------------------------
# Task 3.3: content-negotiated 401
# ---------------------------------------------------------------------------


def test_negotiated_401_classifies_requests_by_accept_and_upgrade() -> None:
    """A `map` decides navigation-vs-program from the Accept header, with the
    Upgrade header concatenated in front of it.

    The concatenation is what excludes WebSocket handshakes: an upgrade token
    in front pushes `text/html` off the `^` anchor, so a handshake scores 0
    whatever its Accept header claims.
    """
    # Act
    nginx_conf = _render_nginx(_auth_config())

    # Assert
    assert "map $http_upgrade$http_accept $osprey_auth_wants_login_page {" in nginx_conf
    map_body = re.search(
        r"map \$http_upgrade\$http_accept \$osprey_auth_wants_login_page \{(.*?)\n\}",
        nginx_conf,
        flags=re.DOTALL,
    )
    assert map_body is not None
    assert "default 0;" in map_body.group(1)
    assert '"~*^text/html" 1;' in map_body.group(1)


def test_negotiated_401_every_user_location_intercepts_with_the_shared_handler() -> None:
    """Each `/u/<user>/` sends its 401 to the named handler and states, as a
    render-time literal, which user the handler should prompt for.

    `error_page 401 = @…` (with the `=`) is what lets the handler's own status
    replace the 401 — without it a browser would receive a 401 carrying a
    Location header, which it does not follow.
    """
    # Arrange
    users = ["alice", "bob", "carol"]

    # Act
    nginx_conf = _render_nginx(_auth_config(users))

    # Assert
    for user in users:
        body = _location_body(nginx_conf, f"location /u/{user}/")
        assert f"set $osprey_auth_user {user};" in body
        assert "error_page 401 = @osprey_auth_401;" in body


def test_negotiated_401_handler_redirects_navigations_and_bares_everything_else() -> None:
    """The handler redirects only when the map said "navigation", and answers a
    bare 401 otherwise.

    A blanket redirect would hand a JSON caller a page of login HTML and leave
    an EventSource reconnecting forever against a login page that never becomes
    the stream it asked for.
    """
    # Act
    body = _location_body(_render_nginx(_auth_config()), "location @osprey_auth_401")

    # Assert
    assert "if ($osprey_auth_wants_login_page) {" in body
    assert "return 302 /auth/login?user=$osprey_auth_user&next=$osprey_auth_next;" in body
    # The unconditional fallthrough, outside the `if`.
    assert re.search(r"\n        return 401;", body)


def test_negotiated_401_return_to_is_allowlisted_never_a_raw_path_variable() -> None:
    """The return-to emits `$osprey_auth_next` — and neither raw path variable.

    `$uri` is decoded by the time it is read, so emitting it copies a decoded
    CR LF straight into the Location header (response-header injection).
    `$request_uri` is injection-safe but still client-controlled, and a leading
    `//` in it reads as an absolute URL that walks the browser off-site. Only
    the allowlist map produces a value that is safe to emit AND safe to trust.
    """
    # Act
    nginx_conf = _render_nginx(_auth_config())
    body = _location_body(nginx_conf, "location @osprey_auth_401")

    # Assert
    assert "next=$osprey_auth_next;" in body
    directives = _directives(body)
    assert "next=$uri" not in directives
    assert "$request_uri" not in directives


def test_negotiated_401_return_to_allowlist_rejects_injection_and_keeps_deep_links() -> None:
    """The allowlist regex itself, exercised against what an attacker sends.

    This runs the pattern the template actually emits rather than restating it,
    so a widened character class fails here. nginx decodes percent-escapes
    before this map is consulted, hence the decoded forms below: `%0d%0a`
    arrives as a real CR LF, `%26` as a real `&`.
    """
    # Arrange
    nginx_conf = _render_nginx(_auth_config())
    pattern = re.search(r'\n    "~(\S+)" \$uri;', nginx_conf)
    assert pattern is not None, "no allowlist entry in the $osprey_auth_next map"
    # PCRE's `\z` is spelled `\Z` in Python; the anchor semantics are the same
    # (end of subject, NOT "before a trailing newline" the way `$` behaves).
    allowlist = re.compile(pattern.group(1).replace(r"\z", r"\Z"))

    # Act / Assert — legitimate deep links survive
    for path in (
        "/u/alice/",
        "/u/alice/files/notes_v2.1-final.txt",
        "/u/alice-b/panel/~cache/x",
    ):
        assert allowlist.match(path), f"deep link {path!r} should round-trip"

    # Act / Assert — every hostile decoded form collapses to the empty default
    for path in (
        "/u/alice/x\r\nX-Injected: yes",  # header injection
        "/u/alice/x\nSet-Cookie: evil=1",  # session fixation on our own origin
        "/u/alice/x\n",  # trailing bare LF: why the anchor is \z, not $
        "/u/alice/x&user=bob",  # second `user` param wins in most parsers
        "/u/alice/x?next=/u/bob/",  # smuggled query
        "//evil.example.org/",  # protocol-relative open redirect
        "/auth/login",  # outside every user prefix
        "/u/alice/x y",  # raw space splits the header value
    ):
        assert not allowlist.match(path), f"{path!r} must not be reflected"


def test_negotiated_401_every_redirect_is_relative_to_the_clients_own_origin() -> None:
    """`absolute_redirect off` sits at SERVER level, not in one location.

    Left on, nginx rebuilds a relative redirect target into an absolute URL
    from `$host` plus its OWN listening port — wrong for the very topology
    `auth.allow_insecure_http` exists to serve: behind a facility TLS
    terminator, `https://facility/u/alice` becomes `http://facility:9080/u/alice/`,
    downgrading the scheme and advertising the internal port. Scoping it to the
    login handler alone would leave the `/u/<user>` bookmark 301s — a link an
    operator is far more likely to have saved — still absolute.
    """
    # Act
    nginx_conf = _render_nginx(_auth_config(["alice"]))
    content = _server_blocks(nginx_conf)[0]

    # Assert — declared before the first location, so every one inherits it
    server_preamble = _directives(content).split("location", 1)[0]
    assert "absolute_redirect off;" in server_preamble
    assert content.count("absolute_redirect off;") == 1
    assert "return 302 http" not in content
    # The bookmark redirect is the one this placement exists to cover.
    assert "return 301 /u/alice/;" in _location_body(nginx_conf, "location = /u/alice")


def test_negotiated_401_login_target_is_the_public_auth_prefix() -> None:
    """The place denied users are sent must itself be reachable without a
    session — the redirect target lives under the ungated `/auth/` prefix."""
    # Act
    nginx_conf = _render_nginx(_auth_config())
    body = _location_body(nginx_conf, "location @osprey_auth_401")

    # Assert
    assert "/auth/login" in body
    assert "    location /auth/ {" in nginx_conf
    assert "auth_request" not in _location_body(nginx_conf, "location /auth/")


def test_negotiated_401_absent_when_method_is_none() -> None:
    """Nothing to negotiate without authentication: no map, no `error_page`,
    no handler."""
    # Act
    nginx_conf = _render_nginx(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert "$osprey_auth_wants_login_page" not in nginx_conf
    assert "error_page" not in nginx_conf
    assert "@osprey_auth_401" not in nginx_conf


# ---------------------------------------------------------------------------
# Task 3.4: TLS redirect server block
# ---------------------------------------------------------------------------


def _server_blocks(nginx_conf: str) -> list[str]:
    """The top-level `server { … }` blocks, in render order.

    Split on the closing brace in column 0, which only a top-level block has
    (every nested `location` closes at an indent).
    """
    blocks = re.findall(r"^server \{\n(.*?)^\}$", nginx_conf, flags=re.DOTALL | re.MULTILINE)
    assert blocks, "no top-level server block in the rendered fragment"
    return blocks


def test_tls_redirect_off_renders_exactly_one_server_block() -> None:
    """Without TLS there is nothing to redirect to: one server, on the plain
    port, serving everything — unchanged from before TLS was renderable."""
    # Act
    nginx_conf = _render_nginx(copy.deepcopy(_MULTI_USER_CONFIG))

    # Assert
    assert len(_server_blocks(nginx_conf)) == 1
    assert "return 301 https://" not in nginx_conf
    assert "listen 9080;" in nginx_conf


def test_tls_redirect_splits_into_a_redirect_server_and_a_content_server() -> None:
    """With TLS on, the plain port and 443 become two separate servers.

    One server listening on both ports would answer the same content over
    cleartext, and any session cookie it set would cross the network in the
    clear — the exact risk TLS was enabled to remove.
    """
    # Act
    blocks = _server_blocks(_render_nginx(_tls_config()))

    # Assert
    assert len(blocks) == 2
    redirect, content = blocks
    assert "listen 9080;" in redirect
    assert "listen [::]:9080;" in redirect
    assert "listen 443 ssl;" in content
    assert "listen [::]:443 ssl;" in content
    # Neither listener answers on the other's port.
    assert "443" not in redirect
    assert "listen 9080;" not in content


def test_tls_redirect_server_serves_nothing_but_the_redirect() -> None:
    """The plain-port server has no locations, no root, and no proxying — its
    only content is the 301. Anything else served there would be served over
    cleartext."""
    # Act
    redirect = _server_blocks(_render_nginx(_tls_config()))[0]

    # Assert
    assert "return 301 https://$host$request_uri;" in redirect
    assert "location" not in redirect
    assert "root " not in redirect
    assert "proxy_pass" not in redirect
    assert "auth_request" not in redirect


def test_tls_redirect_target_is_the_requested_host_not_the_render_time_origin() -> None:
    """The 301 goes to `$host` — the name the client actually used.

    A deployment answers to more names than config knows (aliases, internal DNS
    names, a bare IP); rewriting the host at render time would bounce those
    clients to a name they may not resolve. `external_origin` is for values
    that must be fixed at render time, like an IdP-registered redirect_uri.
    """
    # Arrange
    config = _tls_config()
    fqdn = config["deploy"]["fqdn"]

    # Act
    redirect = _server_blocks(_render_nginx(config))[0]

    # Assert
    assert "https://$host$request_uri" in redirect
    assert fqdn not in redirect


def test_tls_redirect_content_server_holds_the_cert_and_every_user_route() -> None:
    """The 443 server is the sole content server: cert pair, landing page, and
    every per-user route live there, not in the redirect server."""
    # Arrange
    config = _tls_config(["alice", "bob"])

    # Act
    content = _server_blocks(_render_nginx(config))[1]

    # Assert
    assert "ssl_certificate /etc/nginx/certs/dls.crt;" in content
    assert "ssl_certificate_key /etc/nginx/certs/dls.key;" in content
    assert "location = / {" in content
    assert "location /u/alice/ {" in content
    assert "location /u/bob/ {" in content


def test_tls_redirect_leaves_the_auth_surface_only_on_the_secure_server() -> None:
    """With auth and TLS both on, the login flow and the per-user gates exist
    only behind TLS — there is no plain-http path to a login form, so no cookie
    can be issued on the cleartext origin."""
    # Arrange
    config = _auth_config()
    config["modules"]["web_terminals"]["auth"].pop("allow_insecure_http")
    config["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }

    # Act
    redirect, content = _server_blocks(_render_nginx(config))

    # Assert
    assert "location /auth/" not in redirect
    assert "_osprey_auth" not in redirect
    assert "location /auth/ {" in content
    assert "auth_request /_osprey_auth/alice;" in content
    # Count-equality across the two-server matrix: every gated location and
    # every auth target lives in the 443 server and nowhere else, so the split
    # cannot silently leave a duplicate of one on the cleartext listener.
    users = config["modules"]["web_terminals"]["users"]
    assert _directives(content).count("auth_request ") == len(users)
    assert _directives(redirect).count("auth_request ") == 0
    assert content.count("location = /_osprey_auth/") == len(users)
    assert redirect.count("location") == 0


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


# ---------------------------------------------------------------------------
# Task 4.4: the auth sidecar's compose service, and the secrets isolation
# between it and the per-user containers
# ---------------------------------------------------------------------------


def _compose(config: dict) -> dict:
    """The rendered compose overlay, parsed."""
    return yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])


def _env_names(service: dict) -> list[str]:
    """The env-var names a compose service declares inline, in rendered order."""
    return [line.split("=", 1)[0] for line in service.get("environment", [])]


def test_auth_sidecar_service_is_absent_for_method_none() -> None:
    """The default posture renders the overlay it renders today: no sidecar service,
    no auth environment, no reference to the credential file. Every deployment that
    has not opted into authentication must be untouched by this feature."""
    # Arrange
    config = copy.deepcopy(_MULTI_USER_CONFIG)

    # Act
    rendered = render_web_terminals(config)["docker-compose.web.yml"]

    # Assert
    assert set(yaml.safe_load(rendered)["services"]) == {
        "nginx",
        "web-alice",
        "web-bob",
        "web-carol",
    }
    assert "OSPREY_AUTH" not in rendered
    assert AUTH_ENV_FILENAME not in rendered


def test_auth_sidecar_service_is_the_only_service_authentication_adds() -> None:
    """Turning auth on adds exactly one service — `auth` — and changes nothing about
    the per-user set. The name is deliberately not `web-auth`: usernames render as
    `web-<user>`, so a roster user named "auth" would collide with that spelling but
    cannot collide with this one."""
    # Act
    services = set(_compose(_auth_config())["services"])

    # Assert
    assert services == {"auth", "nginx", "web-alice", "web-bob", "web-carol"}


def test_auth_sidecar_service_serves_a_single_uvicorn_bound_to_loopback() -> None:
    """The sidecar binds the loopback address on `auth.port` under host networking,
    exactly like the per-user apps: nginx shares the netns and is the only intended
    caller, so the login surface is unreachable off-host on its own port.

    Single process, not a worker pool — the revocation list and the brute-force
    throttle are in-memory, so a second worker would answer from a different view of
    both (a logged-out session still valid on whichever worker missed the logout).
    """
    # Act
    auth = _compose(_auth_config(port=9411))["services"]["auth"]

    # Assert
    assert auth["command"] == [
        "uvicorn",
        "osprey.services.auth_sidecar.app:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        "9411",
    ]
    assert "--workers" not in auth["command"]
    assert auth["network_mode"] == "host"


def test_auth_sidecar_service_image_defaults_to_the_local_build_tag() -> None:
    """Local mode: no `auth.image`, so the service names the tag the runtime build
    produces — the persona-image pattern (`<project>-<name>:local`) applied to the
    sidecar, whose project is resolve_personas()'s default `<prefix>-assistant`."""
    # Act
    auth = _compose(_auth_config())["services"]["auth"]

    # Assert
    assert auth["image"] == "dls-assistant-auth:local"


def test_auth_sidecar_service_image_pins_a_configured_registry_image() -> None:
    """Registry mode: `auth.image` is used verbatim — publishing it is the facility
    CI's job, and nothing is built on the deploy host."""
    # Arrange
    image = "git.dls.example.org:5050/physics/production/dls-auth:2026.8.1"

    # Act
    auth = _compose(_auth_config(image=image))["services"]["auth"]

    # Assert
    assert auth["image"] == image


def test_auth_sidecar_service_carries_the_roster_in_declaration_order() -> None:
    """The sidecar answers for exactly the users nginx routes to, from the same
    resolved `services` list — a user cannot exist in one and be missing from the
    other. Order is the roster's declaration order, which the login page reuses."""
    # Act
    auth = _compose(_auth_config(["carol", "alice"]))["services"]["auth"]

    # Assert
    assert f"{ENV_USERS}=carol,alice" in auth["environment"]


def test_auth_sidecar_service_environment_is_exactly_the_non_secret_settings() -> None:
    """The inline `environment:` block is pinned key for key, and every key in it is
    non-secret. This is the assertion that fails if a later task reaches for a secret
    here instead of `.env.auth`: compose interpolates `environment:` values (a stored
    `scrypt$...` hash would be eaten by `$` expansion) and, unlike an env_file, the
    values land in the committed compose artifact."""
    # Act
    auth = _compose(_auth_config())["services"]["auth"]

    # Assert
    assert _env_names(auth) == [
        "TZ",
        ENV_METHOD,
        ENV_SESSION_LIFETIME,
        ENV_TLS_ENABLED,
        ENV_EXTERNAL_ORIGIN,
        ENV_USERS,
    ]


def test_auth_sidecar_service_tls_flag_and_origin_follow_the_deployment() -> None:
    """`OSPREY_AUTH_TLS_ENABLED` is not about the sidecar's own listener (nginx
    terminates TLS; nothing is mounted here) — it decides the `Secure` attribute on
    the cookies it issues, so it must track the public origin, not the local bind."""
    # Arrange
    plain = _auth_config()
    secured = _auth_config()
    secured["modules"]["web_terminals"]["tls"] = {
        "enabled": True,
        "cert": "/etc/nginx/certs/dls.crt",
        "key": "/etc/nginx/certs/dls.key",
    }

    # Act
    plain_env = _compose(plain)["services"]["auth"]["environment"]
    secured_env = _compose(secured)["services"]["auth"]["environment"]

    # Assert
    assert f"{ENV_TLS_ENABLED}=false" in plain_env
    assert f"{ENV_EXTERNAL_ORIGIN}=http://dls-deploy.dls.example.org:9080" in plain_env
    assert f"{ENV_TLS_ENABLED}=true" in secured_env
    assert f"{ENV_EXTERNAL_ORIGIN}=https://dls-deploy.dls.example.org" in secured_env


def test_auth_sidecar_service_names_oidc_credential_variables_only_in_oidc_mode() -> None:
    """`oidc` mode adds the issuer plus the NAMES of the two variables holding the
    client credentials — one level of indirection, so the credentials themselves stay
    in `.env.auth` under whatever names the facility already uses.

    Password mode emits none of it rather than an inert empty issuer, and the whole
    OIDC group is gated on the mode: the password roster here still declares an
    `oidc_subject` left over from an earlier posture, and none of it renders.
    """
    # Arrange
    issuer = "https://sso.dls.example.org/realms/facility"
    password_roster = [{"name": "alice", "index": 0, "oidc_subject": "alice@dls.example.org"}]

    # Act
    password_env = _compose(_auth_config(password_roster))["services"]["auth"]["environment"]
    oidc_env = _compose(
        _auth_config(
            method="oidc",
            oidc={
                "issuer": issuer,
                "client_id_env": "DLS_SSO_CLIENT_ID",
                "client_secret_env": "DLS_SSO_CLIENT_SECRET",
            },
        )
    )["services"]["auth"]["environment"]

    # Assert
    assert not [line for line in password_env if "OIDC" in line]
    assert f"{ENV_OIDC_ISSUER}={issuer}" in oidc_env
    assert f"{ENV_OIDC_CLIENT_ID_ENV}=DLS_SSO_CLIENT_ID" in oidc_env
    assert f"{ENV_OIDC_CLIENT_SECRET_ENV}=DLS_SSO_CLIENT_SECRET" in oidc_env
    # The variable names are named; the credentials behind them never appear.
    assert not [line for line in oidc_env if line.startswith("DLS_SSO_CLIENT")]


def test_auth_sidecar_service_maps_each_users_oidc_subject_from_the_roster() -> None:
    """The non-secret half of the roster->identity mapping travels from the roster
    entry to the sidecar's environment under the name `env_var_suffix()` gives it —
    asserted against that function rather than a hardcoded literal, so this test
    cannot quietly disagree with the mapping that keys the password hashes.

    The value is quoted as a whole `KEY=value` scalar because a subject is typically
    an email or a DN, and a DN's `": "` would otherwise derail the YAML parse. A
    roster user who declares no subject gets no line at all, rather than an empty one
    that would map every unidentified caller onto them.
    """
    # Arrange
    config = _auth_config(
        [
            {"name": "alice", "index": 0, "oidc_subject": "cn=Alice Smith, ou=Users"},
            {"name": "dana-b", "index": 1, "oidc_subject": "dana@dls.example.org"},
            "bob",
        ],
        method="oidc",
        oidc={"issuer": "https://sso.dls.example.org"},
    )

    # Act
    auth_env = _compose(config)["services"]["auth"]["environment"]

    # Assert — the suffix is whatever env_var_suffix() says it is, including for
    # the hyphenated name, which is where a hand-rolled second mapping would drift.
    assert (
        f"{ENV_OIDC_SUBJECT_PREFIX}{env_var_suffix('alice')}=cn=Alice Smith, ou=Users" in auth_env
    )
    assert f"{ENV_OIDC_SUBJECT_PREFIX}{env_var_suffix('dana-b')}=dana@dls.example.org" in auth_env
    assert not [
        line
        for line in auth_env
        if line.startswith(f"{ENV_OIDC_SUBJECT_PREFIX}{env_var_suffix('bob')}")
    ]


def test_auth_sidecar_service_omits_the_oidc_claim_unless_the_facility_names_one() -> None:
    """No configured claim means no env line, so the sidecar applies its own
    documented default. Restating that default in the renderer would give the
    deployment two places to change it and one of them would eventually be wrong."""
    # Act
    auth_env = _compose(
        _auth_config(method="oidc", oidc={"issuer": "https://sso.dls.example.org"})
    )["services"]["auth"]["environment"]

    # Assert
    assert not [line for line in auth_env if line.startswith("OSPREY_AUTH_OIDC_CLAIM")]


def test_auth_sidecar_service_names_a_configured_non_default_oidc_claim() -> None:
    """An IdP that carries the mappable identity somewhere other than `sub` (a
    facility SSO keyed on `email` or `preferred_username`) is configured by name, and
    that name reaches the sidecar."""
    # Act
    auth_env = _compose(
        _auth_config(
            method="oidc",
            oidc={"issuer": "https://sso.dls.example.org", "claim": "preferred_username"},
        )
    )["services"]["auth"]["environment"]

    # Assert
    assert "OSPREY_AUTH_OIDC_CLAIM=preferred_username" in auth_env


def test_auth_context_reads_the_oidc_claim_and_leaves_it_none_when_unusable() -> None:
    """The parse side of the claim: configured strings come through, and anything
    unusable (absent, blank, wrong-typed) resolves to None — this module's defensive
    -read convention, and the value that makes the sidecar's default apply."""
    # Arrange
    configured = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    configured["auth"] = {"method": "oidc", "oidc": {"claim": "email"}}
    malformed = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    malformed["auth"] = {"method": "oidc", "oidc": {"claim": ["not", "a", "string"]}}
    blank = copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"]
    blank["auth"] = {"method": "oidc", "oidc": {"claim": "   "}}

    # Act / Assert
    assert _auth_tls_context(configured)["auth_oidc_claim"] == "email"
    assert _auth_tls_context(malformed)["auth_oidc_claim"] is None
    assert _auth_tls_context(blank)["auth_oidc_claim"] is None
    assert (
        _auth_tls_context(copy.deepcopy(_MULTI_USER_CONFIG)["modules"]["web_terminals"])[
            "auth_oidc_claim"
        ]
        is None
    )


def test_auth_sidecar_service_healthcheck_probes_its_own_health_route() -> None:
    """`/health` answers 200 even when the sidecar is unconfigured and refusing
    everything, so the container stays up to explain the lockout instead of
    crash-looping. Probed in-image with python: `curl` is not in an OSPREY service
    image (the nginx service's curl probe runs in the nginx image)."""
    # Act
    auth = _compose(_auth_config(port=9411))["services"]["auth"]

    # Assert
    probe = auth["healthcheck"]["test"]
    assert probe[0] == "CMD-SHELL"
    assert "http://127.0.0.1:9411/health" in probe[1]
    assert "curl" not in probe[1]


def test_auth_env_isolation_secrets_reach_the_sidecar_only_through_env_file() -> None:
    """`.env.auth` (0600, sidecar-only) is the sidecar's whole secret surface, and it
    arrives as a service-level `env_file`. That is the only path that survives the
    values intact: compose interpolates `environment:` entries, and a stored scrypt
    hash (`scrypt$16384$8$1$salt$key`) is full of `$`."""
    # Act
    rendered = render_web_terminals(_auth_config())["docker-compose.web.yml"]
    auth = yaml.safe_load(rendered)["services"]["auth"]

    # Assert
    assert auth["env_file"] == AUTH_ENV_FILENAME
    for secret_var in (ENV_SESSION_SECRET, ENV_STATE_SECRET, ENV_PW_HASH_PREFIX):
        assert secret_var not in rendered


def test_auth_env_isolation_sidecar_never_reads_env_production() -> None:
    """The inverse of the isolation rule: `.env.production` is the per-user
    containers' shared env file (provider keys, facility credentials). The login
    surface has no business reading it, and pulling it in would also drag the agent
    tier's variables into the process that mints sessions."""
    # Act
    auth = _compose(_auth_config())["services"]["auth"]

    # Assert
    assert ".env.production" not in str(auth.get("env_file"))
    assert not [name for name in _env_names(auth) if name.startswith("ANTHROPIC")]


def test_auth_env_isolation_no_web_service_carries_an_auth_variable() -> None:
    """The proposal's central isolation assertion. Every per-user container runs the
    agent; the session signing key would let any user's agent mint a valid cookie for
    every other user, which is exactly the isolation this feature establishes. So no
    `web-*` service may carry an `OSPREY_AUTH_*` variable inline or reach `.env.auth`
    through an env_file — while keeping the `.env.production` it has today."""
    # Act
    compose = _compose(_auth_config())

    # Assert
    for name, service in compose["services"].items():
        if not name.startswith("web-"):
            continue
        assert service["env_file"] == ".env.production"
        assert AUTH_ENV_FILENAME not in str(service["env_file"])
        assert not [env for env in _env_names(service) if env.startswith("OSPREY_AUTH")]


def test_render_auth_on_refuses_a_roster_name_outside_the_username_charset() -> None:
    """Defense in depth on the render path itself. The charset is enforced by lint
    (skippable with `--no-lint`) and by credential provisioning (password mode only),
    so a malformed name could reach the renderer with neither having run — and the
    name is written literally into an nginx location key and the verify subrequest's
    `?user=` value. `bob&user=alice` would render a query carrying two user
    parameters, making the authorized identity a function of how the sidecar's query
    parser breaks the tie. With auth on, that is a refusal, not a render."""
    # Arrange
    injected = _auth_config(["alice", "bob&user=alice"])
    uppercase = _auth_config(["Alice"])

    # Act / Assert
    with pytest.raises(ValueError, match="modules.web_terminals.users") as exc_info:
        render_web_terminals(injected)
    message = str(exc_info.value)
    # The offender is named, and only the offender — a message that just says
    # "a username is invalid" leaves the operator grepping a 40-user roster.
    assert "bob&user=alice" in message
    assert "'alice'" not in message
    assert "auth.method" in message
    with pytest.raises(ValueError, match="Alice"):
        render_web_terminals(uppercase)


def test_render_auth_off_still_renders_a_roster_name_outside_the_charset() -> None:
    """The gate is scoped to authentication being on. With `auth.method: none` the
    username is a routing label rather than an authorization identity, and every
    deployment rendering such a name today must keep rendering — lint still reports
    it. Tightening this would be a silent breaking change for no security gain."""
    # Arrange
    config = _config(["alice", "bob.jones"])

    # Act
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])

    # Assert
    assert "web-bob.jones" in compose["services"]


def test_render_auth_on_charset_gate_rejects_a_trailing_newline_in_a_roster_name() -> None:
    """`$` in the charset pattern also matches *before* a trailing newline, so a
    `.match()` gate would accept `"alice\\n"` — a name that renders into an nginx
    location key mid-directive. The gate uses `.fullmatch()`; this pins that, because
    the difference is invisible in every other test's inputs."""
    # Act / Assert
    with pytest.raises(ValueError, match="modules.web_terminals.users"):
        render_web_terminals(_auth_config(["alice\n"]))


def test_render_auth_on_refuses_roster_names_that_collide_on_their_env_var_suffix() -> None:
    """`env_var_suffix()` is lossy: `alice-b` and `alice_b` both key `..._ALICE_B`, so
    the pair would share ONE `OSPREY_AUTH_PW_HASH_ALICE_B` — one user's password
    opening the other's terminal, the exact isolation failure this feature exists to
    prevent.

    Both names pass the charset check, so this needs its own gate. Downstream it is
    caught only on the password-mode deploy path; in `oidc` mode it surfaces as a
    whole-deployment 503 that names neither user. Render time knows both names, so it
    says both.
    """
    # Act / Assert
    with pytest.raises(ValueError, match="env-var suffix") as exc_info:
        render_web_terminals(_auth_config(["alice-b", "alice_b"], method="oidc"))
    message = str(exc_info.value)
    assert "alice-b" in message
    assert "alice_b" in message
    assert env_var_suffix("alice-b") in message


def test_render_auth_off_still_renders_roster_names_that_collide_on_their_suffix() -> None:
    """Scoped like the charset gate: with no authentication there are no per-user
    credential variables to collide over, so a roster that renders today keeps
    rendering. Lint still reports the collision."""
    # Act
    compose = yaml.safe_load(
        render_web_terminals(_config(["alice-b", "alice_b"]))["docker-compose.web.yml"]
    )

    # Assert
    assert {"web-alice-b", "web-alice_b"} <= set(compose["services"])
